/**
 * 视觉大模型解析模块
 *
 * 当本地解析失败（扫描件 PDF）时，将 PDF 每页转成图片，
 * 通过 OpenAI 兼容的视觉大模型 API 逐页识别文字，拼接为 Markdown。
 */

const fs = require('node:fs');
const path = require('node:path');

const DEFAULT_DPI = 150;
const HIGH_DPI = 300;
const MIN_RESULT_CHARS = 50;
const VISION_REQUEST_TIMEOUT_MS = 120000;
const VISION_SYSTEM_PROMPT = '你是一个文档 OCR 助手。请将图片中的所有文字内容准确提取为 Markdown 格式。保留标题层级、列表、表格等结构。只输出提取的文本内容，不要添加任何解释。';

/**
 * 解析扫描件 PDF，逐页调用视觉大模型识别文字。
 *
 * @param {string} filePath - PDF 文件路径
 * @param {object} visionConfig - { base_url, api_key, model_name }
 * @param {object} options
 * @param {function} options.onProgress - 进度回调 (currentPage, totalPages)
 * @returns {Promise<string>} Markdown 文本
 */
async function parseWithVisionModel(filePath, visionConfig, options = {}) {
  const { base_url, api_key, model_name } = visionConfig;

  if (!base_url || !api_key || !model_name) {
    throw new Error('请先在设置中配置视觉大模型的 Base URL、API Key 和模型名称');
  }

  const onProgress = typeof options.onProgress === 'function' ? options.onProgress : () => {};

  // 1. PDF 转图片
  onProgress(0, 0);
  const pageImages = await renderPdfToImages(filePath, DEFAULT_DPI);
  const totalPages = pageImages.length;

  if (totalPages === 0) {
    throw new Error('PDF 文件无法渲染为图片，可能已损坏');
  }

  onProgress(0, totalPages);

  // 2. 逐页调用视觉模型
  const pageTexts = [];
  let needHighDpiRetry = false;

  for (let i = 0; i < totalPages; i++) {
    const imageBase64 = pageImages[i];
    onProgress(i + 1, totalPages);

    const text = await callVisionApi(base_url, api_key, model_name, imageBase64);
    pageTexts.push(text);

    // 检查第一页结果，如果文字太少，标记需要高 DPI 重试
    if (i === 0 && text.trim().length < MIN_RESULT_CHARS) {
      needHighDpiRetry = true;
      break;
    }
  }

  // 3. 如果第一页结果太短，用高 DPI 重试全部页面
  if (needHighDpiRetry) {
    const highDpiImages = await renderPdfToImages(filePath, HIGH_DPI);
    pageTexts.length = 0;

    for (let i = 0; i < highDpiImages.length; i++) {
      onProgress(i + 1, highDpiImages.length);
      const text = await callVisionApi(base_url, api_key, model_name, highDpiImages[i]);
      pageTexts.push(text);
    }
  }

  // 4. 拼接为 Markdown
  return pageTexts.join('\n\n---\n\n');
}

/**
 * 将 PDF 每页渲染为 base64 编码的 PNG 图片。
 *
 * @param {string} filePath
 * @param {number} dpi
 * @returns {Promise<string[]>} base64 图片数组
 */
async function renderPdfToImages(filePath, dpi) {
  const { getDocument } = require('pdfjs-dist/legacy/build/pdf.mjs');
  const { createCanvas } = require('@napi-rs/canvas');

  const buffer = fs.readFileSync(filePath);
  const loadingTask = getDocument({
    data: new Uint8Array(buffer),
    disableWorker: true,
  });

  const pdfDocument = await loadingTask.promise;
  const images = [];
  const scale = dpi / 72;

  for (let pageNum = 1; pageNum <= pdfDocument.numPages; pageNum++) {
    const page = await pdfDocument.getPage(pageNum);
    const viewport = page.getViewport({ scale });

    const canvas = createCanvas(viewport.width, viewport.height);
    const ctx = canvas.getContext('2d');

    await page.render({
      canvasContext: ctx,
      viewport,
    }).promise;

    const pngBuffer = canvas.toBuffer('image/png');
    images.push(pngBuffer.toString('base64'));
  }

  await pdfDocument.destroy();
  return images;
}

/**
 * 调用 OpenAI 兼容的视觉大模型 API。
 *
 * @param {string} baseUrl
 * @param {string} apiKey
 * @param {string} modelName
 * @param {string} imageBase64 - base64 编码的图片
 * @returns {Promise<string>} 模型返回的文字
 */
async function callVisionApi(baseUrl, apiKey, modelName, imageBase64) {
  const url = `${baseUrl.replace(/\/+$/, '')}/chat/completions`;

  const body = {
    model: modelName,
    messages: [
      { role: 'system', content: VISION_SYSTEM_PROMPT },
      {
        role: 'user',
        content: [
          {
            type: 'image_url',
            image_url: {
              url: `data:image/png;base64,${imageBase64}`,
            },
          },
          {
            type: 'text',
            text: '请提取图片中的所有文字内容，输出为 Markdown 格式。',
          },
        ],
      },
    ],
    max_tokens: 4096,
  };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), VISION_REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => '');
      throw new Error(`视觉模型 API 请求失败：HTTP ${response.status} ${errorText}`);
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content;

    if (typeof content !== 'string') {
      throw new Error('视觉模型 API 未返回有效文本内容');
    }

    return content.trim();
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('视觉模型 API 请求超时，请检查网络或换一个更快的模型');
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

module.exports = {
  parseWithVisionModel,
};
