"""OpenAI 服务。"""

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List

import openai

from ..config import settings
from ..utils import prompt_manager
from ..utils.config_manager import config_manager
from ..utils.errors import AppError

logger = logging.getLogger(__name__)


class OpenAIService:
    """封装 OpenAI 模型调用与标书相关生成逻辑。"""

    def __init__(self):
        config = config_manager.load_config()
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "")
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        if not self.api_key:
            raise AppError("请先配置OpenAI API密钥", status_code=400)
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url or None,
        )

    def _chat_endpoint_url(self) -> str:
        """获取聊天完成接口地址。"""
        base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        return f"{base_url}/chat/completions"

    def _log_ai_request(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        temperature: float,
        response_format: dict | None,
    ) -> None:
        """记录 AI 请求日志。"""
        if not settings.enable_file_logging:
            return

        logger.debug(
            "AI_REQUEST %s",
            json.dumps(
                {
                    "request_id": request_id,
                    "url": self._chat_endpoint_url(),
                    "model": self.model_name,
                    "temperature": temperature,
                    "response_format": response_format,
                    "messages": messages,
                },
                ensure_ascii=False,
            ),
        )

    def _log_ai_response(self, request_id: str, content: str) -> None:
        """记录 AI 响应日志。"""
        if not settings.enable_file_logging:
            return

        logger.debug(
            "AI_RESPONSE %s",
            json.dumps(
                {
                    "request_id": request_id,
                    "url": self._chat_endpoint_url(),
                    "model": self.model_name,
                    "content": content,
                },
                ensure_ascii=False,
            ),
        )

    def _log_ai_raw_response(
        self,
        request_id: str,
        raw_chunks: list[dict[str, Any]],
        content: str,
    ) -> None:
        """记录 AI 接口原始响应日志。"""
        if not settings.enable_file_logging:
            return

        logger.debug(
            "AI_RAW_RESPONSE %s",
            json.dumps(
                {
                    "request_id": request_id,
                    "url": self._chat_endpoint_url(),
                    "model": self.model_name,
                    "raw_chunks": raw_chunks,
                    "content": content,
                },
                ensure_ascii=False,
                default=str,
            ),
        )

    def _log_ai_error(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        temperature: float,
        response_format: dict | None,
        partial_content: str,
        raw_chunks: list[dict[str, Any]],
        error: Exception,
    ) -> None:
        """记录 AI 异常日志。"""
        if not settings.enable_file_logging:
            return

        logger.debug(
            "AI_ERROR %s",
            json.dumps(
                {
                    "request_id": request_id,
                    "url": self._chat_endpoint_url(),
                    "model": self.model_name,
                    "temperature": temperature,
                    "response_format": response_format,
                    "messages": messages,
                    "partial_content": partial_content,
                    "raw_chunks": raw_chunks,
                    "error": str(error),
                },
                ensure_ascii=False,
                default=str,
            ),
        )

    @staticmethod
    def _dump_chunk(chunk: Any) -> dict[str, Any]:
        """序列化 OpenAI SDK 返回的 chunk。"""
        if hasattr(chunk, "model_dump"):
            return chunk.model_dump(mode="json")
        return {"raw": str(chunk)}

    async def get_available_models(self) -> List[str]:
        """获取可用模型列表。"""
        try:
            models = await self.client.models.list()
        except Exception as exc:
            raise AppError(f"获取模型列表失败: {exc}", status_code=502) from exc

        chat_models: list[str] = []
        for model in models.data:
            model_id = model.id.lower()
            if any(
                keyword in model_id
                for keyword in ["gpt", "claude", "chat", "llama", "qwen", "deepseek"]
            ):
                chat_models.append(model.id)
        return sorted(set(chat_models))

    async def stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        response_format: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """流式调用聊天完成接口。"""
        request_id = uuid.uuid4().hex
        parts: list[str] = []
        raw_chunks: list[dict[str, Any]] = []
        self._log_ai_request(request_id, messages, temperature, response_format)

        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=True,
                **(
                    {"response_format": response_format}
                    if response_format is not None
                    else {}
                ),
            )
        except Exception as exc:
            self._log_ai_error(
                request_id,
                messages,
                temperature,
                response_format,
                "",
                raw_chunks,
                exc,
            )
            raise AppError(f"模型调用失败: {exc}", status_code=502) from exc

        try:
            async for chunk in stream:
                raw_chunks.append(self._dump_chunk(chunk))
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content is not None:
                    parts.append(content)
                    yield content
        except Exception as exc:
            self._log_ai_error(
                request_id,
                messages,
                temperature,
                response_format,
                "".join(parts),
                raw_chunks,
                exc,
            )
            raise AppError(f"模型调用失败: {exc}", status_code=502) from exc

        self._log_ai_response(request_id, "".join(parts))
        self._log_ai_raw_response(request_id, raw_chunks, "".join(parts))

    async def collect_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        response_format: dict | None = None,
    ) -> str:
        """收集流式输出并拼接为完整文本。"""
        parts: list[str] = []
        async for chunk in self.stream_chat_completion(
            messages,
            temperature=temperature,
            response_format=response_format,
        ):
            parts.append(chunk)
        return "".join(parts)

    async def generate_outline(
        self,
        overview: str,
        requirements: str,
        uploaded_expand: bool = False,
        old_outline: str | None = None,
    ) -> Dict[str, Any]:
        """生成目录结构。"""
        if uploaded_expand:
            messages = prompt_manager.generate_outline_with_old_prompt(
                overview,
                requirements,
                old_outline,
            )
        else:
            messages = prompt_manager.generate_outline_prompt(overview, requirements)

        return await self._collect_json_response(
            messages=messages,
            temperature=0.7,
        )

    async def generate_expand_outline(self, file_content: str) -> Dict[str, Any]:
        """从已有技术方案中提取目录结构。"""
        return await self._collect_json_response(
            messages=prompt_manager.build_expand_outline_messages(file_content),
            temperature=0.7,
        )

    async def stream_chapter_content(
        self,
        chapter: Dict[str, Any],
        parent_chapters: list[dict[str, Any]] | None = None,
        sibling_chapters: list[dict[str, Any]] | None = None,
        project_overview: str = "",
    ) -> AsyncGenerator[str, None]:
        """流式生成单章节内容。"""
        messages = prompt_manager.build_chapter_content_messages(
            chapter=chapter,
            parent_chapters=parent_chapters,
            sibling_chapters=sibling_chapters,
            project_overview=project_overview,
        )
        async for chunk in self.stream_chat_completion(messages, temperature=0.7):
            yield chunk

    async def generate_chapter_content(
        self,
        chapter: Dict[str, Any],
        parent_chapters: list[dict[str, Any]] | None = None,
        sibling_chapters: list[dict[str, Any]] | None = None,
        project_overview: str = "",
    ) -> str:
        """生成单章节完整正文。"""
        return await self.collect_chat_completion(
            prompt_manager.build_chapter_content_messages(
                chapter=chapter,
                parent_chapters=parent_chapters,
                sibling_chapters=sibling_chapters,
                project_overview=project_overview,
            ),
            temperature=0.7,
        )

    async def _collect_json_response(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """收集并校验 JSON 响应。"""
        content = await self.collect_chat_completion(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("模型返回非法 JSON: %s", content)
            raise AppError("模型返回的目录数据格式无效", status_code=502) from exc
