const { ipcMain } = require('electron');

function registerAiIpc({ aiService }) {
  ipcMain.handle('ai:chat', (_event, request) => aiService.chat(request));
  ipcMain.handle('ai:request-json', (_event, request) => aiService.requestJson(request));
  ipcMain.handle('ai:test-image-model', (_event, config) => aiService.testImageModel(config));
  ipcMain.handle('ai:list-vision-models', (_event, visionConfig) => aiService.listVisionModels(visionConfig));
  ipcMain.handle('ai:test-vision-model', (_event, visionConfig) => aiService.testVisionModel(visionConfig));
}

module.exports = {
  registerAiIpc,
};
