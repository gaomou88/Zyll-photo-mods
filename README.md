# Zyll 相册端侧 AI 模型

这里只存放 Zyll 相册按需下载的模型文件，不含 App 代码。

- 模型：[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) RN50（MIT 许可），导出为 int8 ONNX
- 下载：Release `ai-models-v1`（`cnclip-rn50-img.onnx`、`cnclip-rn50-txt.onnx`、`manifest.json`）
- 导出脚本：`tools/ai-models/export_cnclip.py`，由 `.github/workflows/ai-models.yml` 自动运行并发布
