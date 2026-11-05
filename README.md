# Zyll 相册端侧 AI 模型

这里只存放 Zyll 相册按需下载的模型文件，不含 App 代码。

- 模型：[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) RN50（MIT 许可），导出为 int8 ONNX
- 下载：Release `ai-models-v1`（`cnclip-rn50-img.onnx`、`cnclip-rn50-txt.onnx`、`manifest.json`）
- 导出脚本：`tools/ai-models/export_cnclip.py`，由 `.github/workflows/ai-models.yml` 自动运行并发布

## 修图模型（魔术橡皮擦）

- Release `editor-models-v1`：`editor-catalog.json` + 各模型 `.onnx`，用户在编辑页自己选一个下载
- [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN)（MIT，Picsart AI Research）：约 28 MB，快
- [LaMa](https://github.com/advimman/lama)（Apache-2.0，Samsung AI Center）：约 200 MB，大块区域更自然
- 脚本：`tools/editor-models/fetch_editor_models.py` 下载后先做功能校验（真擦一块再看结果）才发布，由 `.github/workflows/editor-models.yml` 运行
- 去反光暂时没有许可干净、能在手机上跑的现成模型，App 里保留“导入模型”
