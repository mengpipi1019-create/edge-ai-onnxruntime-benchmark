# 复现说明

## 环境

原实验使用 Windows 11、Python 3.12.10、GCC 16.1.0、CMake/Ninja，以及 ONNX Runtime Windows x64 SDK 1.23.2。Python 依赖固定在 `requirements.runtime.txt`。安装 CPU PyTorch 时可使用官方 wheel 源：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.runtime.txt
.\.venv\Scripts\python.exe -m pip check
```

下载依赖和预训练权重需要网络。没有将虚拟环境、编译器、SDK、exe 或 DLL 提交到仓库。C++ 需要另行安装 [w64devkit](https://github.com/skeeto/w64devkit/releases) 或兼容的 MinGW 工具链，以及 [ONNX Runtime 1.23.2 Windows x64 SDK](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2)。仅安装 Python 的 onnxruntime 包不能替代 C++ SDK。

## 模型和推理

在仓库根目录按 README 命令导出 `models/resnet18.onnx`，再运行 Python 推理和 C++ 构建。导出脚本固定输入形状 `[1,3,224,224]` 和 opset 18，不支持动态 batch。

三端对齐用 `--results-dir results/local_alignment` 保留新结果；不要覆盖历史证据。生成器会重建 `images/s3_alignment`，不同 Pillow 版本可能改变 JPEG 字节和哈希。

## 历史性能结果

历史模型的 SHA-256 为 `c5ef909be6cda03f0f87a7354355468944e7f8840d5d2d83c344fdbf4a5d3bf6`。原 C++ SDK DLL 哈希为 `dec964ab1ee36cc9b0ae247d13b376627992fc57dec0454354017ab8fd84f1ea`。基准脚本会检查这些值，防止把不同模型或运行库混入同一次实验。

只查看历史结果时使用 `scripts/verify_published_results.py`，不需要模型和 SDK。

`run_s4_fp32_benchmark.py` 是原始协调脚本，默认会写入固定的历史目录；不要在唯一一份证据上直接重跑。若需重测，请先复制仓库到单独实验目录，按原相对路径导出模型并核对哈希、准备相同 SDK、编译 C++，再运行原脚本。导出环境不同可能产生不同模型字节；哈希不同时应建立新的协议和结果目录，不能修改原协议冒充同一次实验。

本次发布复核完成公开 CSV 重算和独立目录的 C++ 构建。Python 端重跑在导入 PyTorch 时遇到 c10.dll 初始化错误，尚未重跑完整三端对齐；五图对齐和性能结果来自 2026-08-25 的实验记录。没有重新进行完整性能测量，历史绝对延迟不是其他机器应当达到的标准。
