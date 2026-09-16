# C++ ResNet18 推理

Windows 上的独立 C++17 程序。用 WIC 读取 JPEG，用 ONNX Runtime CPU 执行模型，输出分类结果；也可以直接读取 float32 输入张量。

## 构建

需要 MinGW GCC、CMake、Ninja 和 ONNX Runtime Windows x64 SDK 1.23.2。将编译工具的 bin 目录加入当前终端 PATH，从仓库根目录执行：

```powershell
cmake -S cpp_inference -B cpp_inference/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DORT_ROOT="C:/tools/onnxruntime-win-x64-1.23.2"
cmake --build cpp_inference/build
.\cpp_inference\build\resnet18_ort_cpp.exe --model models/resnet18.onnx --image images/cat.jpg --labels cpp_inference/assets/imagenet_classes.txt
```

`ORT_ROOT` 必须改为实际 SDK 路径。CMake 会把 `onnxruntime.dll` 复制到 exe 同目录。若 MinGW 运行库不在 PATH 中，运行 exe 时也需保留编译器 bin 路径。当前 CMake 参数面向 MinGW，未验证 MSVC 或 Linux。

## 输入输出

- `--image`：WIC 解码 RGB，短边缩放至 256，中心裁剪 224×224，再按 ImageNet 均值和标准差归一化。
- `--input-bin`：读取恰好 150528 个小端 float32 数值，即 `[1,3,224,224]`，跳过图片预处理。
- `--dump-input`：保存图片模式实际输入张量。
- `--dump-logits`：保存 1000 个 float32 输出值。

`--image` 与 `--input-bin` 二选一；错误字节长度会返回非零退出码。WIC/双线性预处理与 PIL/torchvision 不完全相同，端到端结果差异见 [实验说明](../docs/EXPERIMENTS.md)。

## 基准模式

`--warmup 50 --runs 300 --repeat-id N --run-order N --expected-top1 285 --benchmark-output worker.csv` 启用固定协议工作进程。它固定逻辑 CPU 0、使用单线程并仅计时 Session::Run。通常由 Python 协调脚本调用，不应与图片端到端计时混用。
