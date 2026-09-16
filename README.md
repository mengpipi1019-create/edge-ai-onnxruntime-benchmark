# ResNet18 CPU 推理部署与验证

将 ImageNet 预训练 ResNet18 导出为 ONNX，在 PyTorch、ONNX Runtime Python 和独立 C++ 程序中运行。重点是检查三端结果是否一致、定位图像预处理差异，以及在统一条件下测量 CPU 推理延迟。

C++ 程序使用 Windows WIC 读取 JPEG，不调用 Python；支持读取固定输入张量，方便把预处理差异与推理计算差异分开比较。

## 结果

2026-08-25 在 Windows 11、Intel Core i9-14900HX 上完成以下实验：

- 五张固定一致性样本使用同一输入张量时，三端 Top-5 顺序全部一致。PyTorch 与 ORT Python 的最大 logits 绝对误差为 `8.583e-06`；同版本 ORT Python 与 C++ 的最大误差为 `0`。
- 各自读取 JPEG 时，五张样本 Top-1 一致，但输入张量和 logits 有差异。PIL/torchvision 与 WIC/自写双线性插值不是同一预处理实现；没有声称端到端逐像素一致。
- FP32、batch=1、单线程、固定逻辑 CPU 0、同一输入张量的 forward-only 测量如下。每端 5 个独立进程，每进程 50 次预热和 300 次计时，共 4,500 条数据。

| 运行方式 | 延迟 ms | 相对 PyTorch 加速比 | 重复实验中位数 CV |
| --- | ---: | ---: | ---: |
| PyTorch | 48.13255 | 1.00000x | 2.59475% |
| ORT Python | 29.87945 | 1.61089x | 2.66780% |
| ORT C++ | 29.30250 | 1.64261x | 2.14238% |

主延迟取 5 次独立实验各自中位数的中位数，不是单次最快值。计时不含图像预处理、文件读写、模型加载或结果格式化。这些结果不能推广为任意设备、模型或端到端应用的加速比。

详细说明见 [实验方法](docs/EXPERIMENTS.md)，原始数据见 [三端对齐](results/evidence/2026-08-25_s3_three_runtime_alignment/) 和 [CPU 基准](results/evidence/2026-08-25_s4_fp32_benchmark/)。旧版探索实验的性能数字不再作为本项目当前结论；历史文件可在提交记录中查阅。

## 文件入口

```text
scripts/day4_export_onnx.py              导出模型
scripts/day5_onnxruntime_inference.py    单图 Python 推理检查
scripts/generate_s3_alignment_samples.py 生成固定样本
scripts/run_s3_three_runtime_alignment.py 三端对齐
scripts/run_s4_fp32_benchmark.py         原始固定协议基准
scripts/verify_published_results.py      只读复核公开基准数据
cpp_inference/                          C++17 源码、CMake 和分类标签
images/                                两张原始图片及生成脚本所需输入
results/evidence/                      对齐结果、协议、逐次计时和汇总
docs/                                  复现说明与实验边界
```

## 先检查已发布结果

这一步只用 Python 标准库，不下载模型，不重测性能：

```powershell
python scripts/verify_published_results.py
```

脚本重新统计 4,500 条延迟、检查 15 组 worker CSV 与合并数据一致，并对照公开汇总。五图样本是两张原图和三个确定性变换，不代表真实数据集的分类准确率。

## 运行推理

验证环境为 Python 3.12.10、PyTorch 2.12.0+cpu、torchvision 0.27.0+cpu、ONNX 1.22.0、ONNX Runtime 1.23.2。版本与安装方式见 [复现说明](docs/REPRODUCE.md)。模型权重由 torchvision 从官方下载源获取，仓库不提交权重和本地工具链。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.runtime.txt
.\.venv\Scripts\python.exe scripts/day4_export_onnx.py --output models/resnet18.onnx
.\.venv\Scripts\python.exe scripts/day5_onnxruntime_inference.py --image images/cat.jpg --onnx_model models/resnet18.onnx --num_runs 1
```

单图脚本中的临时延迟输出仅用于检查运行，不属于上方正式基准。C++ 编译方法见 [C++ README](cpp_inference/README.md)。完成编译后运行：

```powershell
.\.venv\Scripts\python.exe scripts/run_s3_three_runtime_alignment.py --model models/resnet18.onnx --results-dir results/local_alignment
```

正式历史基准保留了模型、DLL 哈希和 CPU 亲和性检查。不要为通过检查而修改冻结哈希，也不要把重新测量的数据覆盖到历史目录；复现限制见 [复现说明](docs/REPRODUCE.md)。

## 范围

本项目是预训练模型部署与验证实验，没有进行模型训练、INT8 量化、真实设备部署、内存或功耗测量。Windows C++ 实现未验证 Linux 构建。依赖与来源说明见 [NOTICE](NOTICE.md)。
