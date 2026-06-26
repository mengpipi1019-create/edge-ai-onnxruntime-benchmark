# 基于 PyTorch 与 ONNX Runtime 的轻量化图像分类模型部署与推理优化系统

## 1. 项目简介

本项目实现了一个从 PyTorch 模型推理、ONNX 模型导出、ONNX Runtime 部署推理到性能测试与可视化分析的完整端侧 AI 推理流程。

项目以 ImageNet 预训练 ResNet18 为基础模型，完成真实图片分类推理，并对比 PyTorch 与 ONNX Runtime 在 CPU 环境下的推理结果一致性、平均延迟、FPS 和不同 batch size 下的性能变化。

本项目的目标是验证 ONNX Runtime 在端侧 AI / 边缘计算场景中的部署价值，并形成一个可复现、可展示、可扩展的轻量化模型部署实验框架。

---

## 2. 技术栈

* Python
* PyTorch
* torchvision
* ONNX
* ONNX Runtime
* NumPy
* Pandas
* Matplotlib
* Pillow
* OpenCV

---

## 3. 项目功能

本项目目前完成了以下功能：

* PyTorch ResNet18 随机输入推理
* 真实图片输入与图像预处理
* ImageNet 预训练模型单图分类推理
* PyTorch 模型导出为 ONNX 格式
* ONNX Runtime 加载 ONNX 模型并完成推理
* PyTorch 与 ONNX Runtime Top-K 分类结果对比
* PyTorch 与 ONNX Runtime 输出 logits 数值误差对比
* 不同 batch size 下的推理延迟、FPS 和加速比测试
* 性能测试结果保存为 CSV
* 性能曲线图可视化

---

## 4. 项目结构

```text
edge_ai_onnx_project
├── images
│   ├── cat.jpg
│   └── test.jpg
├── models
│   ├── resnet18_imagenet.onnx
│   └── resnet18_imagenet_dynamic.onnx
├── results
│   ├── day6_batch_benchmark.csv
│   └── day7_figures
│       ├── day7_batch_latency_curve.png
│       ├── day7_image_latency_curve.png
│       ├── day7_fps_curve.png
│       └── day7_speedup_curve.png
├── scripts
│   ├── day2_pytorch_inference.py
│   ├── day3_real_image_inference.py
│   ├── day3_simple_inference.py
│   ├── day4_export_onnx.py
│   ├── day5_onnxruntime_inference.py
│   ├── day6_batch_benchmark.py
│   └── day7_plot_curves.py
├── README.md
└── requirements.txt
```

---

## 5. 环境配置

### 创建 Conda 环境

```bash
conda create -n edge_ai python=3.10 -y
conda activate edge_ai
```

### 安装依赖

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib pillow opencv-python
pip install onnx onnxruntime onnxscript
```

或者使用项目中的依赖文件：

```bash
pip install -r requirements.txt
```

---

## 6. 运行流程

### 6.1 PyTorch 随机输入推理

```bash
python scripts/day2_pytorch_inference.py
```

该脚本用于验证 PyTorch 模型加载、输入构造、前向推理和延迟统计流程。

---

### 6.2 真实图片 PyTorch 推理

```bash
python scripts/day3_real_image_inference.py --image images/cat.jpg
```

该脚本使用 ImageNet 预训练 ResNet18 对真实图片进行分类，并输出 Top-K 分类结果和推理延迟。

---

### 6.3 PyTorch 模型导出 ONNX

```bash
python scripts/day4_export_onnx.py
```

该脚本将 PyTorch ResNet18 模型导出为 ONNX 格式，并使用 `onnx.checker.check_model` 检查模型是否合法。

输出文件：

```text
models/resnet18_imagenet.onnx
```

---

### 6.4 ONNX Runtime 推理与 PyTorch 对比

```bash
python scripts/day5_onnxruntime_inference.py --image images/cat.jpg
python scripts/day5_onnxruntime_inference.py --image images/test.jpg
```

该脚本对比 PyTorch 与 ONNX Runtime 在真实图片上的 Top-5 分类结果、logits 数值误差、平均推理延迟和 FPS。

---

### 6.5 Batch Size 推理性能测试

```bash
python scripts/day6_batch_benchmark.py --image images/cat.jpg
```

该脚本导出支持动态 batch size 的 ONNX 模型，并测试 batch size = 1、2、4、8 时 PyTorch 与 ONNX Runtime 的推理性能。

输出文件：

```text
models/resnet18_imagenet_dynamic.onnx
results/day6_batch_benchmark.csv
```

---

### 6.6 性能曲线可视化

```bash
python scripts/day7_plot_curves.py
```

该脚本读取 Day 6 生成的 CSV 文件，并绘制性能曲线图。

输出图片：

```text
results/day7_figures/day7_batch_latency_curve.png
results/day7_figures/day7_image_latency_curve.png
results/day7_figures/day7_fps_curve.png
results/day7_figures/day7_speedup_curve.png
```

---

## 7. 实验结果

### 7.1 PyTorch 与 ONNX Runtime 单图推理对比

在 `cat.jpg` 上：

| 框架           | Top-1 结果     | 平均延迟 ms |    FPS |
| ------------ | ------------ | ------: | -----: |
| PyTorch      | Egyptian cat |  12.930 |  77.34 |
| ONNX Runtime | Egyptian cat |   4.851 | 206.16 |

ONNX Runtime 相比 PyTorch 加速约：

```text
2.67x
```

在 `test.jpg` 上：

| 框架           | Top-1 结果   | 平均延迟 ms |    FPS |
| ------------ | ---------- | ------: | -----: |
| PyTorch      | Pomeranian |  11.021 |  90.73 |
| ONNX Runtime | Pomeranian |   4.601 | 217.34 |

ONNX Runtime 相比 PyTorch 加速约：

```text
2.40x
```

两张图片上 PyTorch 与 ONNX Runtime 的 Top-5 分类结果完全一致，logits 最大绝对误差约为 `1e-5`，说明 ONNX 导出后的模型输出与 PyTorch 基本一致。

---

### 7.2 Batch Size 性能测试结果

| Batch size | PyTorch batch latency ms | ONNX batch latency ms | PyTorch image latency ms | ONNX image latency ms | PyTorch FPS | ONNX FPS | Speedup | Max diff |
| ---------: | -----------------------: | --------------------: | -----------------------: | --------------------: | ----------: | -------: | ------: | -------: |
|          1 |                   12.290 |                 5.352 |                   12.290 |                 5.352 |       81.37 |   186.85 |   2.30x | 0.000008 |
|          2 |                   15.216 |                 7.049 |                    7.608 |                 3.524 |      131.44 |   283.74 |   2.16x | 0.000008 |
|          4 |                   23.561 |                11.433 |                    5.890 |                 2.858 |      169.77 |   349.87 |   2.06x | 0.000008 |
|          8 |                   42.773 |                21.405 |                    5.347 |                 2.676 |      187.04 |   373.75 |   2.00x | 0.000008 |

实验结果表明，ONNX Runtime 在所有 batch size 下均明显快于 PyTorch，并保持约 `2.00x~2.30x` 的 CPU 推理加速效果。

---

## 8. 性能曲线

### Batch Size vs Batch Latency

![Batch Size vs Batch Latency](results/day7_figures/day7_batch_latency_curve.png)

### Batch Size vs Image Latency

![Batch Size vs Image Latency](results/day7_figures/day7_image_latency_curve.png)

### Batch Size vs FPS

![Batch Size vs FPS](results/day7_figures/day7_fps_curve.png)

### ONNX Runtime Speedup over PyTorch

![ONNX Runtime Speedup over PyTorch](results/day7_figures/day7_speedup_curve.png)

---

## 9. 结果分析

从延迟角度看，ONNX Runtime 在所有 batch size 下都具有更低的 batch latency 和 image latency。例如在 batch size = 1 时，PyTorch batch latency 为 `12.290 ms`，ONNX Runtime 为 `5.352 ms`；在 batch size = 8 时，PyTorch batch latency 为 `42.773 ms`，ONNX Runtime 为 `21.405 ms`。

从吞吐角度看，随着 batch size 增大，PyTorch 与 ONNX Runtime 的 FPS 均明显提升。其中 ONNX Runtime FPS 从 `186.85` 提升到 `373.75`，说明批量推理能够显著提升整体吞吐率。

从加速比角度看，ONNX Runtime 在 batch size = 1、2、4、8 下分别实现了 `2.30x`、`2.16x`、`2.06x` 和 `2.00x` 的加速，说明 ONNX Runtime 在 CPU 推理场景下具有稳定的性能优势。

从输出一致性角度看，所有 batch size 下 PyTorch 与 ONNX Runtime 的最大 logits 误差均为 `0.000008`，说明 ONNX Runtime 推理结果与 PyTorch 基本一致，不存在明显数值偏差。

---

## 10. 项目结论

本项目完成了从 PyTorch 模型推理到 ONNX Runtime 部署推理的完整最小闭环。

实验结果表明：

* ONNX Runtime 可以正确加载 PyTorch 导出的 ONNX 模型
* ONNX Runtime 与 PyTorch 的 Top-K 分类结果保持一致
* ONNX Runtime 与 PyTorch 的输出 logits 误差约为 `1e-5`
* ONNX Runtime 在 CPU 上相比 PyTorch 实现约 `2.00x~2.67x` 的推理加速
* batch size 增大可以提升整体 FPS，并降低单图平均推理延迟
* ONNX Runtime 适合作为端侧 AI / 边缘计算场景中的模型部署和推理优化工具

---

## 11. 后续优化方向

后续可以继续扩展：

* 对比 ResNet18 与 MobileNetV2 的模型大小、延迟和 FPS
* 增加 ONNX 动态量化实验
* 测试 batch size 对内存占用的影响
* 增加多图片批量推理功能
* 将推理流程封装为命令行工具
* 在 Linux 或边缘设备上进行部署测试
