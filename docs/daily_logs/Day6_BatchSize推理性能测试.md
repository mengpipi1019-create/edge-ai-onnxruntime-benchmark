# Day 6：Batch Size 推理性能测试

## 今日目标

本次实验目标是在 Day 5 单张图片推理对比的基础上，进一步测试不同 batch size 下 PyTorch 与 ONNX Runtime 的 CPU 推理性能。

具体目标包括：

* 导出支持动态 batch size 的 ResNet18 ONNX 模型
* 构造 batch size = 1、2、4、8 的输入数据
* 分别测试 PyTorch 和 ONNX Runtime 的推理性能
* 对比 batch latency、image latency、FPS 和 ONNX Runtime 加速比
* 检查 PyTorch 与 ONNX Runtime 输出之间的数值差异
* 将测试结果保存为 CSV 文件

---

## 项目目录

```text
edge_ai_onnx_project
├── images
│   └── cat.jpg
├── models
│   ├── resnet18_imagenet.onnx
│   └── resnet18_imagenet_dynamic.onnx
├── results
│   └── day6_batch_benchmark.csv
├── scripts
│   ├── day4_export_onnx.py
│   ├── day5_onnxruntime_inference.py
│   └── day6_batch_benchmark.py
└── Day6_BatchSize推理性能测试.md
```

---

## 运行环境

```text
Python: 3.10.20
PyTorch: 2.12.0+cpu
torchvision: 0.27.0+cpu
ONNX Runtime: 1.23.2
Device: CPU
```

---

## 运行命令

```bash
python scripts/day6_batch_benchmark.py --image images/cat.jpg
```

---

## 输出文件

本次实验生成了两个关键文件：

```text
models/resnet18_imagenet_dynamic.onnx
results/day6_batch_benchmark.csv
```

其中：

* `resnet18_imagenet_dynamic.onnx` 是支持动态 batch size 的 ONNX 模型
* `day6_batch_benchmark.csv` 保存了不同 batch size 下的性能测试结果

---

## 动态 Batch ONNX 模型导出结果

```text
Exporting dynamic-batch ONNX model...
ONNX output path: models\resnet18_imagenet_dynamic.onnx
Dynamic ONNX export done.
ONNX file size: 44.58 MB
ONNX model check: PASSED
```

本次导出的 ONNX 模型文件大小为 `44.58 MB`，与 Day 4 导出的固定 batch 模型大小一致，说明模型结构和权重均已正常保存。

同时，`ONNX model check: PASSED` 表明该 ONNX 模型格式合法，可以被 ONNX Runtime 正常加载。

---

## 输入与测试设置

输入图片：

```text
images/cat.jpg
```

ONNX 模型：

```text
models/resnet18_imagenet_dynamic.onnx
```

ONNX 输入输出名称：

```text
ONNX input name: input
ONNX output name: output
```

测试 batch size：

```text
[1, 2, 4, 8]
```

测试设置：

```text
Warmup runs: 5
Timed runs: 50
```

其中，warmup 用于减少首次推理、缓存初始化、线程调度等因素带来的偶然影响；timed runs 表示正式计时推理次数，本次每个 batch size 下重复测试 50 次后取平均值。

---

## 实验结果

| Batch size | PyTorch batch latency ms | ONNX batch latency ms | PyTorch FPS | ONNX FPS | Speedup | Max diff |
| ---------: | -----------------------: | --------------------: | ----------: | -------: | ------: | -------: |
|          1 |                   12.290 |                 5.352 |       81.37 |   186.85 |   2.30x | 0.000008 |
|          2 |                   15.216 |                 7.049 |      131.44 |   283.74 |   2.16x | 0.000008 |
|          4 |                   23.561 |                11.433 |      169.77 |   349.87 |   2.06x | 0.000008 |
|          8 |                   42.773 |                21.405 |      187.04 |   373.75 |   2.00x | 0.000008 |

---

## 指标解释

### Batch latency

Batch latency 表示模型一次处理整个 batch 所需要的总时间。

例如 batch size = 8 时：

```text
PyTorch batch latency = 42.773 ms
ONNX Runtime batch latency = 21.405 ms
```

这表示 PyTorch 一次处理 8 张图片平均需要 `42.773 ms`，ONNX Runtime 一次处理 8 张图片平均需要 `21.405 ms`。

---

### Image latency

Image latency 表示平均到单张图片上的推理耗时，计算方式为：

```text
image latency = batch latency / batch size
```

例如 batch size = 8 时：

```text
PyTorch image latency = 42.773 / 8 = 5.347 ms
ONNX Runtime image latency = 21.405 / 8 = 2.676 ms
```

说明 batch size 增大后，虽然一次处理整个 batch 的总耗时会上升，但平均到单张图片的耗时会下降。

---

### FPS

FPS 表示理论每秒可以处理的图片数量，计算方式为：

```text
FPS = batch size × 1000 / batch latency_ms
```

例如 batch size = 8 时：

```text
PyTorch FPS = 187.04
ONNX Runtime FPS = 373.75
```

说明在 CPU 上，ONNX Runtime 的批量推理吞吐明显高于 PyTorch。

---

### Max diff

Max diff 表示 PyTorch 输出 logits 与 ONNX Runtime 输出 logits 之间的最大绝对误差。

本次实验中，所有 batch size 下的最大误差均为：

```text
0.000008
```

该误差非常小，说明 ONNX Runtime 的推理结果与 PyTorch 基本一致，模型导出没有造成明显数值偏差。

---

## 结果分析

从延迟表现看，ONNX Runtime 在所有 batch size 下均明显快于 PyTorch。

当 batch size = 1 时：

```text
PyTorch batch latency = 12.290 ms
ONNX Runtime batch latency = 5.352 ms
ONNX Runtime speedup = 2.30x
```

当 batch size = 8 时：

```text
PyTorch batch latency = 42.773 ms
ONNX Runtime batch latency = 21.405 ms
ONNX Runtime speedup = 2.00x
```

可以看到，随着 batch size 增大，PyTorch 和 ONNX Runtime 的 batch latency 都会上升，但 ONNX Runtime 始终保持约 `2.00x~2.30x` 的加速效果。

从吞吐性能看，batch size 增大后，PyTorch 和 ONNX Runtime 的 FPS 都有明显提升：

```text
PyTorch FPS: 81.37 → 187.04
ONNX Runtime FPS: 186.85 → 373.75
```

这说明批量推理能够提升整体吞吐率，适合多张图片同时处理的场景。

从单图平均耗时看，batch size 增大后，平均到每张图片的推理时间下降。例如 ONNX Runtime 从 batch size = 1 时的 `5.352 ms/image`，下降到 batch size = 8 时的 `2.676 ms/image`。这说明 batch 推理可以更充分地利用 CPU 计算资源。

从输出一致性看，所有 batch size 下 PyTorch 与 ONNX Runtime 的最大 logits 差异均为 `0.000008`，说明动态 batch ONNX 模型在不同 batch size 下均能保持稳定、正确的输出结果。

---

## 今日结论

今天完成了 batch size 推理性能测试，将项目从单张图片推理对比扩展到不同 batch size 下的系统性性能评估。

本次实验表明：

* 动态 batch ONNX 模型导出成功，并通过 ONNX 格式检查
* ONNX Runtime 在 batch size = 1、2、4、8 下均可正常推理
* ONNX Runtime 在所有 batch size 下均明显快于 PyTorch
* ONNX Runtime 在 CPU 上实现了约 `2.00x~2.30x` 的推理加速
* 随着 batch size 增大，整体 FPS 明显提升
* PyTorch 与 ONNX Runtime 的输出误差稳定在 `0.000008`，说明推理结果高度一致

至此，项目已经完成了从单图推理到 batch size 性能评估的扩展，具备了更完整的端侧 AI 推理性能分析基础。

---

## 当前项目进度

```text
✅ Day 1：求职材料与项目方案
✅ Day 2：PyTorch 随机输入推理
✅ Day 3：真实图片 PyTorch 单图分类推理
✅ Day 4：PyTorch 模型导出 ONNX
✅ Day 5：ONNX Runtime 推理与 PyTorch 对比
✅ Day 6：Batch Size 推理性能测试
```

---

## 下一步计划

后续可以继续进行：

* 绘制 batch size 与 FPS / latency 的性能曲线
* 对比 ResNet18 与 MobileNetV2 的模型大小和推理速度
* 进行 ONNX 动态量化实验
* 生成项目 README
* 将项目整理为简历中的工程项目经历
