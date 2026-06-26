# Day 7：Batch Size 推理性能曲线可视化

## 今日目标

本次实验目标是在 Day 6 的 batch size 推理性能测试结果基础上，将 CSV 表格数据可视化为性能曲线图，从而更直观地分析 PyTorch 与 ONNX Runtime 在不同 batch size 下的推理延迟、吞吐率和加速比变化趋势。

具体目标包括：

* 读取 Day 6 生成的 `results/day6_batch_benchmark.csv`
* 绘制 batch size 与 batch latency 的关系曲线
* 绘制 batch size 与 image latency 的关系曲线
* 绘制 batch size 与 FPS 的关系曲线
* 绘制 ONNX Runtime 相对 PyTorch 的 speedup 曲线
* 将图表保存到 `results/day7_figures`
* 总结不同 batch size 下的推理性能变化规律

---

## 项目目录

```text
edge_ai_onnx_project
├── results
│   ├── day6_batch_benchmark.csv
│   └── day7_figures
│       ├── day7_batch_latency_curve.png
│       ├── day7_image_latency_curve.png
│       ├── day7_fps_curve.png
│       └── day7_speedup_curve.png
├── scripts
│   ├── day6_batch_benchmark.py
│   └── day7_plot_curves.py
└── Day7_性能曲线可视化.md
```

---

## 运行命令

```bash
python scripts/day7_plot_curves.py
```

---

## 终端输出

```text
Loaded CSV: results\day6_batch_benchmark.csv

CSV columns:
['batch_size', 'pytorch_batch_latency_ms', 'pytorch_image_latency_ms', 'pytorch_fps', 'onnx_batch_latency_ms', 'onnx_image_latency_ms', 'onnx_fps', 'speedup', 'max_abs_diff']

Data preview:
   batch_size  pytorch_batch_latency_ms  pytorch_image_latency_ms  pytorch_fps  onnx_batch_latency_ms  onnx_image_latency_ms    onnx_fps   speedup  max_abs_diff
0           1                 12.289884                 12.289884    81.367733               5.351790               5.351790  186.853371  2.296406      0.000008
1           2                 15.215528                  7.607764   131.444666               7.048686               3.524343  283.740828  2.158633      0.000008
2           4                 23.560852                  5.890213   169.773147              11.432912               2.858228  349.867121  2.060792      0.000008
3           8                 42.772554                  5.346569   187.035827              21.404616               2.675577  373.751157  1.998286      0.000008

Saved: results\day7_figures\day7_batch_latency_curve.png
Saved: results\day7_figures\day7_image_latency_curve.png
Saved: results\day7_figures\day7_fps_curve.png
Saved: results\day7_figures\day7_speedup_curve.png

All figures saved to: results\day7_figures
```

---

## 输入数据

本次可视化使用 Day 6 生成的 benchmark CSV 文件：

```text
results/day6_batch_benchmark.csv
```

核心数据如下：

| Batch size | PyTorch batch latency ms | ONNX batch latency ms | PyTorch image latency ms | ONNX image latency ms | PyTorch FPS | ONNX FPS | Speedup | Max diff |
| ---------: | -----------------------: | --------------------: | -----------------------: | --------------------: | ----------: | -------: | ------: | -------: |
|          1 |                   12.290 |                 5.352 |                   12.290 |                 5.352 |       81.37 |   186.85 |   2.30x | 0.000008 |
|          2 |                   15.216 |                 7.049 |                    7.608 |                 3.524 |      131.44 |   283.74 |   2.16x | 0.000008 |
|          4 |                   23.561 |                11.433 |                    5.890 |                 2.858 |      169.77 |   349.87 |   2.06x | 0.000008 |
|          8 |                   42.773 |                21.405 |                    5.347 |                 2.676 |      187.04 |   373.75 |   2.00x | 0.000008 |

---

## 图 1：Batch Size vs Batch Latency

![Batch Size vs Batch Latency](results/day7_figures/day7_batch_latency_curve.png)

### 图表说明

该图展示了 batch size 增大时，PyTorch 与 ONNX Runtime 的 batch latency 变化趋势。

Batch latency 表示模型一次处理整个 batch 所需要的总时间。随着 batch size 增大，一次输入的图片数量增加，因此 batch latency 上升是正常现象。

### 结果分析

从图中可以看到：

* PyTorch 的 batch latency 从 `12.290 ms` 增加到 `42.773 ms`
* ONNX Runtime 的 batch latency 从 `5.352 ms` 增加到 `21.405 ms`
* 在所有 batch size 下，ONNX Runtime 的 batch latency 都明显低于 PyTorch

这说明 ONNX Runtime 在 CPU 推理场景下具有更低的批量推理总耗时。

---

## 图 2：Batch Size vs Image Latency

![Batch Size vs Image Latency](results/day7_figures/day7_image_latency_curve.png)

### 图表说明

该图展示了 batch size 增大时，平均到单张图片上的 image latency 变化趋势。

Image latency 的计算方式为：

```text
image latency = batch latency / batch size
```

### 结果分析

从图中可以看到：

* PyTorch 的 image latency 从 `12.290 ms/image` 降低到 `5.347 ms/image`
* ONNX Runtime 的 image latency 从 `5.352 ms/image` 降低到 `2.676 ms/image`
* 随着 batch size 增大，二者的单图平均延迟均明显下降
* ONNX Runtime 在所有 batch size 下的 image latency 都低于 PyTorch

这说明 batch 推理可以摊薄单张图片的推理开销，提高计算资源利用率。对于端侧或边缘设备中的批量图像处理任务，适当增加 batch size 有助于提升单位图片处理效率。

---

## 图 3：Batch Size vs FPS

![Batch Size vs FPS](results/day7_figures/day7_fps_curve.png)

### 图表说明

该图展示了不同 batch size 下 PyTorch 与 ONNX Runtime 的 FPS 变化趋势。

FPS 表示理论每秒可以处理的图片数量，计算方式为：

```text
FPS = batch size × 1000 / batch latency_ms
```

### 结果分析

从图中可以看到：

* PyTorch FPS 从 `81.37` 提升到 `187.04`
* ONNX Runtime FPS 从 `186.85` 提升到 `373.75`
* batch size 增大后，两种框架的 FPS 均明显提升
* ONNX Runtime 在所有 batch size 下的 FPS 都高于 PyTorch

这说明 batch 推理能够提升整体吞吐率，而 ONNX Runtime 在 CPU 上具有更强的推理吞吐能力。

---

## 图 4：ONNX Runtime Speedup over PyTorch

![ONNX Runtime Speedup over PyTorch](results/day7_figures/day7_speedup_curve.png)

### 图表说明

该图展示了 ONNX Runtime 相对 PyTorch 的推理加速比。

Speedup 的计算方式为：

```text
speedup = PyTorch batch latency / ONNX Runtime batch latency
```

### 结果分析

从图中可以看到：

* batch size = 1 时，ONNX Runtime 加速比为 `2.30x`
* batch size = 2 时，ONNX Runtime 加速比为 `2.16x`
* batch size = 4 时，ONNX Runtime 加速比为 `2.06x`
* batch size = 8 时，ONNX Runtime 加速比为 `2.00x`

ONNX Runtime 在所有 batch size 下都比 PyTorch 更快，整体保持约 `2.00x~2.30x` 的加速效果。

同时，加速比随着 batch size 增大略有下降。这说明在 batch size 较大时，PyTorch 也能更充分利用 CPU 计算资源，因此与 ONNX Runtime 的差距略有缩小。但即便如此，ONNX Runtime 仍然保持稳定优势。

---

## 数值一致性分析

本次实验中，不同 batch size 下的最大输出误差均为：

```text
Max diff = 0.000008
```

这表示 PyTorch 输出 logits 与 ONNX Runtime 输出 logits 之间的最大绝对误差非常小。

因此可以认为，ONNX Runtime 在获得更高推理速度的同时，基本保持了与 PyTorch 一致的模型输出结果。

这对于模型部署非常重要，因为部署优化不能只看速度，还必须保证推理结果正确。

---

## 综合结果分析

本次可视化实验进一步验证了 Day 6 的性能测试结论。

从延迟角度看，ONNX Runtime 在所有 batch size 下都具有更低的 batch latency 和 image latency。这说明在 CPU 推理场景中，ONNX Runtime 对 ResNet18 模型有更好的执行效率。

从吞吐角度看，随着 batch size 从 1 增大到 8，PyTorch 与 ONNX Runtime 的 FPS 均明显提升。其中 ONNX Runtime 的 FPS 从 `186.85` 提升到 `373.75`，说明批量推理能够显著提升整体吞吐率。

从加速比角度看，ONNX Runtime 在不同 batch size 下均保持约 2 倍以上的加速效果，证明 ONNX Runtime 在端侧 AI 模型部署中具有稳定的性能优势。

从数值一致性角度看，所有 batch size 下的最大误差均为 `0.000008`，说明 ONNX Runtime 推理结果与 PyTorch 基本一致，不存在明显的模型输出偏差。

---

## 今日结论

今天完成了 Day 6 benchmark 数据的可视化，将表格形式的性能测试结果转化为更直观的曲线图。

本次实验表明：

* ONNX Runtime 在所有 batch size 下都比 PyTorch 更快
* batch size 增大后，整体吞吐 FPS 明显提升
* batch size 增大后，单图平均推理延迟下降
* ONNX Runtime 在 CPU 上保持约 `2.00x~2.30x` 的稳定加速效果
* PyTorch 与 ONNX Runtime 的输出误差稳定在 `0.000008`，说明输出高度一致

通过 Day 7 的曲线图，项目结果表达从“命令行输出和 CSV 表格”进一步提升为“可视化性能分析”，更适合后续整理 README、项目报告和简历材料。

---

## 当前项目进度

```text
✅ Day 1：求职材料与项目方案
✅ Day 2：PyTorch 随机输入推理
✅ Day 3：真实图片 PyTorch 单图分类推理
✅ Day 4：PyTorch 模型导出 ONNX
✅ Day 5：ONNX Runtime 推理与 PyTorch 对比
✅ Day 6：Batch Size 推理性能测试
✅ Day 7：性能曲线可视化
```

---

## 下一步计划

后续可以继续进行：

* Day 8：README 初版与项目结构整理
* 整理项目运行流程
* 整理项目技术栈与实验结果
* 生成简历项目描述
* 准备项目面试讲解稿
* 后续扩展 MobileNetV2 对比或 ONNX 动态量化实验
