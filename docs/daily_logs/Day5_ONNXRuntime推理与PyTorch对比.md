# Day 5：ONNX Runtime 推理与 PyTorch 对比

## 今日目标

本次实验目标是使用 ONNX Runtime 加载 Day 4 导出的 ResNet18 ONNX 模型，并与 PyTorch 原模型在真实图片分类任务上的推理结果和推理性能进行对比。

具体目标包括：

* 使用 ONNX Runtime 加载 `models/resnet18_imagenet.onnx`
* 使用真实图片完成 ONNX Runtime 推理
* 对比 PyTorch 与 ONNX Runtime 的 Top-5 分类结果
* 对比 PyTorch 与 ONNX Runtime 的平均推理延迟和 FPS
* 检查 PyTorch 输出与 ONNX Runtime 输出之间的数值差异

---

## 项目目录

```text
edge_ai_onnx_project
├── images
│   ├── cat.jpg
│   └── test.jpg
├── models
│   └── resnet18_imagenet.onnx
├── scripts
│   ├── day3_real_image_inference.py
│   ├── day4_export_onnx.py
│   └── day5_onnxruntime_inference.py
└── Day5_ONNXRuntime推理与PyTorch对比.md
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

分别对猫图和狗图进行推理：

```bash
python scripts/day5_onnxruntime_inference.py --image images/cat.jpg
python scripts/day5_onnxruntime_inference.py --image images/test.jpg
```

---

## 输入与模型信息

ONNX 模型路径：

```text
models/resnet18_imagenet.onnx
```

输入图片路径：

```text
images/cat.jpg
images/test.jpg
```

模型输入张量形状：

```text
torch.Size([1, 3, 224, 224])
```

其中：

* `1` 表示 batch size 为 1
* `3` 表示 RGB 三通道
* `224 × 224` 表示 ResNet18 的标准输入尺寸

ONNX Runtime 加载到的输入输出名称：

```text
Input name: input
Output name: output
```

---

## cat.jpg 运行结果

### PyTorch Top-5 predictions

```text
1. Egyptian cat | prob: 0.6455
2. tabby | prob: 0.2016
3. tiger cat | prob: 0.1440
4. lynx | prob: 0.0030
5. sunscreen | prob: 0.0006
```

### ONNX Runtime Top-5 predictions

```text
1. Egyptian cat | prob: 0.6455
2. tabby | prob: 0.2016
3. tiger cat | prob: 0.1440
4. lynx | prob: 0.0030
5. sunscreen | prob: 0.0006
```

### 输出误差

```text
Max absolute difference between PyTorch and ONNX logits: 0.000008
```

### 性能对比

```text
PyTorch average latency: 12.930 ms | FPS: 77.34
ONNX Runtime average latency: 4.851 ms | FPS: 206.16
ONNX Runtime speedup over PyTorch: 2.67x
```

---

## test.jpg 运行结果

### PyTorch Top-5 predictions

```text
1. Pomeranian | prob: 0.8993
2. Chihuahua | prob: 0.0274
3. Pekinese | prob: 0.0267
4. Samoyed | prob: 0.0208
5. chow | prob: 0.0085
```

### ONNX Runtime Top-5 predictions

```text
1. Pomeranian | prob: 0.8993
2. Chihuahua | prob: 0.0274
3. Pekinese | prob: 0.0267
4. Samoyed | prob: 0.0208
5. chow | prob: 0.0085
```

### 输出误差

```text
Max absolute difference between PyTorch and ONNX logits: 0.000007
```

### 性能对比

```text
PyTorch average latency: 11.021 ms | FPS: 90.73
ONNX Runtime average latency: 4.601 ms | FPS: 217.34
ONNX Runtime speedup over PyTorch: 2.40x
```

---

## 汇总表格

| 图片       | 框架           | Top-1 结果     | 平均延迟 ms |    FPS |   加速比 |
| -------- | ------------ | ------------ | ------: | -----: | ----: |
| cat.jpg  | PyTorch      | Egyptian cat |  12.930 |  77.34 | 1.00x |
| cat.jpg  | ONNX Runtime | Egyptian cat |   4.851 | 206.16 | 2.67x |
| test.jpg | PyTorch      | Pomeranian   |  11.021 |  90.73 | 1.00x |
| test.jpg | ONNX Runtime | Pomeranian   |   4.601 | 217.34 | 2.40x |

---

## 结果分析

本次实验中，PyTorch 与 ONNX Runtime 在两张真实图片上的 Top-5 分类结果完全一致。对于 `cat.jpg`，两者均将 Top-1 预测为 `Egyptian cat`；对于 `test.jpg`，两者均将 Top-1 预测为 `Pomeranian`。

从数值一致性来看，`cat.jpg` 的 logits 最大绝对误差为 `0.000008`，`test.jpg` 的 logits 最大绝对误差为 `0.000007`。误差量级约为 `1e-5`，说明 PyTorch 模型导出为 ONNX 后，ONNX Runtime 推理结果与原 PyTorch 模型基本一致。

从推理性能来看，ONNX Runtime 在 CPU 上明显快于 PyTorch。对于 `cat.jpg`，ONNX Runtime 平均延迟为 `4.851 ms`，相比 PyTorch 的 `12.930 ms` 实现了 `2.67x` 加速；对于 `test.jpg`，ONNX Runtime 平均延迟为 `4.601 ms`，相比 PyTorch 的 `11.021 ms` 实现了 `2.40x` 加速。

这说明 ONNX Runtime 在 CPU 推理场景下能够有效降低模型推理延迟，并提升 FPS，适合作为端侧 AI 部署和推理优化的基础工具。

---

## 关键代码理解

### 1. PyTorch Tensor 转 NumPy

ONNX Runtime 不直接接收 PyTorch Tensor，而是接收 NumPy array，因此需要将输入从 PyTorch Tensor 转换为 NumPy 格式：

```python
input_numpy = input_batch.cpu().numpy().astype(np.float32)
```

这一步表示：

```text
PyTorch Tensor → CPU → NumPy array → float32
```

### 2. 创建 ONNX Runtime 推理会话

```python
ort_session = ort.InferenceSession(
    str(onnx_model_path),
    providers=["CPUExecutionProvider"]
)
```

这一步用于加载 ONNX 模型，并指定使用 CPU 作为推理后端。

### 3. 获取 ONNX 输入输出名称

```python
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
```

Day 4 导出 ONNX 时已经设置了输入名和输出名：

```text
input
output
```

Day 5 使用 ONNX Runtime 推理时，需要根据这些名称传入输入并获取输出。

### 4. ONNX Runtime 推理

```python
onnx_outputs = ort_session.run([output_name], {input_name: input_numpy})
```

这行代码的含义是：

```text
将 input_numpy 传入 ONNX 模型的 input 节点，并获取 output 节点的输出结果。
```

### 5. 输出一致性检查

```python
max_abs_diff = np.max(np.abs(pytorch_logits - onnx_logits))
```

这行代码用于计算 PyTorch 输出和 ONNX Runtime 输出之间的最大绝对误差。误差越小，说明模型导出后的推理结果越接近原 PyTorch 模型。

---

## 今日结论

今天完成了 ONNX Runtime 推理流程，并与 PyTorch 推理结果进行了对比。实验验证了 Day 4 导出的 ResNet18 ONNX 模型可以被 ONNX Runtime 正常加载和推理。

本次实验结果表明：

* ONNX Runtime 与 PyTorch 的 Top-5 分类结果完全一致
* 两者 logits 最大绝对误差约为 `1e-5`
* ONNX Runtime 在 CPU 上相比 PyTorch 实现了约 `2.40x~2.67x` 的推理加速
* ONNX Runtime 可以作为后续端侧 AI 模型部署和推理优化的基础框架

至此，项目已经完成了从 PyTorch 预训练模型、真实图片推理、ONNX 模型导出，到 ONNX Runtime 推理和性能对比的完整最小闭环。

---

## 下一步计划

后续可以继续进行：

* batch size 对推理速度的影响测试
* 多张图片批量推理
* ResNet18 与 MobileNetV2 的模型对比
* ONNX 动态量化实验
* 推理延迟、FPS、模型大小的可视化图表生成
* 项目 README 和简历项目描述整理
