# Day 3：真实图片输入与 PyTorch 单图分类推理

## 今日目标

* 准备一张真实图片作为模型输入
* 使用 ImageNet 预训练 ResNet18 模型
* 完成图像读取与预处理
* 完成 PyTorch 单图分类推理
* 输出 Top-5 分类结果
* 统计平均推理延迟和 FPS

## 项目目录

```text
edge_ai_onnx_project
├── images
│   └── test.jpg
├── scripts
│   ├── day2_pytorch_inference.py
│   └── day3_real_image_inference.py
```

## 运行命令

```bash
python scripts/day3_real_image_inference.py --image images/test.jpg
```

## 输入说明

输入图片路径：

```text
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

## 运行结果

```text
Using device:
Image path:
Input batch shape:

Top-5 predictions:
1.
2.
3.
4.
5.

Performance:
Average latency:
FPS:
```

## 今日结论

今天完成了真实图片输入下的 PyTorch 单图分类推理流程。相比 Day 2 使用随机输入验证模型前向传播，Day 3 增加了真实图片读取、图像预处理、预训练模型加载、Top-5 分类结果输出和推理延迟统计。

本次实验说明，端侧 AI 项目的 PyTorch 推理部分已经从“随机输入 demo”推进到“真实图片分类 demo”。后续可以在此基础上继续完成 ONNX 模型导出，并使用 ONNX Runtime 进行推理性能对比。
