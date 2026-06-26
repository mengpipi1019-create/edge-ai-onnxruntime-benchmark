# Day 4：PyTorch 模型导出 ONNX

## 今日目标

* 加载 ImageNet 预训练 ResNet18 模型
* 构造符合模型输入格式的 dummy input
* 使用 `torch.onnx.export` 将 PyTorch 模型导出为 ONNX 格式
* 使用 `onnx.checker.check_model` 检查 ONNX 模型文件是否合法
* 记录 ONNX 文件路径、文件大小、输入名和输出名

## 项目目录

```text
edge_ai_onnx_project
├── models
│   └── resnet18_imagenet.onnx
├── scripts
│   └── day4_export_onnx.py
```

## 运行命令

```bash
python scripts/day4_export_onnx.py
```

## 运行结果

```text
Using device: cpu
Dummy input shape: torch.Size([1, 3, 224, 224])
C:\Users\mengp\desktop\edge_ai_onnx_project\scripts\day4_export_onnx.py:48: DeprecationWarning: You are using the legacy TorchScript-based ONNX export. Starting in PyTorch 2.9, the new torch.export-based ONNX exporter has become the default. Learn more about the new export logic: https://docs.pytorch.org/docs/stable/onnx_export.html. For exporting control flow: https://pytorch.org/tutorials/beginner/onnx/export_control_flow_model_to_onnx_tutorial.html
  torch.onnx.export(
ONNX model exported to: models\resnet18_imagenet.onnx
ONNX file size: 44.58 MB
ONNX model check: PASSED

ONNX model inputs:
- input

ONNX model outputs:
- output
```

## 关键代码理解

`dummy_input = torch.randn(1, 3, 224, 224)` 用于告诉导出器模型输入的形状，其中 `1` 表示 batch size，`3` 表示 RGB 三通道，`224×224` 表示图像尺寸。

`torch.onnx.export` 是本次实验的核心函数，用于将 PyTorch 模型导出为 ONNX 格式。ONNX 是一种通用模型表示格式，方便后续使用 ONNX Runtime、TensorRT、OpenVINO 等工具进行部署和推理优化。

`input_names=["input"]` 和 `output_names=["output"]` 用于给 ONNX 模型的输入和输出命名，后续 ONNX Runtime 推理时需要根据输入名传入数据。

`dynamic_axes` 用于设置动态 batch size，使导出的 ONNX 模型后续能够支持不同 batch size 的输入。

## 今日结论

今天完成了从 PyTorch 预训练 ResNet18 到 ONNX 模型文件的导出流程，并使用 `onnx.checker.check_model` 验证了 ONNX 文件格式合法。

这一步是端侧 AI 部署项目中的关键中间环节：Day 2 和 Day 3 验证了 PyTorch 推理流程，Day 4 将模型转换为更适合部署的 ONNX 格式。后续可以继续使用 ONNX Runtime 加载该模型，并与 PyTorch 推理结果和推理延迟进行对比。
