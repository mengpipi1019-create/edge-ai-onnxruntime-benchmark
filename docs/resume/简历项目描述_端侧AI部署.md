# 简历项目描述：端侧 AI 部署项目

## 项目名称

基于 PyTorch 与 ONNX Runtime 的轻量化图像分类模型部署与推理优化系统

## 技术栈

Python、PyTorch、ONNX、ONNX Runtime、NumPy、Pandas、Matplotlib、Pillow

## 简历 Bullet 初稿

* 基于 ImageNet 预训练 ResNet18 构建图像分类推理流程，完成真实图片读取、预处理、Top-K 分类输出和推理延迟统计。
* 使用 `torch.onnx.export` 将 PyTorch 模型导出为 ONNX 格式，并通过 `onnx.checker.check_model` 验证模型格式合法性。
* 基于 ONNX Runtime 实现 ONNX 模型 CPU 推理，对比 PyTorch 与 ONNX Runtime 的 Top-5 分类结果、logits 数值误差、平均延迟和 FPS。
* 设计 batch size = 1、2、4、8 的推理性能测试，生成 CSV 结果和性能曲线图；实验中 ONNX Runtime 在 CPU 上相比 PyTorch 实现约 `2.00x~2.67x` 推理加速，输出误差稳定在 `1e-5` 量级。

## 面试讲解版

这个项目主要是为了补充端侧 AI 模型部署能力。我使用 ImageNet 预训练 ResNet18 作为基础模型，先完成了 PyTorch 真实图片分类推理，然后将模型导出为 ONNX 格式，并用 ONNX Runtime 在 CPU 上进行部署推理。

在结果验证方面，我对比了 PyTorch 和 ONNX Runtime 的 Top-5 分类结果和 logits 数值误差，确认 ONNX 模型的推理结果与 PyTorch 基本一致。在性能测试方面，我进一步测试了 batch size = 1、2、4、8 下的推理延迟和 FPS，并生成 CSV 和曲线图进行分析。实验结果显示，ONNX Runtime 在 CPU 上相比 PyTorch 有约 2 倍以上的推理加速效果。
