\# 最终简历项目描述：端侧 AI 模型部署与推理优化



\## 项目名称



基于 PyTorch 与 ONNX Runtime 的轻量化图像分类模型部署与推理优化系统



\## 项目链接



GitHub：https://github.com/mengpipi1019-create/edge-ai-onnxruntime-benchmark



\## 技术栈



Python、PyTorch、torchvision、ONNX、ONNX Runtime、NumPy、Pandas、Matplotlib、Pillow、Git



\## 推荐放入简历版本



\*\*基于 PyTorch 与 ONNX Runtime 的轻量化图像分类模型部署与推理优化系统\*\*

GitHub: https://github.com/mengpipi1019-create/edge-ai-onnxruntime-benchmark



\* 围绕端侧 AI 模型部署场景，构建 PyTorch → ONNX → ONNX Runtime 的图像分类推理流程，覆盖模型加载、真实图片预处理、ONNX 导出、部署推理、性能测试与可视化分析。

\* 基于 ImageNet 预训练 ResNet18 完成真实图片 Top-K 分类推理，并统计 CPU 环境下的平均推理延迟、FPS 和 batch size 性能变化。

\* 使用 `torch.onnx.export` 导出 ONNX 模型，并结合 `onnx.checker`、Top-K 分类一致性和 logits 最大绝对误差验证模型转换正确性，误差稳定在 `1e-5` 量级。

\* 设计 batch size = 1、2、4、8 的推理 benchmark，生成 CSV 结果和性能曲线；实验显示 ONNX Runtime 在 CPU 上相比 PyTorch 实现约 `2.00x\~2.67x` 推理加速。



\## 更短版本



\*\*基于 PyTorch 与 ONNX Runtime 的端侧 AI 推理优化系统\*\*

GitHub: https://github.com/mengpipi1019-create/edge-ai-onnxruntime-benchmark



\* 构建 PyTorch → ONNX → ONNX Runtime 的图像分类模型部署流程，完成真实图片预处理、ONNX 导出、CPU 推理和性能测试。

\* 对比 PyTorch 与 ONNX Runtime 的 Top-K 结果、logits 数值误差、平均延迟和 FPS，验证模型转换后输出一致性。

\* 设计 batch size = 1、2、4、8 的 benchmark，生成 CSV 和性能曲线；ONNX Runtime 在 CPU 上相比 PyTorch 实现约 `2.00x\~2.67x` 推理加速。



\## 面试 30 秒介绍



这个项目主要是为了补充端侧 AI 模型部署能力。我使用 ImageNet 预训练 ResNet18 作为基础模型，先完成 PyTorch 真实图片推理，然后导出 ONNX 模型，并用 ONNX Runtime 在 CPU 上部署推理。为了验证导出正确性，我对比了 PyTorch 和 ONNX Runtime 的 Top-K 结果以及 logits 最大误差，误差稳定在 1e-5 量级。最后我设计了不同 batch size 下的 benchmark，发现 ONNX Runtime 在 CPU 上相比 PyTorch 有约 2 倍以上的推理加速。



\## 面试 1 分钟介绍



我做了一个基于 PyTorch 与 ONNX Runtime 的端侧 AI 推理部署项目，目标是跑通从训练框架模型到部署推理引擎的完整流程。



项目中我使用 ImageNet 预训练 ResNet18 作为基础模型，先完成 PyTorch 真实图片分类推理，然后使用 `torch.onnx.export` 将模型导出为 ONNX 格式，再基于 ONNX Runtime 进行 CPU 推理。为了验证模型转换是否正确，我对比了 PyTorch 和 ONNX Runtime 的 Top-K 分类结果以及 logits 最大绝对误差，实验中 Top-K 结果一致，误差约为 1e-5。



在性能测试部分，我设计了 batch size = 1、2、4、8 的 benchmark，统计 batch latency、单图延迟、FPS 和 speedup，并生成 CSV 和可视化曲线。实验结果显示，ONNX Runtime 在 CPU 上相比 PyTorch 实现约 2.00x 到 2.67x 的推理加速。这个项目让我完整掌握了 PyTorch 模型导出、ONNX Runtime 部署、结果一致性验证和推理性能测试的基本流程。



