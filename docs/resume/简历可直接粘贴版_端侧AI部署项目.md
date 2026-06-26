\# 简历可直接粘贴版：端侧 AI 部署项目



\## 项目经历版本



\*\*基于 PyTorch 与 ONNX Runtime 的轻量化图像分类模型部署与推理优化系统\*\*

GitHub: https://github.com/mengpipi1019-create/edge-ai-onnxruntime-benchmark

技术栈：Python、PyTorch、torchvision、ONNX、ONNX Runtime、NumPy、Pandas、Matplotlib、Pillow、Git



\* 围绕端侧 AI 模型部署场景，构建 PyTorch → ONNX → ONNX Runtime 的图像分类推理流程，覆盖模型加载、真实图片预处理、ONNX 导出、部署推理、性能测试与可视化分析。

\* 基于 ImageNet 预训练 ResNet18 完成真实图片 Top-K 分类推理，并统计 CPU 环境下的平均推理延迟、FPS 和 batch size 性能变化。

\* 使用 `torch.onnx.export` 导出 ONNX 模型，并结合 `onnx.checker`、Top-K 分类一致性和 logits 最大绝对误差验证模型转换正确性，误差稳定在 `1e-5` 量级。

\* 设计 batch size = 1、2、4、8 的推理 benchmark，生成 CSV 结果和性能曲线；实验显示 ONNX Runtime 在 CPU 上相比 PyTorch 实现约 `2.00x\~2.67x` 推理加速。



\---



\## 如果简历空间不够，用这个短版



\*\*基于 PyTorch 与 ONNX Runtime 的端侧 AI 推理优化系统\*\*

GitHub: https://github.com/mengpipi1019-create/edge-ai-onnxruntime-benchmark



\* 构建 PyTorch → ONNX → ONNX Runtime 的图像分类模型部署流程，完成真实图片预处理、ONNX 导出、CPU 推理和性能测试。

\* 对比 PyTorch 与 ONNX Runtime 的 Top-K 结果、logits 数值误差、平均延迟和 FPS，验证模型转换后输出一致性。

\* 设计 batch size = 1、2、4、8 的 benchmark，生成 CSV 和性能曲线；ONNX Runtime 在 CPU 上相比 PyTorch 实现约 `2.00x\~2.67x` 推理加速。



\---



\## 简历中推荐放置位置



建议放在“项目经历”模块中，顺序建议为：



1\. 科研项目：低轨卫星边缘智能 / 异步联邦学习 / 通信资源优化

2\. 工程项目：基于 PyTorch 与 ONNX Runtime 的端侧 AI 推理优化系统



这个顺序能形成你的个人标签：



\*\*通信边缘智能科研 + AI 模型部署工程项目\*\*



\---



\## 不建议写法



不要写成：



\* 熟悉 ONNX Runtime

\* 了解端侧 AI

\* 做过图像分类



这些写法太泛，不够工程化。



应该突出：



\* 完整部署链路

\* 结果一致性验证

\* benchmark 设计

\* latency / FPS / speedup 指标

\* GitHub 可展示项目



