\# 项目面试问答：端侧 AI / ONNX Runtime 部署项目



\## 1. 这个项目做了什么？



本项目实现了一个从 PyTorch 模型推理、ONNX 模型导出、ONNX Runtime 部署推理到性能测试与可视化分析的完整端侧 AI 推理流程。



项目以 ImageNet 预训练 ResNet18 为基础模型，完成真实图片分类推理，并对比 PyTorch 与 ONNX Runtime 在 CPU 环境下的分类结果一致性、平均延迟、FPS 和 batch size 性能变化。



\## 2. 为什么要用 ONNX？



ONNX 是一种通用模型表示格式，可以将 PyTorch 等训练框架中的模型转换成更适合部署的中间表示。



这样模型后续可以接入 ONNX Runtime、TensorRT、OpenVINO 等推理引擎，方便在端侧设备、边缘设备或服务器 CPU/GPU 环境中进行推理优化。



\## 3. 为什么要用 ONNX Runtime？



ONNX Runtime 是高性能推理引擎，支持 CPU、GPU 以及多种硬件加速后端。



本项目中，在 CPU 推理场景下，ONNX Runtime 相比 PyTorch 实现了约 2.00x 到 2.67x 的推理加速，同时保持了与 PyTorch 基本一致的输出结果。



\## 4. 怎么验证 ONNX 模型没有导出错？



本项目从两个层面验证：



第一，使用 onnx.checker.check\_model 检查 ONNX 文件格式是否合法。



第二，对比 PyTorch 与 ONNX Runtime 在真实图片上的 Top-K 分类结果和 logits 最大绝对误差。实验中 Top-K 结果一致，最大误差约为 1e-5，说明导出后结果基本一致。



\## 5. batch size 测试说明了什么？



batch size 测试用于评估模型在批量推理场景下的吞吐性能。



实验结果显示，随着 batch size 从 1 增加到 8，PyTorch 和 ONNX Runtime 的 FPS 都明显提升，说明批量推理能提高整体吞吐率。同时，ONNX Runtime 在所有 batch size 下均快于 PyTorch。



\## 6. 这个项目还有哪些可以继续优化？



后续可以继续扩展：



\- 对比 ResNet18 和 MobileNetV2

\- 增加 ONNX 动态量化

\- 测试内存占用

\- 增加多图片批量推理

\- 在 Linux 或真实边缘设备上部署

\- 尝试 TensorRT / OpenVINO / NCNN 等推理后端

