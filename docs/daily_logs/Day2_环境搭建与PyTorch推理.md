# Day 2：环境搭建与第一个 PyTorch 推理

## 今日目标

- 创建端侧 AI 项目目录
- 创建 conda 环境 edge_ai
- 安装 PyTorch、torchvision、ONNX、ONNX Runtime
- 跑通 ResNet18 的 PyTorch 推理脚本

## 环境信息

Python 环境：

```bash
conda create -n edge_ai python=3.10 -y
conda activate edge_ai


pip install torch torchvision torchaudio
pip install numpy matplotlib pillow opencv-python
pip install onnx onnxruntime

scripts/day2_pytorch_inference.py

Using device:
Input shape:
Output shape:
Predicted class index:
Inference latency:


---

# 今天完成标准

你今天只要跑出这几行，就算 Day 2 完成：

```text
Using device: cpu 或 cuda
Input shape: torch.Size([1, 3, 224, 224])
Output shape: torch.Size([1, 1000])
Inference latency: xxx ms