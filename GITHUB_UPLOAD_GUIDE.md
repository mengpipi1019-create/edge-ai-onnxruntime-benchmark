# GitHub 上传说明

## 1. 建议上传内容

建议上传：

- configs/
- docs/
- images/
- results/day6_batch_benchmark.csv
- results/day7_figures/
- scripts/
- README.md
- requirements.txt
- .gitignore
- run_pipeline.bat
- GITHUB_UPLOAD_GUIDE.md

## 2. 不建议上传内容

不建议直接上传大型模型文件：

- models/*.onnx
- models/*.pth
- models/*.pt

原因：

- ONNX 模型文件较大
- GitHub 普通仓库不适合直接存放大模型文件
- 这些模型可以通过脚本重新生成

## 3. 推荐仓库名

edge-ai-onnxruntime-benchmark

## 4. 推荐项目描述

基于 PyTorch 与 ONNX Runtime 的轻量化图像分类模型部署与推理优化系统，实现 PyTorch 推理、ONNX 导出、ONNX Runtime 部署、batch size benchmark 与性能曲线可视化。

## 5. 推荐标签

- PyTorch
- ONNX
- ONNX Runtime
- Edge AI
- Model Deployment
- Inference Optimization
- Computer Vision

## 6. 首次提交命令

git init  
git add .  
git commit -m "Initial edge AI ONNX Runtime benchmark project"