# 依赖与数据来源

- ResNet18 网络和 ImageNet 预训练参数使用 torchvision 提供的实现与权重。权重首次运行时下载，仓库未重新发布模型参数。请遵守 PyTorch、torchvision、ONNX、ONNX Runtime 各自的许可及使用条件。
- `cpp_inference/assets/imagenet_classes.txt` 来自所用 torchvision 权重的类别元数据，顺序与输出索引一致。
- `images/cat.jpg` 和 `images/test.jpg` 沿用项目早期已经公开的测试图片。本次整理未找到可核实的原始图片授权记录，不对其版权作额外授权；商业再使用前请核实来源或替换为有权使用的图片。替换图片后不应沿用历史样本哈希或结果。

本次只整理公开工程内容，不为历史第三方图片新增许可，也不将第三方依赖的许可证覆盖为本项目许可。
