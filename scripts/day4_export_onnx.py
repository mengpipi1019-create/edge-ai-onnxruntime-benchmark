import argparse
from pathlib import Path

import torch
import torchvision.models as models
import onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Day 4: Export PyTorch ResNet18 to ONNX")
    parser.add_argument(
        "--output",
        type=str,
        default="models/resnet18_imagenet.onnx",
        help="Path to save ONNX model"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 导出 ONNX 时优先用 CPU，流程最稳定
    device = torch.device("cpu")
    print("Using device:", device)

    # 2. 加载 ImageNet 预训练 ResNet18
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model = model.to(device)
    model.eval()

    # 3. 构造一个假的输入，用来告诉 ONNX 模型输入长什么样
    # shape = [batch_size, channels, height, width]
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    print("Dummy input shape:", dummy_input.shape)

    # 4. 导出 ONNX
    torch.onnx.export(
    model,
    dummy_input,
    str(output_path),
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamo=False
)

    print("ONNX model exported to:", output_path)

    # 5. 检查 ONNX 文件是否存在
    if not output_path.exists():
        raise FileNotFoundError(f"ONNX file was not created: {output_path}")

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print("ONNX file size: {:.2f} MB".format(file_size_mb))

    # 6. 使用 onnx.checker 检查模型格式是否合法
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check: PASSED")

    # 7. 打印输入输出名称
    print("\nONNX model inputs:")
    for input_node in onnx_model.graph.input:
        print("-", input_node.name)

    print("\nONNX model outputs:")
    for output_node in onnx_model.graph.output:
        print("-", output_node.name)


if __name__ == "__main__":
    main()