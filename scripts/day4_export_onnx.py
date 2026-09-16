"""Export pretrained ResNet18 with fixed batch-one input and ONNX opset 18."""
import argparse
from pathlib import Path

import onnx
import torch
from torchvision.models import ResNet18_Weights, resnet18


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/resnet18.onnx'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval().cpu()
    sample = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(model, sample, str(args.output), export_params=True,
                      opset_version=18, do_constant_folding=True,
                      input_names=['input'], output_names=['output'], dynamo=False)
    onnx.checker.check_model(onnx.load(str(args.output)))
    print(f'Exported and checked: {args.output}')


if __name__ == '__main__':
    main()
