import argparse
import time
from pathlib import Path

import torch
import torchvision.models as models
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Day 3: PyTorch real image inference")
    parser.add_argument(
        "--image",
        type=str,
        default="images/test.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Show top-k predicted classes"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}\n"
            f"Please put a test image at: {image_path}"
        )

    # 1. 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. 加载 ImageNet 预训练 ResNet18
    # 第一次运行会自动下载预训练权重
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model = model.to(device)
    model.eval()

    # 3. 获取官方预处理流程和类别名
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    # 4. 读取真实图片
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    print("Input shape:", input_batch.shape)

    print("Image path:", image_path)
    print("Input batch shape:", input_batch.shape)

    # 5. 先 warm up，减少第一次推理的偶然开销
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_batch)

    # 6. 正式计时推理
    num_runs = 20
    latencies = []

    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            output = model(input_batch)
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency

    # 7. 计算 Top-K 分类结果
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    topk_prob, topk_id = torch.topk(probabilities, args.topk)

    print("\nTop-{} predictions:".format(args.topk))
    for rank, (prob, class_id) in enumerate(zip(topk_prob, topk_id), start=1):
        class_name = categories[class_id.item()]
        print(
            "{}. {} | prob: {:.4f}".format(
                rank,
                class_name,
                prob.item()
            )
        )

    print("\nPerformance:")
    print("Average latency: {:.3f} ms".format(avg_latency))
    print("FPS: {:.2f}".format(fps))


if __name__ == "__main__":
    main()