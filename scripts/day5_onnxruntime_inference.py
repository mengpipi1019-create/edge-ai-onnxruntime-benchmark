import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import onnxruntime as ort
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Day 5: ONNX Runtime inference")
    parser.add_argument(
        "--image",
        type=str,
        default="images/cat.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        default="models/resnet18_imagenet.onnx",
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Show top-k predictions"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=20,
        help="Number of inference runs for latency test"
    )
    return parser.parse_args()


def print_topk(title, logits, categories, topk):
    """
    logits: shape [1000]
    """
    probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=0)
    topk_prob, topk_id = torch.topk(probabilities, topk)

    print(f"\n{title}")
    for rank, (prob, class_id) in enumerate(zip(topk_prob, topk_id), start=1):
        class_name = categories[class_id.item()]
        print(f"{rank}. {class_name} | prob: {prob.item():.4f}")


def main():
    args = parse_args()

    image_path = Path(args.image)
    onnx_model_path = Path(args.onnx_model)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    # 1. 当前先使用 CPU
    device = torch.device("cpu")
    print("Using device:", device)

    # 2. 加载 PyTorch 预训练 ResNet18，用于和 ONNX 结果对比
    weights = models.ResNet18_Weights.DEFAULT
    pytorch_model = models.resnet18(weights=weights)
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()

    # 3. 获取和 Day 3 一样的官方预处理流程
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    # 4. 读取图片并预处理
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    print("Image path:", image_path)
    print("Input batch shape:", input_batch.shape)

    # 5. 转成 ONNX Runtime 需要的 numpy 格式
    input_numpy = input_batch.cpu().numpy().astype(np.float32)

    # 6. PyTorch 推理
    with torch.no_grad():
        for _ in range(3):
            _ = pytorch_model(input_batch)

    pytorch_latencies = []

    with torch.no_grad():
        for _ in range(args.num_runs):
            start_time = time.time()
            pytorch_output = pytorch_model(input_batch)
            end_time = time.time()
            pytorch_latencies.append((end_time - start_time) * 1000)

    pytorch_avg_latency = sum(pytorch_latencies) / len(pytorch_latencies)
    pytorch_fps = 1000.0 / pytorch_avg_latency

    pytorch_logits = pytorch_output[0].detach().cpu().numpy()

    # 7. 创建 ONNX Runtime 推理会话
    ort_session = ort.InferenceSession(
        str(onnx_model_path),
        providers=["CPUExecutionProvider"]
    )

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    print("\nONNX Runtime info:")
    print("ONNX model:", onnx_model_path)
    print("Input name:", input_name)
    print("Output name:", output_name)

    # 8. ONNX Runtime 推理
    for _ in range(3):
        _ = ort_session.run([output_name], {input_name: input_numpy})

    onnx_latencies = []

    for _ in range(args.num_runs):
        start_time = time.time()
        onnx_outputs = ort_session.run([output_name], {input_name: input_numpy})
        end_time = time.time()
        onnx_latencies.append((end_time - start_time) * 1000)

    onnx_avg_latency = sum(onnx_latencies) / len(onnx_latencies)
    onnx_fps = 1000.0 / onnx_avg_latency

    onnx_logits = onnx_outputs[0][0]

    # 9. 打印分类结果
    print_topk("PyTorch Top-{} predictions:".format(args.topk), pytorch_logits, categories, args.topk)
    print_topk("ONNX Runtime Top-{} predictions:".format(args.topk), onnx_logits, categories, args.topk)

    # 10. 对比数值差异
    max_abs_diff = np.max(np.abs(pytorch_logits - onnx_logits))

    print("\nOutput difference:")
    print("Max absolute difference between PyTorch and ONNX logits: {:.6f}".format(max_abs_diff))

    # 11. 打印性能对比
    print("\nPerformance comparison:")
    print("PyTorch average latency: {:.3f} ms | FPS: {:.2f}".format(
        pytorch_avg_latency,
        pytorch_fps
    ))
    print("ONNX Runtime average latency: {:.3f} ms | FPS: {:.2f}".format(
        onnx_avg_latency,
        onnx_fps
    ))

    speedup = pytorch_avg_latency / onnx_avg_latency
    print("ONNX Runtime speedup over PyTorch: {:.2f}x".format(speedup))


if __name__ == "__main__":
    main()