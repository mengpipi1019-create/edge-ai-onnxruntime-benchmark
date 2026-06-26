import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import onnx
import onnxruntime as ort
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Day 6: Batch size benchmark for PyTorch and ONNX Runtime")
    parser.add_argument(
        "--image",
        type=str,
        default="images/cat.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        default="models/resnet18_imagenet_dynamic.onnx",
        help="Path to dynamic-batch ONNX model"
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to benchmark"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=50,
        help="Number of timed inference runs"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/day6_batch_benchmark.csv",
        help="Path to save benchmark results"
    )
    parser.add_argument(
        "--force_export",
        action="store_true",
        help="Force re-export dynamic ONNX model"
    )
    return parser.parse_args()


def export_dynamic_onnx_model(onnx_path):
    """
    导出支持动态 batch size 的 ONNX 模型。
    Day 4 的模型是固定 batch=1。
    Day 6 要测试 batch=1/2/4/8，所以这里单独导出 dynamic batch 模型。
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    print("Exporting dynamic-batch ONNX model...")
    print("ONNX output path:", onnx_path)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        dynamo=False
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    file_size_mb = onnx_path.stat().st_size / 1024 / 1024
    print("Dynamic ONNX export done.")
    print("ONNX file size: {:.2f} MB".format(file_size_mb))
    print("ONNX model check: PASSED")


def build_input_batch(image_path, batch_size, preprocess, device):
    """
    读取一张图片，并复制成指定 batch size。
    比如 batch_size=4，就是构造 4 张相同图片组成一个 batch。
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)

    # [3, 224, 224] -> [1, 3, 224, 224] -> [batch_size, 3, 224, 224]
    input_batch = input_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    input_batch = input_batch.contiguous().to(device)

    return input_batch


def benchmark_pytorch(model, input_batch, warmup, num_runs):
    """
    测试 PyTorch 推理耗时。
    返回平均 batch 延迟、单图延迟、FPS。
    """
    batch_size = input_batch.shape[0]

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_batch)

    latencies = []

    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            output = model(input_batch)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

    avg_batch_latency = sum(latencies) / len(latencies)
    avg_image_latency = avg_batch_latency / batch_size
    fps = batch_size * 1000.0 / avg_batch_latency

    return output.detach().cpu().numpy(), avg_batch_latency, avg_image_latency, fps


def benchmark_onnxruntime(session, input_name, output_name, input_numpy, warmup, num_runs):
    """
    测试 ONNX Runtime 推理耗时。
    返回平均 batch 延迟、单图延迟、FPS。
    """
    batch_size = input_numpy.shape[0]

    for _ in range(warmup):
        _ = session.run([output_name], {input_name: input_numpy})

    latencies = []

    for _ in range(num_runs):
        start = time.perf_counter()
        output = session.run([output_name], {input_name: input_numpy})
        end = time.perf_counter()

        latencies.append((end - start) * 1000)

    avg_batch_latency = sum(latencies) / len(latencies)
    avg_image_latency = avg_batch_latency / batch_size
    fps = batch_size * 1000.0 / avg_batch_latency

    return output[0], avg_batch_latency, avg_image_latency, fps


def main():
    args = parse_args()

    image_path = Path(args.image)
    onnx_model_path = Path(args.onnx_model)
    output_csv_path = Path(args.output_csv)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 如果动态 ONNX 模型不存在，就先导出
    if args.force_export or not onnx_model_path.exists():
        export_dynamic_onnx_model(onnx_model_path)

    # 2. 加载 PyTorch 模型
    device = torch.device("cpu")
    print("\nUsing device:", device)

    weights = models.ResNet18_Weights.DEFAULT
    pytorch_model = models.resnet18(weights=weights)
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()

    preprocess = weights.transforms()

    # 3. 加载 ONNX Runtime 模型
    ort_session = ort.InferenceSession(
        str(onnx_model_path),
        providers=["CPUExecutionProvider"]
    )

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    print("Image path:", image_path)
    print("ONNX model:", onnx_model_path)
    print("ONNX input name:", input_name)
    print("ONNX output name:", output_name)
    print("Batch sizes:", args.batch_sizes)
    print("Warmup runs:", args.warmup)
    print("Timed runs:", args.num_runs)

    rows = []

    for batch_size in args.batch_sizes:
        print("\nBenchmarking batch size:", batch_size)

        input_batch = build_input_batch(
            image_path=image_path,
            batch_size=batch_size,
            preprocess=preprocess,
            device=device
        )

        input_numpy = input_batch.cpu().numpy().astype(np.float32)

        pytorch_output, pt_batch_latency, pt_image_latency, pt_fps = benchmark_pytorch(
            model=pytorch_model,
            input_batch=input_batch,
            warmup=args.warmup,
            num_runs=args.num_runs
        )

        onnx_output, ort_batch_latency, ort_image_latency, ort_fps = benchmark_onnxruntime(
            session=ort_session,
            input_name=input_name,
            output_name=output_name,
            input_numpy=input_numpy,
            warmup=args.warmup,
            num_runs=args.num_runs
        )

        max_abs_diff = np.max(np.abs(pytorch_output - onnx_output))
        speedup = pt_batch_latency / ort_batch_latency

        print(
            "PyTorch: batch latency = {:.3f} ms, image latency = {:.3f} ms, FPS = {:.2f}".format(
                pt_batch_latency,
                pt_image_latency,
                pt_fps
            )
        )
        print(
            "ONNX Runtime: batch latency = {:.3f} ms, image latency = {:.3f} ms, FPS = {:.2f}".format(
                ort_batch_latency,
                ort_image_latency,
                ort_fps
            )
        )
        print("Max absolute difference: {:.6f}".format(max_abs_diff))
        print("ONNX Runtime speedup: {:.2f}x".format(speedup))

        rows.append({
            "batch_size": batch_size,
            "pytorch_batch_latency_ms": pt_batch_latency,
            "pytorch_image_latency_ms": pt_image_latency,
            "pytorch_fps": pt_fps,
            "onnx_batch_latency_ms": ort_batch_latency,
            "onnx_image_latency_ms": ort_image_latency,
            "onnx_fps": ort_fps,
            "speedup": speedup,
            "max_abs_diff": max_abs_diff
        })

    # 4. 保存 CSV
    fieldnames = [
        "batch_size",
        "pytorch_batch_latency_ms",
        "pytorch_image_latency_ms",
        "pytorch_fps",
        "onnx_batch_latency_ms",
        "onnx_image_latency_ms",
        "onnx_fps",
        "speedup",
        "max_abs_diff"
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print("\nBenchmark results saved to:", output_csv_path)

    # 5. 打印 Markdown 表格，方便复制到 md 文档
    print("\nMarkdown table:")
    print("| Batch size | PyTorch batch latency ms | ONNX batch latency ms | PyTorch FPS | ONNX FPS | Speedup | Max diff |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        print(
            "| {batch_size} | {pt_b:.3f} | {ort_b:.3f} | {pt_fps:.2f} | {ort_fps:.2f} | {speedup:.2f}x | {diff:.6f} |".format(
                batch_size=row["batch_size"],
                pt_b=row["pytorch_batch_latency_ms"],
                ort_b=row["onnx_batch_latency_ms"],
                pt_fps=row["pytorch_fps"],
                ort_fps=row["onnx_fps"],
                speedup=row["speedup"],
                diff=row["max_abs_diff"]
            )
        )


if __name__ == "__main__":
    main()