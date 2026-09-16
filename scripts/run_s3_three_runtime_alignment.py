"""Run S3's same-tensor and end-to-end five-image three-runtime alignment checks."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision
import torchvision.models as models
from PIL import Image, __version__ as PILLOW_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_COUNT = 1 * 3 * 224 * 224
LOGIT_COUNT = 1000
SAME_TENSOR_PT_ORT_MAX_THRESHOLD = 1e-4
SAME_TENSOR_ORT_CPP_MAX_THRESHOLD = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
  path = Path(value)
  return path if path.is_absolute() else PROJECT_ROOT / path


def cpp_path_argument(path: Path) -> str:
    """Use a project-relative path so MinGW receives ASCII-only CLI arguments."""
    return str(path.relative_to(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="results/evidence/2026-08-25_s1_baseline/resnet18_imagenet_s1.onnx")
    parser.add_argument("--cpp-exe", default="cpp_inference/build/resnet18_ort_cpp.exe")
    parser.add_argument("--labels", default="cpp_inference/assets/imagenet_classes.txt")
    parser.add_argument("--samples-dir", default="images/s3_alignment")
    parser.add_argument("--results-dir", default="results/evidence/2026-08-25_s3_three_runtime_alignment")
    return parser.parse_args()


def write_f32(path: Path, values: np.ndarray, expected_count: int) -> None:
    array = np.ascontiguousarray(values, dtype="<f4")
    if array.size != expected_count:
        raise ValueError(f"Expected {expected_count} float32 values for {path}, got {array.size}")
    array.tofile(path)


def read_f32(path: Path, expected_count: int) -> np.ndarray:
    expected_bytes = expected_count * 4
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{path} expected {expected_bytes} bytes, got {actual_bytes}")
    values = np.fromfile(path, dtype="<f4")
    if values.size != expected_count:
        raise ValueError(f"{path} expected {expected_count} float32 values, got {values.size}")
    return values


def top5(values: np.ndarray) -> list[int]:
    return np.argsort(-values, kind="stable")[:5].astype(int).tolist()


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def mean_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


def invoke_cpp(command: list[str]) -> None:
    print("C++ command:", subprocess.list2cmdline(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise RuntimeError(f"C++ inference failed with exit code {completed.returncode}")


def worst_sample(records: list[dict], key: str) -> dict:
    return max(records, key=lambda record: float(record[key]))


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    cpp_exe = resolve_path(args.cpp_exe)
    labels_path = resolve_path(args.labels)
    samples_dir = resolve_path(args.samples_dir)
    results_dir = resolve_path(args.results_dir)
    if not model_path.is_file() or not cpp_exe.is_file() or not labels_path.is_file():
        raise FileNotFoundError("Model, C++ executable, or labels file is missing")

    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_s3_alignment_samples.py"),
                    "--output-dir", str(samples_dir)], cwd=PROJECT_ROOT, check=True)
    manifest_path = samples_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    canonical_dir = results_dir / "canonical_inputs"
    cpp_input_dir = results_dir / "cpp_inputs"
    logits_dir = results_dir / "logits"
    for directory in (canonical_dir, cpp_input_dir, logits_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, results_dir / "sample_manifest.json")

    weights = models.ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    pytorch_model = models.resnet18(weights=weights).eval()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Unexpected ORT Python providers: {session.get_providers()}")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("S3 same-tensor thresholds:", SAME_TENSOR_PT_ORT_MAX_THRESHOLD, SAME_TENSOR_ORT_CPP_MAX_THRESHOLD)
    print("Model SHA-256:", sha256(model_path))
    print("ORT Python:", ort.__version__, session.get_providers())
    records = []

    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        image_path = PROJECT_ROOT / sample["file"]
        print(f"\n===== S3 sample: {sample_id} =====")
        with Image.open(image_path) as opened:
            canonical = preprocess(opened.convert("RGB")).unsqueeze(0).contiguous().numpy().astype("<f4", copy=False)
        canonical_path = canonical_dir / f"{sample_id}.f32"
        write_f32(canonical_path, canonical, INPUT_COUNT)

        with torch.inference_mode():
            pytorch_logits = pytorch_model(torch.from_numpy(canonical.copy())).numpy().astype("<f4", copy=False).reshape(-1)
        ort_logits = session.run([output_name], {input_name: canonical})[0].astype("<f4", copy=False).reshape(-1)
        write_f32(logits_dir / f"{sample_id}_pytorch.f32", pytorch_logits, LOGIT_COUNT)
        write_f32(logits_dir / f"{sample_id}_ort_python.f32", ort_logits, LOGIT_COUNT)

        cpp_same_logits_path = logits_dir / f"{sample_id}_ort_cpp_same_tensor.f32"
        invoke_cpp([
            str(cpp_exe), "--model", cpp_path_argument(model_path), "--input-bin", cpp_path_argument(canonical_path),
            "--labels", cpp_path_argument(labels_path), "--dump-logits", cpp_path_argument(cpp_same_logits_path),
        ])
        cpp_same_logits = read_f32(cpp_same_logits_path, LOGIT_COUNT)

        cpp_input_path = cpp_input_dir / f"{sample_id}.f32"
        cpp_e2e_logits_path = logits_dir / f"{sample_id}_ort_cpp_end_to_end.f32"
        invoke_cpp([
            str(cpp_exe), "--model", cpp_path_argument(model_path), "--image", cpp_path_argument(image_path),
            "--labels", cpp_path_argument(labels_path), "--dump-input", cpp_path_argument(cpp_input_path),
            "--dump-logits", cpp_path_argument(cpp_e2e_logits_path),
        ])
        cpp_input = read_f32(cpp_input_path, INPUT_COUNT)
        cpp_e2e_logits = read_f32(cpp_e2e_logits_path, LOGIT_COUNT)

        pytorch_top5 = top5(pytorch_logits)
        ort_top5 = top5(ort_logits)
        cpp_same_top5 = top5(cpp_same_logits)
        cpp_e2e_top5 = top5(cpp_e2e_logits)
        same_tensor_top5_match = pytorch_top5 == ort_top5 == cpp_same_top5
        record = {
            "sample_id": sample_id,
            "sample_sha256": sample["sha256"],
            "pytorch_top5": json.dumps(pytorch_top5),
            "ort_python_top5": json.dumps(ort_top5),
            "ort_cpp_same_tensor_top5": json.dumps(cpp_same_top5),
            "same_tensor_top5_exact_match": same_tensor_top5_match,
            "pytorch_vs_ort_python_max_abs": max_abs(pytorch_logits, ort_logits),
            "pytorch_vs_ort_python_mean_abs": mean_abs(pytorch_logits, ort_logits),
            "ort_python_vs_ort_cpp_same_tensor_max_abs": max_abs(ort_logits, cpp_same_logits),
            "ort_python_vs_ort_cpp_same_tensor_mean_abs": mean_abs(ort_logits, cpp_same_logits),
            "pytorch_vs_ort_cpp_same_tensor_max_abs": max_abs(pytorch_logits, cpp_same_logits),
            "pytorch_vs_ort_cpp_same_tensor_mean_abs": mean_abs(pytorch_logits, cpp_same_logits),
            "python_vs_cpp_input_max_abs": max_abs(canonical.reshape(-1), cpp_input),
            "python_vs_cpp_input_mean_abs": mean_abs(canonical.reshape(-1), cpp_input),
            "ort_python_canonical_vs_cpp_end_to_end_logits_max_abs": max_abs(ort_logits, cpp_e2e_logits),
            "ort_python_canonical_vs_cpp_end_to_end_logits_mean_abs": mean_abs(ort_logits, cpp_e2e_logits),
            "ort_python_canonical_top5": json.dumps(ort_top5),
            "ort_cpp_end_to_end_top5": json.dumps(cpp_e2e_top5),
            "end_to_end_top1_match": ort_top5[0] == cpp_e2e_top5[0],
            "end_to_end_top5_intersection_count": len(set(ort_top5).intersection(cpp_e2e_top5)),
            "end_to_end_top5_exact_order_match": ort_top5 == cpp_e2e_top5,
        }
        records.append(record)
        print("same_tensor_top5_exact_match=", same_tensor_top5_match)
        print("end_to_end_top1_match=", record["end_to_end_top1_match"])

    csv_path = results_dir / "alignment_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    same_tensor_top5_count = sum(record["same_tensor_top5_exact_match"] for record in records)
    end_to_end_top1_count = sum(record["end_to_end_top1_match"] for record in records)
    worst_pt_ort = worst_sample(records, "pytorch_vs_ort_python_max_abs")
    worst_ort_cpp = worst_sample(records, "ort_python_vs_ort_cpp_same_tensor_max_abs")
    summary = {
        "sample_count": len(records),
        "model_sha256": sha256(model_path),
        "python_environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "onnxruntime": ort.__version__,
            "onnxruntime_session_providers": session.get_providers(),
            "pillow": PILLOW_VERSION,
        },
        "same_tensor_thresholds": {
            "pytorch_vs_ort_python_max_abs": SAME_TENSOR_PT_ORT_MAX_THRESHOLD,
            "ort_python_vs_ort_cpp_max_abs": SAME_TENSOR_ORT_CPP_MAX_THRESHOLD,
        },
        "same_tensor": {
            "top5_exact_match_count": same_tensor_top5_count,
            "pytorch_vs_ort_python_max_abs_overall": max(float(row["pytorch_vs_ort_python_max_abs"]) for row in records),
            "pytorch_vs_ort_python_worst_sample": worst_pt_ort["sample_id"],
            "ort_python_vs_ort_cpp_max_abs_overall": max(float(row["ort_python_vs_ort_cpp_same_tensor_max_abs"]) for row in records),
            "ort_python_vs_ort_cpp_worst_sample": worst_ort_cpp["sample_id"],
        },
        "end_to_end": {
            "top1_match_count": end_to_end_top1_count,
            "input_max_abs_overall": max(float(row["python_vs_cpp_input_max_abs"]) for row in records),
            "logits_max_abs_overall": max(float(row["ort_python_canonical_vs_cpp_end_to_end_logits_max_abs"]) for row in records),
        },
    }
    summary["same_tensor_pass"] = (
        same_tensor_top5_count == len(records)
        and summary["same_tensor"]["pytorch_vs_ort_python_max_abs_overall"] <= SAME_TENSOR_PT_ORT_MAX_THRESHOLD
        and summary["same_tensor"]["ort_python_vs_ort_cpp_max_abs_overall"] <= SAME_TENSOR_ORT_CPP_MAX_THRESHOLD
    )
    summary["end_to_end_top1_pass"] = end_to_end_top1_count == len(records)
    summary["s3_pass"] = summary["same_tensor_pass"] and summary["end_to_end_top1_pass"]
    (results_dir / "alignment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\nS3 same-tensor Top-5 exact matches:", f"{same_tensor_top5_count}/{len(records)}")
    print("S3 end-to-end Top-1 matches:", f"{end_to_end_top1_count}/{len(records)}")
    print("S3 pass:", summary["s3_pass"])
    if not summary["s3_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
