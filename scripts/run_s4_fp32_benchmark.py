"""Frozen S4 FP32 CPU benchmark: isolated workers, one logical CPU, canonical tensor only."""

import argparse
import csv
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision
import torchvision.models as models
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_ROOT / "results" / "evidence" / "2026-08-25_s4_fp32_benchmark"
PROTOCOL_PATH = EVIDENCE_DIR / "benchmark_protocol.json"
MODEL_PATH = PROJECT_ROOT / "results" / "evidence" / "2026-08-25_s1_baseline" / "resnet18_imagenet_s1.onnx"
CPP_EXE = PROJECT_ROOT / "cpp_inference" / "build" / "resnet18_ort_cpp.exe"
CPP_DLL = PROJECT_ROOT / "cpp_inference" / "build" / "onnxruntime.dll"
LABELS_PATH = PROJECT_ROOT / "cpp_inference" / "assets" / "imagenet_classes.txt"
CANONICAL_INPUT = EVIDENCE_DIR / "canonical_input_cat_original.f32"
EXPECTED_MODEL_SHA256 = "c5ef909be6cda03f0f87a7354355468944e7f8840d5d2d83c344fdbf4a5d3bf6"
EXPECTED_CPP_DLL_SHA256 = "dec964ab1ee36cc9b0ae247d13b376627992fc57dec0454354017ab8fd84f1ea"
INPUT_COUNT = 1 * 3 * 224 * 224
LOGIT_COUNT = 1000
WARMUP = 50
RUNS = 300
REPEATS = 5
LOGICAL_CPU = 0
SCHEDULE_SEED = 20260825


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_root(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def set_and_verify_affinity_cpu0() -> None:
    if os.name != "nt":
        raise RuntimeError("S4 protocol requires Windows logical CPU affinity verification")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    if not kernel32.SetProcessAffinityMask(process, ctypes.c_size_t(1)):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask(logical CPU 0) failed")
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(process, ctypes.byref(process_mask), ctypes.byref(system_mask)):
        raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask verification failed")
    if process_mask.value != 1:
        raise RuntimeError(f"Affinity verification failed: expected mask 1, got {process_mask.value}")
    print("Affinity verified: logical CPU 0 (process mask=1)", flush=True)


def read_input(path: Path) -> np.ndarray:
    expected_bytes = INPUT_COUNT * 4
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Canonical input must be {expected_bytes} bytes: {path}")
    values = np.fromfile(path, dtype="<f4")
    if values.size != INPUT_COUNT:
        raise ValueError(f"Canonical input must contain {INPUT_COUNT} float32 values")
    return values.reshape(1, 3, 224, 224)


def write_worker_csv(path: Path, repeat_id: int, latencies_ms: list[float]) -> None:
    if len(latencies_ms) != RUNS:
        raise ValueError(f"Worker must write exactly {RUNS} measured values")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["repeat_id", "iteration", "latency_ms"])
        writer.writeheader()
        for iteration, latency_ms in enumerate(latencies_ms):
            writer.writerow({"repeat_id": repeat_id, "iteration": iteration, "latency_ms": f"{latency_ms:.9f}"})


def write_worker_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_python_worker(runtime: str, input_path: Path, output_path: Path, repeat_id: int, run_order: int) -> None:
    set_and_verify_affinity_cpu0()
    input_array = read_input(input_path)
    tensor = torch.from_numpy(input_array.copy())
    if runtime == "pytorch":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
        with torch.inference_mode():
            correctness = model(tensor).detach().cpu().numpy().reshape(-1)
        if correctness.size != LOGIT_COUNT or int(np.argmax(correctness)) != 285:
            raise RuntimeError("PyTorch worker correctness check failed")
        with torch.inference_mode():
            for _ in range(WARMUP):
                _ = model(tensor)
            latencies_ms = []
            for _ in range(RUNS):
                start = time.perf_counter_ns()
                _ = model(tensor)
                end = time.perf_counter_ns()
                latencies_ms.append((end - start) / 1_000_000.0)
        metadata = {
            "runtime": runtime,
            "repeat_id": repeat_id,
            "run_order": run_order,
            "affinity_mask": 1,
            "torch_threads": {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()},
            "correctness_top1": int(np.argmax(correctness)),
        }
    elif runtime == "ort_python":
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(f"Unexpected ORT Python providers: {session.get_providers()}")
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        correctness = session.run([output_name], {input_name: input_array})[0].reshape(-1)
        if correctness.size != LOGIT_COUNT or int(np.argmax(correctness)) != 285:
            raise RuntimeError("ORT Python worker correctness check failed")
        for _ in range(WARMUP):
            _ = session.run([output_name], {input_name: input_array})
        latencies_ms = []
        for _ in range(RUNS):
            start = time.perf_counter_ns()
            _ = session.run([output_name], {input_name: input_array})
            end = time.perf_counter_ns()
            latencies_ms.append((end - start) / 1_000_000.0)
        metadata = {
            "runtime": runtime,
            "repeat_id": repeat_id,
            "run_order": run_order,
            "affinity_mask": 1,
            "session_providers": session.get_providers(),
            "session_options": {"intra_op_threads": 1, "inter_op_threads": 1, "execution_mode": "ORT_SEQUENTIAL", "graph_optimization": "ORT_ENABLE_ALL"},
            "correctness_top1": int(np.argmax(correctness)),
        }
    else:
        raise ValueError(f"Unsupported Python worker runtime: {runtime}")

    write_worker_csv(output_path, repeat_id, latencies_ms)
    write_worker_metadata(output_path.with_suffix(".json"), metadata)
    print(f"Worker complete: runtime={runtime}, repeat={repeat_id}, measured={len(latencies_ms)}, run_order={run_order}")


def ensure_canonical_input() -> str:
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_s3_alignment_samples.py")], cwd=PROJECT_ROOT, check=True)
    sample = PROJECT_ROOT / "images" / "s3_alignment" / "cat_original.jpg"
    with Image.open(sample) as image:
        canonical = models.ResNet18_Weights.DEFAULT.transforms()(image.convert("RGB")).unsqueeze(0).contiguous().numpy()
    np.ascontiguousarray(canonical, dtype="<f4").tofile(CANONICAL_INPUT)
    if CANONICAL_INPUT.stat().st_size != INPUT_COUNT * 4:
        raise RuntimeError("Generated canonical input has incorrect byte size")
    return sha256(CANONICAL_INPUT)


def check_frozen_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    measurement = protocol["measurement"]
    if (measurement["warmup_iterations"], measurement["measured_iterations_per_process"],
            measurement["independent_process_repeats_per_runtime"], measurement["total_measured_rows"]) != (WARMUP, RUNS, REPEATS, 4500):
        raise RuntimeError("Frozen protocol does not match S4 required measurement counts")
    if protocol["resources"]["logical_cpu"] != LOGICAL_CPU or protocol["resources"]["threads"] != 1:
        raise RuntimeError("Frozen protocol does not match S4 resource constraints")
    return protocol


def environment_snapshot(input_sha256: str) -> dict:
    power = subprocess.run(
        ["powercfg", "/getactivescheme"], text=True, capture_output=True, encoding="utf-8", errors="replace", check=False
    )
    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte), ("BatteryLifePercent", ctypes.c_byte),
                    ("SystemStatusFlag", ctypes.c_byte), ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]
    battery = SystemPowerStatus()
    battery_ok = bool(ctypes.WinDLL("kernel32").GetSystemPowerStatus(ctypes.byref(battery))) if os.name == "nt" else False
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "windows_version": platform.version(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER"),
        "logical_cpu_count": os.cpu_count(),
        "power_scheme": power.stdout.strip() or power.stderr.strip(),
        "on_ac_power": None if not battery_ok else battery.ACLineStatus == 1,
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "onnxruntime": ort.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "model_sha256": sha256(MODEL_PATH),
        "input_sha256": input_sha256,
        "cpp_dll_sha256": sha256(CPP_DLL),
        "background_load_note": "No synthetic load was started; normal desktop background activity was not otherwise controlled.",
    }


def invoke_worker(task: dict, input_sha256: str) -> Path:
    workers_dir = EVIDENCE_DIR / "workers"
    workers_dir.mkdir(exist_ok=True)
    runtime = task["runtime"]
    repeat_id = task["repeat_id"]
    run_order = task["run_order"]
    output_path = workers_dir / f"{run_order:02d}_{runtime}_repeat_{repeat_id}.csv"
    worker_env = os.environ.copy()
    worker_env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    if runtime in {"pytorch", "ort_python"}:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-runtime", runtime,
                   "--worker-output", str(output_path), "--repeat-id", str(repeat_id), "--run-order", str(run_order),
                   "--input", str(CANONICAL_INPUT)]
    else:
        command = [str(CPP_EXE), "--model", relative_to_root(MODEL_PATH), "--input-bin", relative_to_root(CANONICAL_INPUT),
                   "--labels", relative_to_root(LABELS_PATH), "--benchmark-output", relative_to_root(output_path),
                   "--warmup", str(WARMUP), "--runs", str(RUNS), "--repeat-id", str(repeat_id),
                   "--run-order", str(run_order), "--expected-top1", "285"]
    print("Worker command:", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, env=worker_env)
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise RuntimeError(f"Worker failed: runtime={runtime}, repeat={repeat_id}, exit={completed.returncode}")
    with output_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != RUNS or any(int(row["repeat_id"]) != repeat_id for row in rows):
        raise RuntimeError(f"Worker CSV is invalid: {output_path}")
    return output_path


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def aggregate(tasks: list[dict], worker_paths: dict[tuple[str, int], Path], input_sha256: str) -> None:
    raw_rows = []
    for task in tasks:
        with worker_paths[(task["runtime"], task["repeat_id"])].open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                raw_rows.append({
                    "runtime": task["runtime"], "repeat_id": task["repeat_id"], "iteration": int(row["iteration"]),
                    "latency_ms": float(row["latency_ms"]), "batch_size": 1, "threads": 1, "logical_cpu": 0,
                    "model_sha256": EXPECTED_MODEL_SHA256, "input_sha256": input_sha256, "run_order": task["run_order"],
                })
    if len(raw_rows) != 4500:
        raise RuntimeError(f"raw_latency.csv must contain 4500 rows, got {len(raw_rows)}")
    raw_path = EVIDENCE_DIR / "raw_latency.csv"
    fields = ["runtime", "repeat_id", "iteration", "latency_ms", "batch_size", "threads", "logical_cpu", "model_sha256", "input_sha256", "run_order"]
    with raw_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    repeat_rows = []
    summary_rows = []
    runtime_main = {}
    for runtime in ("pytorch", "ort_python", "ort_cpp"):
        runtime_rows = [row for row in raw_rows if row["runtime"] == runtime]
        if len(runtime_rows) != 1500:
            raise RuntimeError(f"{runtime} does not have 1500 measured rows")
        repeat_medians = []
        for repeat_id in range(1, REPEATS + 1):
            values = [row["latency_ms"] for row in runtime_rows if row["repeat_id"] == repeat_id]
            if len(values) != RUNS:
                raise RuntimeError(f"{runtime} repeat {repeat_id} does not have 300 measured rows")
            median = statistics.median(values)
            repeat_medians.append(median)
            repeat_rows.append({
                "runtime": runtime, "repeat_id": repeat_id, "count": len(values), "median_ms": median,
                "mean_ms": statistics.fmean(values), "p90_ms": percentile(values, 90), "p95_ms": percentile(values, 95),
                "p99_ms": percentile(values, 99), "std_ms": statistics.stdev(values), "min_ms": min(values), "max_ms": max(values),
            })
        main_latency = statistics.median(repeat_medians)
        runtime_main[runtime] = main_latency
        all_values = [row["latency_ms"] for row in runtime_rows]
        repeat_cv = statistics.stdev(repeat_medians) / statistics.fmean(repeat_medians) * 100.0
        summary_rows.append({
            "runtime": runtime, "measured_count": len(all_values), "primary_latency_median_of_repeat_medians_ms": main_latency,
            "throughput_images_per_s": 1000.0 / main_latency, "all_samples_p50_ms": percentile(all_values, 50),
            "all_samples_p90_ms": percentile(all_values, 90), "all_samples_p95_ms": percentile(all_values, 95),
            "all_samples_p99_ms": percentile(all_values, 99), "repeat_median_mean_ms": statistics.fmean(repeat_medians),
            "repeat_median_std_ms": statistics.stdev(repeat_medians), "repeat_median_cv_percent": repeat_cv,
        })
    for row in summary_rows:
        row["speedup_vs_pytorch_primary"] = runtime_main["pytorch"] / row["primary_latency_median_of_repeat_medians_ms"]

    for filename, rows in (("repeat_summary.csv", repeat_rows), ("benchmark_summary.csv", summary_rows)):
        with (EVIDENCE_DIR / filename).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "protocol_path": str(PROTOCOL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "raw_latency_row_count": len(raw_rows),
        "runtime_summaries": summary_rows,
        "speedups_primary_latency": {
            "pytorch_over_ort_python": runtime_main["pytorch"] / runtime_main["ort_python"],
            "pytorch_over_ort_cpp": runtime_main["pytorch"] / runtime_main["ort_cpp"],
            "ort_python_over_ort_cpp": runtime_main["ort_python"] / runtime_main["ort_cpp"],
        },
        "stability_pass": all(row["repeat_median_cv_percent"] <= 10.0 for row in summary_rows),
    }
    (EVIDENCE_DIR / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("raw_latency_rows=", len(raw_rows))
    print("stability_pass=", summary["stability_pass"])
    if not summary["stability_pass"]:
        raise RuntimeError("Repeat median CV exceeds frozen 10% threshold; do not select or delete samples")


def run_coordinator() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    check_frozen_protocol()
    if sha256(MODEL_PATH) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("S1 model SHA-256 does not match the frozen protocol")
    if not CPP_EXE.is_file() or not CPP_DLL.is_file() or not LABELS_PATH.is_file():
        raise FileNotFoundError("C++ executable, DLL, or labels file is missing")
    if sha256(CPP_DLL) != EXPECTED_CPP_DLL_SHA256:
        raise RuntimeError("C++ onnxruntime.dll SHA-256 does not match S2")
    input_sha256 = ensure_canonical_input()
    (EVIDENCE_DIR / "s4_environment.log").write_text(
        json.dumps(environment_snapshot(input_sha256), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tasks = [{"runtime": runtime, "repeat_id": repeat_id} for repeat_id in range(1, REPEATS + 1)
             for runtime in ("pytorch", "ort_python", "ort_cpp")]
    random.Random(SCHEDULE_SEED).shuffle(tasks)
    for run_order, task in enumerate(tasks, start=1):
        task["run_order"] = run_order
    (EVIDENCE_DIR / "run_schedule.json").write_text(json.dumps({"seed": SCHEDULE_SEED, "tasks": tasks}, indent=2) + "\n", encoding="utf-8")
    worker_paths = {}
    for task in tasks:
        worker_paths[(task["runtime"], task["repeat_id"])] = invoke_worker(task, input_sha256)
    aggregate(tasks, worker_paths, input_sha256)


def post_validate() -> None:
    """Validate existing S4 artifacts without starting workers or adding latency data."""
    print("S4 post-measurement validation (no benchmark workers started)")
    print("Original coordinator command: .venv\\Scripts\\python.exe scripts\\run_s4_fp32_benchmark.py")
    check_frozen_protocol()
    with (EVIDENCE_DIR / "raw_latency.csv").open(newline="", encoding="utf-8") as source:
        raw_rows = list(csv.DictReader(source))
    if len(raw_rows) != 4500:
        raise RuntimeError(f"Expected exactly 4500 raw measured rows, got {len(raw_rows)}")
    expected_fields = {"runtime", "repeat_id", "iteration", "latency_ms", "batch_size", "threads", "logical_cpu", "model_sha256", "input_sha256", "run_order"}
    if set(raw_rows[0]) != expected_fields:
        raise RuntimeError("raw_latency.csv columns do not match the frozen protocol")
    for runtime in ("pytorch", "ort_python", "ort_cpp"):
        runtime_rows = [row for row in raw_rows if row["runtime"] == runtime]
        if len(runtime_rows) != 1500:
            raise RuntimeError(f"{runtime} does not have 1500 measured rows")
        if {int(row["repeat_id"]) for row in runtime_rows} != {1, 2, 3, 4, 5}:
            raise RuntimeError(f"{runtime} repeat IDs are incomplete")
    if any(row["batch_size"] != "1" or row["threads"] != "1" or row["logical_cpu"] != "0" for row in raw_rows):
        raise RuntimeError("raw_latency.csv violates frozen batch/thread/affinity fields")
    summary = json.loads((EVIDENCE_DIR / "benchmark_summary.json").read_text(encoding="utf-8"))
    if not summary["stability_pass"]:
        raise RuntimeError("Stored summary reports a stability failure")
    worker_csv_count = len(list((EVIDENCE_DIR / "workers").glob("*.csv")))
    print("protocol_status=frozen_before_measurement")
    print("raw_latency_rows=4500")
    print(f"worker_csv_count={worker_csv_count}")
    print("model_sha256=", sha256(MODEL_PATH))
    print("input_sha256=", sha256(CANONICAL_INPUT))
    print("cpp_dll_sha256=", sha256(CPP_DLL))
    print("stability_pass=True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-runtime", choices=("pytorch", "ort_python"))
    parser.add_argument("--worker-output")
    parser.add_argument("--repeat-id", type=int)
    parser.add_argument("--run-order", type=int)
    parser.add_argument("--input")
    parser.add_argument("--post-validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.post_validate:
        if args.worker_runtime:
            raise SystemExit("--post-validate cannot be combined with worker arguments")
        post_validate()
    elif args.worker_runtime:
        if not all((args.worker_output, args.repeat_id is not None, args.run_order is not None, args.input)):
            raise SystemExit("Python worker requires output, repeat ID, run order, and input")
        run_python_worker(args.worker_runtime, Path(args.input), Path(args.worker_output), args.repeat_id, args.run_order)
    else:
        run_coordinator()


if __name__ == "__main__":
    main()
