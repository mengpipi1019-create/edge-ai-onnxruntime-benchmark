import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command):
    print("\n" + "=" * 80)
    print("Running:", " ".join(command))
    print("=" * 80)

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with return code {}: {}".format(
                result.returncode,
                " ".join(command)
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run the full PyTorch to ONNX Runtime inference pipeline."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="images/cat.jpg",
        help="Input image path."
    )
    parser.add_argument(
        "--skip_day2",
        action="store_true",
        help="Skip Day 2 PyTorch random-input smoke test."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    print("Project root:", project_root)

    image_path = project_root / args.image
    if not image_path.exists():
        raise FileNotFoundError("Input image not found: {}".format(image_path))

    python_exe = sys.executable

    if not args.skip_day2:
        run_command([python_exe, "scripts/day2_pytorch_inference.py"])

    run_command([
        python_exe,
        "scripts/day3_real_image_inference.py",
        "--image",
        args.image
    ])

    run_command([
        python_exe,
        "scripts/day4_export_onnx.py"
    ])

    run_command([
        python_exe,
        "scripts/day5_onnxruntime_inference.py",
        "--image",
        args.image
    ])

    run_command([
        python_exe,
        "scripts/day6_batch_benchmark.py",
        "--image",
        args.image
    ])

    run_command([
        python_exe,
        "scripts/day7_plot_curves.py"
    ])

    print("\nFull pipeline finished successfully.")


if __name__ == "__main__":
    main()
