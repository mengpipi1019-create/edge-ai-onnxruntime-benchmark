"""Create the fixed, deterministic five-image S3 alignment sample set."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, __version__ as PILLOW_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JPEG_SAVE = {
    "format": "JPEG",
    "quality": 95,
    "subsampling": 0,
    "optimize": False,
    "progressive": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="images/s3_alignment")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_transformed(source: Path, target: Path, transform: str) -> tuple[int, int, dict]:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    if transform == "horizontal_flip":
        transformed = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        transform_parameters = {"operation": "horizontal_flip"}
    elif transform == "center_crop_75_percent":
        left = image.width // 8
        top = image.height // 8
        right = image.width - left
        bottom = image.height - top
        transformed = image.crop((left, top, right, bottom))
        transform_parameters = {
            "operation": "center_crop",
            "box": [left, top, right, bottom],
            "retained_fraction_per_axis": 0.75,
        }
    else:
        raise ValueError(f"Unknown transform: {transform}")
    transformed.save(target, **JPEG_SAVE)
    return transformed.width, transformed.height, transform_parameters


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cat = PROJECT_ROOT / "images" / "cat.jpg"
    test = PROJECT_ROOT / "images" / "test.jpg"
    for source in (cat, test):
        if not source.is_file():
            raise FileNotFoundError(f"Missing S3 source image: {source}")

    samples = []
    originals = (("cat_original", cat), ("test_original", test))
    for sample_id, source in originals:
        target = output_dir / f"{sample_id}.jpg"
        shutil.copyfile(source, target)
        with Image.open(target) as image:
            width, height = image.size
        samples.append(
            {
                "sample_id": sample_id,
                "file": str(target.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source": str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "transform": {"operation": "byte_for_byte_copy"},
                "width": width,
                "height": height,
                "sha256": sha256(target),
            }
        )

    derived = (
        ("cat_hflip", cat, "horizontal_flip"),
        ("cat_center_crop", cat, "center_crop_75_percent"),
        ("test_hflip", test, "horizontal_flip"),
    )
    for sample_id, source, transform in derived:
        target = output_dir / f"{sample_id}.jpg"
        width, height, transform_parameters = save_transformed(source, target, transform)
        samples.append(
            {
                "sample_id": sample_id,
                "file": str(target.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source": str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "transform": transform_parameters,
                "width": width,
                "height": height,
                "sha256": sha256(target),
            }
        )

    manifest = {
        "purpose": "S3 runtime-consistency samples; derived images are not independent accuracy data.",
        "pillow_version": PILLOW_VERSION,
        "jpeg_save_parameters": JPEG_SAVE,
        "samples": samples,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} deterministic S3 samples to {output_dir}")
    print(f"Manifest: {manifest_path}")
    for sample in samples:
        print(f"{sample['sample_id']}: {sample['sha256']} ({sample['width']}x{sample['height']})")


if __name__ == "__main__":
    main()
