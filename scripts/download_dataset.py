"""Download a weapon-detection dataset from Roboflow Universe (YOLOv8 format).

Roboflow Universe (https://universe.roboflow.com) hosts many free, labelled
weapon datasets (search "weapon detection", "pistol", "knife"). Downloading one
gives you a ready-to-train folder containing a ``data.yaml`` plus
train/valid/test images and YOLO-format labels.

You need a free Roboflow API key: https://app.roboflow.com -> Settings -> API.
Provide it via ``--api-key`` or the ``ROBOFLOW_API_KEY`` environment variable.

Two ways to specify the dataset:

1. By Universe URL (copy the dataset's page URL)::

    python scripts/download_dataset.py \
        --url https://universe.roboflow.com/<workspace>/<project>/dataset/<version> \
        --location datasets/weapons

2. By explicit workspace / project / version (from the dataset's "Download
   Dataset" -> "show download code" snippet)::

    python scripts/download_dataset.py \
        --workspace <workspace> --project <project> --version <version> \
        --location datasets/weapons

Then train::

    python scripts/train.py --data datasets/weapons/data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path


def _images_in(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not directory.is_dir():
        return []
    return [p for p in directory.iterdir() if p.suffix.lower() in exts]


def _ensure_train_val_split(location: str, val_fraction: float = 0.15) -> None:
    """Guarantee ``train``/``valid`` image folders and a matching ``data.yaml``.

    Some Roboflow versions export a single flat folder (e.g. ``export/images`` +
    ``export/labels``) with no split, whose ``data.yaml`` points at non-existent
    ``train``/``val`` dirs. That fails training, so we build a split here.
    """
    import yaml

    root = Path(location)
    if _images_in(root / "train" / "images") and (
        _images_in(root / "valid" / "images") or _images_in(root / "val" / "images")
    ):
        return  # already split by Roboflow

    pool = next(
        (d for d in (root / "export" / "images", root / "images") if _images_in(d)),
        None,
    )
    if pool is None:
        return  # nothing we can split; leave the download untouched
    labels_dir = pool.parent / "labels"

    images = sorted(_images_in(pool))
    random.seed(42)
    random.shuffle(images)
    n_val = max(1, int(len(images) * val_fraction))
    val = set(images[:n_val])

    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for img in images:
        split = "valid" if img in val else "train"
        shutil.copy(img, root / split / "images" / img.name)
        label = labels_dir / (img.stem + ".txt")
        if label.exists():
            shutil.copy(label, root / split / "labels" / label.name)

    data = {}
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        data = yaml.safe_load(yaml_path.read_text()) or {}
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data.pop("test", None)
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    print(
        f"Split flat dataset into {len(images) - n_val} train / {n_val} val images "
        f"and updated {yaml_path}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a Roboflow dataset.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY).",
    )
    parser.add_argument("--url", default=None, help="Roboflow Universe dataset URL.")
    parser.add_argument("--workspace", default=None, help="Roboflow workspace slug.")
    parser.add_argument("--project", default=None, help="Roboflow project slug.")
    parser.add_argument("--version", type=int, default=None, help="Dataset version number.")
    parser.add_argument(
        "--location",
        default="datasets/weapons",
        help="Where to download the dataset (default: datasets/weapons).",
    )
    parser.add_argument(
        "--format", default="yolov8", help="Export format (default: yolov8)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.api_key:
        print(
            "ERROR: no API key. Pass --api-key or set ROBOFLOW_API_KEY "
            "(get one free at https://app.roboflow.com -> Settings -> API).",
            file=sys.stderr,
        )
        return 2

    try:
        import roboflow
    except ImportError:
        print(
            "ERROR: the 'roboflow' package is required. Install it with:\n"
            "  pip install roboflow",
            file=sys.stderr,
        )
        return 2

    if args.url:
        dataset = roboflow.download_dataset(
            dataset_url=args.url,
            model_format=args.format,
            location=args.location,
        )
    elif args.workspace and args.project and args.version is not None:
        rf = roboflow.Roboflow(api_key=args.api_key)
        project = rf.workspace(args.workspace).project(args.project)
        dataset = project.version(args.version).download(
            args.format, location=args.location
        )
    else:
        print(
            "ERROR: specify either --url, or all of --workspace/--project/--version.",
            file=sys.stderr,
        )
        return 2

    location = getattr(dataset, "location", args.location)
    _ensure_train_val_split(location)
    print(f"\nDataset downloaded to: {location}")
    print(f"Train with:\n  python scripts/train.py --data {location}/data.yaml --epochs 50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
