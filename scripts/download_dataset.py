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
import sys


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
    print(f"\nDataset downloaded to: {location}")
    print(f"Train with:\n  python scripts/train.py --data {location}/data.yaml --epochs 50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
