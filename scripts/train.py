"""Fine-tune a YOLOv8 model for weapon detection.

Usage:
    python scripts/train.py --data data/weapon_dataset.yaml --epochs 50

The resulting best weights are written to
``runs/detect/<name>/weights/best.pt``. Point the app at that file via
``model.weights`` in the config (or the ``WD_MODEL_WEIGHTS`` env var).
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for weapons.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base weights to start from.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help='e.g. "0", "cpu", "mps".')
    parser.add_argument("--name", default="weapon_yolov8", help="Run name.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )
    metrics = model.val()
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
