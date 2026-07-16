"""Fine-tune a YOLOv8 model for weapon detection.

Usage:
    python scripts/train.py --data data/weapon_dataset.yaml --epochs 50

The resulting best weights are written to
``runs/detect/<name>/weights/best.pt``. Point the app at that file via
``model.weights`` in the config (or the ``WD_MODEL_WEIGHTS`` env var).

Tip: training is GPU-intensive. On a machine without an NVIDIA GPU use Google
Colab (see ``notebooks/train_weapon_model.ipynb``) or expect it to be slow.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for weapons.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base weights to start from.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default=None,
        help='e.g. "0" (first GPU), "cpu", "mps". Default: auto-detect.',
    )
    parser.add_argument("--name", default="weapon_yolov8", help="Run name.")
    return parser


def _auto_device(device: str | None) -> str:
    """Pick a sensible device when the user did not specify one."""
    if device is not None:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ultralytics import YOLO

    device = _auto_device(args.device)
    if device == "cpu":
        print(
            "WARNING: training on CPU is very slow. Use a GPU (e.g. Google Colab, "
            "see notebooks/train_weapon_model.ipynb) for realistic training times."
        )
    print(f"Training on device: {device}")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
    )
    metrics = model.val()
    print(metrics)

    save_dir = getattr(results, "save_dir", None)
    if save_dir is not None:
        best = Path(save_dir) / "weights" / "best.pt"
        print(f"\nBest weights: {best}")
        print(
            "Use them with:\n"
            f"  weapon-detector --weights {best}\n"
            "or set model.weights in your config / the WD_MODEL_WEIGHTS env var."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
