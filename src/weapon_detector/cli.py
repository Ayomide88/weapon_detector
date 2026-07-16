"""Command-line entry point for the weapon detection system."""

from __future__ import annotations

import argparse
import logging
import sys

from .app import FrameResult, WeaponDetectionApp
from .config import AppConfig, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weapon-detector",
        description="Real-time weapon detection and alert system.",
    )
    parser.add_argument(
        "-c", "--config", default=None, help="Path to a YAML config file."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Override the video source (webcam index or RTSP/HTTP URL).",
    )
    parser.add_argument(
        "--weights", default=None, help="Override the YOLO model weights path."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override the detection confidence threshold.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the GUI (console output only).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing this many frames (useful for testing).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.source is not None:
        config.video.source = args.source
    if args.weights is not None:
        config.model.weights = args.weights
    if args.confidence is not None:
        config.model.confidence_threshold = args.confidence
    return config


def run_headless(app: WeaponDetectionApp, max_frames: int | None) -> None:
    logger = logging.getLogger("weapon_detector.cli")
    logger.info("Active alert channels: %s", app.alert_manager.active_channels)
    logger.info("Monitoring source %s ...", app.config.video.source)

    def on_frame(result: FrameResult) -> bool:
        if result.has_weapon:
            labels = ", ".join(sorted({d.label for d in result.detections}))
            conf = max(d.confidence for d in result.detections)
            logger.warning("Weapon detected: %s (%.2f)", labels, conf)
        return True

    app.run(on_frame=on_frame, max_frames=max_frames)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = apply_overrides(load_config(args.config), args)

    if args.headless:
        app = WeaponDetectionApp(config)
        try:
            run_headless(app, args.max_frames)
        except RuntimeError as exc:
            logging.getLogger("weapon_detector.cli").error("%s", exc)
            return 1
        return 0

    # GUI mode (default).
    from .gui import launch

    launch(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
