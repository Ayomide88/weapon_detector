from __future__ import annotations

from weapon_detector.cli import apply_overrides, build_parser
from weapon_detector.config import AppConfig


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.config is None
    assert args.headless is False


def test_apply_overrides():
    args = build_parser().parse_args(
        ["--source", "5", "--weights", "best.pt", "--confidence", "0.75"]
    )
    cfg = apply_overrides(AppConfig(), args)
    assert cfg.video.source == "5"
    assert cfg.model.weights == "best.pt"
    assert cfg.model.confidence_threshold == 0.75


def test_headless_flag():
    args = build_parser().parse_args(["--headless", "--max-frames", "3"])
    assert args.headless is True
    assert args.max_frames == 3
