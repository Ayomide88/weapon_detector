"""Shared pytest fixtures and lightweight fakes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from weapon_detector.config import AppConfig


@dataclass
class _FakeBox:
    cls: int
    conf: float
    xyxy: list[float]


class _FakeResult:
    def __init__(self, boxes: Sequence[_FakeBox]) -> None:
        self.boxes = list(boxes)


class FakeYOLO:
    """A stand-in for ultralytics.YOLO that returns pre-programmed detections."""

    def __init__(self, names: dict[int, str], detections: Sequence[_FakeBox]) -> None:
        self.names = names
        self._detections = list(detections)
        self.calls: list[np.ndarray] = []

    def __call__(self, source, **kwargs):  # noqa: ANN001, ANN002
        self.calls.append(source)
        return [_FakeResult(self._detections)]


@pytest.fixture
def fake_box():
    return _FakeBox


@pytest.fixture
def make_yolo(fake_box):
    def _make(detections):
        names = {0: "person", 1: "pistol", 2: "knife", 43: "knife"}
        boxes = [fake_box(cls=c, conf=conf, xyxy=list(box)) for c, conf, box in detections]
        return FakeYOLO(names=names, detections=boxes)

    return _make


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def config(tmp_path):
    cfg = AppConfig()
    cfg.storage.evidence_dir = str(tmp_path / "evidence")
    cfg.storage.log_file = str(tmp_path / "detections.csv")
    cfg.model.confidence_threshold = 0.6
    return cfg
