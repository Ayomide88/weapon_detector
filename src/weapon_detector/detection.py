"""Weapon detection using a YOLOv8 model.

The :class:`WeaponDetector` wraps an Ultralytics YOLO model and exposes a simple
``detect`` method that returns only detections whose class name is configured as
a weapon and whose confidence exceeds the configured threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from .config import ModelConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Detection:
    """A single weapon detection within a frame."""

    label: str
    confidence: float
    # Bounding box in pixel coordinates: (x1, y1, x2, y2).
    box: tuple[int, int, int, int]


class _YOLOLike(Protocol):
    names: dict[int, str]

    def __call__(self, source: object, **kwargs: object) -> object: ...


class WeaponDetector:
    """Detect weapons in image frames with a YOLOv8 model."""

    def __init__(self, config: ModelConfig, model: _YOLOLike | None = None) -> None:
        self.config = config
        self._weapon_classes = {c.lower() for c in config.weapon_classes}
        # Loaded lazily on first use so constructing the detector (and thus the
        # GUI) is instant; the heavy torch/model load happens off the main thread.
        self._model = model

    def _get_model(self) -> _YOLOLike:
        if self._model is None:
            self._model = self._load_model(self.config)
        return self._model

    @staticmethod
    def _load_model(config: ModelConfig) -> _YOLOLike:
        # Imported lazily so tests (and non-inference tooling) do not require the
        # heavyweight ultralytics/torch stack to be installed.
        from ultralytics import YOLO

        model = YOLO(config.weights)
        return cast(_YOLOLike, model)

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._get_model().names)

    def is_weapon_label(self, label: str) -> bool:
        return label.lower() in self._weapon_classes

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run detection on ``frame`` and return weapon detections above threshold."""
        results = self._get_model()(
            frame,
            verbose=False,
            conf=self.config.confidence_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
        )
        return self._parse_results(results)

    def _parse_results(self, results: object) -> list[Detection]:
        detections: list[Detection] = []
        names = self.class_names
        # Ultralytics returns a list of Results; each has a ``boxes`` attribute.
        for result in _as_iterable(results):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(_scalar(box.cls))
                confidence = float(_scalar(box.conf))
                if confidence < self.config.confidence_threshold:
                    continue
                label = names.get(cls_id, str(cls_id))
                if not self.is_weapon_label(label):
                    continue
                xyxy = _to_list(box.xyxy)
                x1, y1, x2, y2 = (int(round(v)) for v in xyxy[:4])
                detections.append(
                    Detection(label=label, confidence=confidence, box=(x1, y1, x2, y2))
                )
        return detections


def _as_iterable(results: object) -> list[object]:
    if isinstance(results, (list, tuple)):
        return list(results)
    return [results]


def _scalar(value: object) -> float:
    """Extract a scalar from a tensor / numpy array / python number."""
    item = getattr(value, "item", None)
    if callable(item):
        return float(item())
    arr = np.asarray(value).ravel()
    return float(arr[0])


def _to_list(value: object) -> list[float]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        flat = np.asarray(tolist()).ravel()
        return [float(v) for v in flat]
    return [float(v) for v in np.asarray(value).ravel()]
