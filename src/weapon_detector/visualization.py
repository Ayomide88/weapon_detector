"""Drawing helpers for annotating frames with weapon detections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from .detection import Detection

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray

_BOX_COLOR = (0, 0, 255)  # red (BGR)
_TEXT_COLOR = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(
    frame: NDArray[np.uint8], detections: list[Detection]
) -> NDArray[np.uint8]:
    """Return a copy of ``frame`` with bounding boxes and labels drawn."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        caption = f"{det.label} {det.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(caption, _FONT, 0.5, 1)
        top = max(y1 - text_h - baseline, 0)
        cv2.rectangle(
            annotated,
            (x1, top),
            (x1 + text_w, top + text_h + baseline),
            _BOX_COLOR,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            caption,
            (x1, top + text_h),
            _FONT,
            0.5,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    return annotated


def draw_banner(frame: NDArray[np.uint8], text: str) -> NDArray[np.uint8]:
    """Overlay a prominent alert banner across the top of the frame."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    cv2.rectangle(annotated, (0, 0), (width, 40), _BOX_COLOR, thickness=-1)
    cv2.putText(
        annotated, text, (10, 28), _FONT, 0.8, _TEXT_COLOR, 2, cv2.LINE_AA
    )
    return annotated
