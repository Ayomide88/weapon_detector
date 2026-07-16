from __future__ import annotations

import numpy as np

from weapon_detector.detection import Detection
from weapon_detector.visualization import draw_banner, draw_detections


def test_draw_detections_does_not_mutate_input(blank_frame):
    original = blank_frame.copy()
    dets = [Detection(label="pistol", confidence=0.9, box=(10, 10, 50, 50))]
    out = draw_detections(blank_frame, dets)
    assert np.array_equal(blank_frame, original)  # input untouched
    assert not np.array_equal(out, original)      # boxes drawn


def test_draw_detections_empty_returns_copy(blank_frame):
    out = draw_detections(blank_frame, [])
    assert np.array_equal(out, blank_frame)
    assert out is not blank_frame


def test_draw_banner_marks_top(blank_frame):
    out = draw_banner(blank_frame, "WEAPON DETECTED")
    assert out.shape == blank_frame.shape
    assert out[:40].sum() > 0  # banner drawn in the top region
