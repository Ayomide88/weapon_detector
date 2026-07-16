from __future__ import annotations

from weapon_detector.config import ModelConfig
from weapon_detector.detection import Detection, WeaponDetector


def test_detects_only_weapon_classes(make_yolo, blank_frame):
    model = make_yolo(
        [
            (0, 0.99, (0, 0, 10, 10)),   # person -> ignored
            (1, 0.80, (5, 5, 50, 60)),   # pistol -> kept
        ]
    )
    detector = WeaponDetector(ModelConfig(confidence_threshold=0.6), model=model)
    detections = detector.detect(blank_frame)
    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det, Detection)
    assert det.label == "pistol"
    assert det.box == (5, 5, 50, 60)


def test_filters_below_threshold(make_yolo, blank_frame):
    model = make_yolo([(1, 0.40, (0, 0, 10, 10))])  # pistol below 0.6
    detector = WeaponDetector(ModelConfig(confidence_threshold=0.6), model=model)
    assert detector.detect(blank_frame) == []


def test_keeps_multiple_weapons(make_yolo, blank_frame):
    model = make_yolo(
        [
            (1, 0.90, (0, 0, 10, 10)),
            (2, 0.75, (20, 20, 40, 40)),
        ]
    )
    detector = WeaponDetector(ModelConfig(confidence_threshold=0.6), model=model)
    labels = {d.label for d in detector.detect(blank_frame)}
    assert labels == {"pistol", "knife"}


def test_is_weapon_label_case_insensitive(make_yolo):
    detector = WeaponDetector(
        ModelConfig(weapon_classes=["Pistol"]), model=make_yolo([])
    )
    assert detector.is_weapon_label("PISTOL")
    assert not detector.is_weapon_label("person")
