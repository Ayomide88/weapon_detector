from __future__ import annotations

import csv
from pathlib import Path

from weapon_detector.detection import Detection
from weapon_detector.evidence import EventLogger, EvidenceStore


def test_evidence_store_saves_image(tmp_path, blank_frame):
    store = EvidenceStore(tmp_path / "evidence")
    path = store.save(blank_frame)
    assert path.exists()
    assert path.suffix == ".jpg"
    assert path.parent == tmp_path / "evidence"


def test_event_logger_writes_header_and_rows(tmp_path):
    log_file = tmp_path / "detections.csv"
    logger = EventLogger(log_file)
    detections = [
        Detection(label="pistol", confidence=0.91, box=(0, 0, 10, 10)),
        Detection(label="knife", confidence=0.77, box=(1, 1, 5, 5)),
    ]
    logger.log("0", detections, evidence_path=Path("output/e.jpg"))

    with log_file.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    assert rows[0] == ["timestamp", "source", "label", "confidence", "evidence_path"]
    assert len(rows) == 3
    assert rows[1][2] == "pistol"
    assert rows[1][3] == "0.9100"
    assert rows[2][2] == "knife"


def test_event_logger_appends(tmp_path):
    log_file = tmp_path / "log.csv"
    logger = EventLogger(log_file)
    det = [Detection(label="knife", confidence=0.8, box=(0, 0, 1, 1))]
    logger.log("0", det)
    logger.log("0", det)
    with log_file.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 3  # header + 2 events
