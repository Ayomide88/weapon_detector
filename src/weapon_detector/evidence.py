"""Evidence capture and CSV event logging."""

from __future__ import annotations

import csv
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from .detection import Detection

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray

_LOG_HEADER = ["timestamp", "source", "label", "confidence", "evidence_path"]


def _timestamp() -> datetime:
    return datetime.now(timezone.utc)


class EvidenceStore:
    """Save annotated frames to disk when weapons are detected."""

    def __init__(self, evidence_dir: str | Path) -> None:
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def save(self, frame: NDArray[np.uint8], *, prefix: str = "weapon") -> Path:
        ts = _timestamp().strftime("%Y%m%d_%H%M%S_%f")
        path = self.evidence_dir / f"{prefix}_{ts}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Failed to write evidence image to {path}")
        return path


class EventLogger:
    """Append detection events to a CSV file in a thread-safe manner."""

    def __init__(self, log_file: str | Path) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.log_file.exists():
            with self.log_file.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(_LOG_HEADER)

    def log(
        self,
        source: str,
        detections: Sequence[Detection],
        evidence_path: str | Path | None = None,
    ) -> None:
        ts = _timestamp().isoformat()
        evidence = str(evidence_path) if evidence_path is not None else ""
        with self._lock, self.log_file.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for det in detections:
                writer.writerow(
                    [ts, source, det.label, f"{det.confidence:.4f}", evidence]
                )
