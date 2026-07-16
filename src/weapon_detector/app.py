"""High-level orchestration of the weapon detection pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from .alerts import Alert, AlertManager
from .config import AppConfig
from .detection import Detection, WeaponDetector
from .evidence import EventLogger, EvidenceStore
from .video import VideoStream
from .visualization import draw_banner, draw_detections

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Outcome of processing a single frame."""

    frame: NDArray[np.uint8]
    detections: list[Detection]
    alerted: bool = False

    @property
    def has_weapon(self) -> bool:
        return bool(self.detections)


class WeaponDetectionApp:
    """Wire together detection, evidence capture, logging and alerting.

    This class is UI-agnostic: the Tkinter GUI and the CLI both drive it. Frames
    can be fed one at a time via :meth:`process_frame` (useful for the GUI event
    loop and for tests) or consumed from a :class:`VideoStream` via :meth:`run`.
    """

    def __init__(
        self,
        config: AppConfig,
        detector: WeaponDetector | None = None,
        alert_manager: AlertManager | None = None,
        evidence_store: EvidenceStore | None = None,
        event_logger: EventLogger | None = None,
    ) -> None:
        self.config = config
        self.detector = detector or WeaponDetector(config.model)
        self.alert_manager = alert_manager or AlertManager.from_config(config.alerts)
        self.evidence_store = evidence_store or EvidenceStore(config.storage.evidence_dir)
        self.event_logger = event_logger or EventLogger(config.storage.log_file)
        self._frame_index = 0

    def process_frame(self, frame: NDArray[np.uint8]) -> FrameResult:
        """Detect weapons in ``frame`` and trigger alerts / logging as needed."""
        self._frame_index += 1
        step = max(1, self.config.video.process_every_n_frames)
        if self._frame_index % step != 0:
            return FrameResult(frame=frame, detections=[])

        try:
            detections = self.detector.detect(frame)
        except Exception:
            # A single malformed frame (occasionally produced by some webcam
            # backends) must not take down a long-running monitoring session.
            logger.exception("Detection failed on a frame; skipping it")
            return FrameResult(frame=frame, detections=[])
        if not detections:
            return FrameResult(frame=frame, detections=[])

        annotated = draw_detections(frame, detections)
        annotated = draw_banner(annotated, "WEAPON DETECTED")
        evidence_path = self.evidence_store.save(annotated)
        self.event_logger.log(self.config.video.source, detections, evidence_path)

        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            source=self.config.video.source,
            detections=detections,
            evidence_path=evidence_path,
        )
        alerted = self.alert_manager.dispatch(alert)
        return FrameResult(frame=annotated, detections=detections, alerted=alerted)

    def run(
        self,
        on_frame: Callable[[FrameResult], bool] | None = None,
        max_frames: int | None = None,
    ) -> None:
        """Consume frames from the configured video source until it ends.

        ``on_frame`` is called with each :class:`FrameResult`; returning
        ``False`` stops the loop (used by the GUI / for graceful shutdown).
        """
        processed = 0
        with VideoStream(self.config.video) as stream:
            for frame in stream.frames():
                result = self.process_frame(frame)
                processed += 1
                if on_frame is not None and on_frame(result) is False:
                    break
                if max_frames is not None and processed >= max_frames:
                    break
