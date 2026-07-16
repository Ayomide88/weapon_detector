"""Live video ingestion from webcams and IP (RTSP/HTTP) cameras."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from types import TracebackType
from typing import TYPE_CHECKING, cast

import cv2

from .config import VideoConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray


def _parse_source(source: str) -> str | int:
    """Interpret a source string as a webcam index or a stream URL."""
    try:
        return int(source)
    except (TypeError, ValueError):
        return source


def _open_capture(source: str | int) -> cv2.VideoCapture:
    """Open a capture, preferring backends that work reliably per platform.

    On Windows the default MSMF backend often fails sustained webcam capture
    ("can't grab frame" errors), so integer (webcam) sources are opened with
    DirectShow first, falling back to MSMF and then the default backend.
    """
    backends: list[int] = []
    if isinstance(source, int) and sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    for backend in backends:
        capture = cv2.VideoCapture(source, backend)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(source)


class VideoStream:
    """Wrapper around :class:`cv2.VideoCapture` with sane configuration.

    Usable as a context manager and as an iterator over frames.
    """

    def __init__(self, config: VideoConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> VideoStream:
        source = _parse_source(self.config.source)
        capture = _open_capture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.config.source!r}")
        if self.config.frame_width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        if self.config.frame_height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self._capture = capture
        return self

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def read(self) -> NDArray[np.uint8] | None:
        """Read a single frame, or ``None`` if the stream ended / failed."""
        if self._capture is None:
            raise RuntimeError("VideoStream is not open; call open() first")
        ok, frame = self._capture.read()
        if not ok:
            return None
        return cast("NDArray[np.uint8]", frame)

    def frames(self) -> Iterator[NDArray[np.uint8]]:
        """Yield frames until the stream ends, honoring the configured FPS limit."""
        if self._capture is None:
            self.open()
        min_interval = 0.0
        if self.config.fps_limit and self.config.fps_limit > 0:
            min_interval = 1.0 / self.config.fps_limit
        last = 0.0
        while True:
            frame = self.read()
            if frame is None:
                break
            if min_interval:
                elapsed = time.monotonic() - last
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last = time.monotonic()
            yield frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> VideoStream:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
