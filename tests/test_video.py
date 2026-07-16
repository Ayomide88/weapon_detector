from __future__ import annotations

import numpy as np
import pytest

import weapon_detector.video as video_module
from weapon_detector.config import VideoConfig
from weapon_detector.video import VideoStream, _parse_source


def test_parse_source_index():
    assert _parse_source("0") == 0
    assert _parse_source("2") == 2


def test_parse_source_url():
    assert _parse_source("rtsp://cam/stream") == "rtsp://cam/stream"


class FakeCapture:
    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0
        self.released = False
        self.props = {}

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.props[prop] = value

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        frame = self._frames[self._i]
        self._i += 1
        return True, frame

    def release(self):
        self.released = True


def test_stream_iterates_frames(monkeypatch):
    frames = [np.zeros((2, 2, 3), np.uint8) for _ in range(3)]
    cap = FakeCapture(frames)
    monkeypatch.setattr(video_module.cv2, "VideoCapture", lambda src: cap)

    with VideoStream(VideoConfig(source="0")) as stream:
        collected = list(stream.frames())
    assert len(collected) == 3
    assert cap.released is True


def test_open_raises_when_capture_fails(monkeypatch):
    class ClosedCapture:
        def isOpened(self):
            return False

    monkeypatch.setattr(video_module.cv2, "VideoCapture", lambda src: ClosedCapture())
    with pytest.raises(RuntimeError):
        VideoStream(VideoConfig(source="9")).open()
