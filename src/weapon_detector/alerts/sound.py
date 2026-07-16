"""Audible alarm alert channel."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from .base import Alert, AlertChannel


class SoundAlertChannel(AlertChannel):
    """Play an audible alarm when a weapon is detected.

    If a ``sound_file`` is provided it is played via the :mod:`playsound`
    package (played on a background thread so monitoring is not blocked).
    Otherwise the terminal bell character is emitted as a lightweight fallback.
    """

    name = "sound"

    def __init__(self, enabled: bool = True, sound_file: str | None = None) -> None:
        self._enabled = enabled
        self.sound_file = sound_file

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _play_file(self, path: str) -> None:
        try:
            from playsound import playsound

            playsound(path)
        except Exception:  # pragma: no cover - environment dependent
            # Fall back to the terminal bell if playback fails.
            sys.stdout.write("\a")
            sys.stdout.flush()

    def send(self, alert: Alert) -> None:
        if self.sound_file and Path(self.sound_file).exists():
            threading.Thread(
                target=self._play_file, args=(self.sound_file,), daemon=True
            ).start()
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
