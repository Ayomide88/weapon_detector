"""Tkinter GUI for the weapon detection system.

Provides a live video panel, Start/Stop controls, a scrolling status/log area
and a panel showing the most recent evidence image. Tkinter and Pillow are only
imported when the GUI is actually launched so the rest of the package can be
used headlessly (and tested) without a display.

The video display is decoupled from detection: a capture thread keeps the latest
camera frame, a detection thread runs (slow, CPU-bound) inference on the most
recent frame, and the Tk event loop redraws the live feed at ~30 fps overlaying
the most recent detections. This keeps the feed smooth even when inference only
manages a frame or two per second on a CPU.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import TYPE_CHECKING

from .app import FrameResult, WeaponDetectionApp
from .config import AppConfig
from .detection import Detection
from .video import VideoStream
from .visualization import draw_banner, draw_detections

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Minimum seconds between successive "Weapon detected" log lines / evidence
# thumbnail refreshes, so the UI is not flooded when a target stays in view.
_UI_EVENT_INTERVAL = 1.0
# Target redraw interval for the live feed (~30 fps).
_DISPLAY_INTERVAL_MS = 33


class WeaponDetectorGUI:
    """A simple Tkinter front-end around :class:`WeaponDetectionApp`."""

    def __init__(self, config: AppConfig, app: WeaponDetectionApp | None = None) -> None:
        import tkinter as tk
        from tkinter import scrolledtext

        self.config = config
        self.app = app or WeaponDetectionApp(config)

        self._tk = tk
        self._monitoring = False
        self._capture_thread: threading.Thread | None = None
        self._detect_thread: threading.Thread | None = None

        self._lock = threading.Lock()
        self._latest_frame: NDArray[np.uint8] | None = None
        self._latest_detections: list[Detection] = []
        self._last_ui_event = 0.0

        self._photo = None  # keep a reference to avoid GC of the current frame image
        self._evidence_photo = None

        self.root = tk.Tk()
        self.root.title("Real-Time Weapon Detection and Alert System")

        self.video_label = tk.Label(
            self.root, text="Live feed (stopped)", width=80, height=24, bg="black", fg="white"
        )
        self.video_label.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

        self.start_button = tk.Button(self.root, text="Start Monitoring", command=self.start)
        self.start_button.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.stop_button = tk.Button(
            self.root, text="Stop Monitoring", command=self.stop, state=tk.DISABLED
        )
        self.stop_button.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        self.status_var = tk.StringVar(value="Idle")
        tk.Label(self.root, textvariable=self.status_var, anchor="w").grid(
            row=1, column=2, sticky="ew", padx=5, pady=5
        )

        self.log_area = scrolledtext.ScrolledText(
            self.root, height=8, width=60, state=tk.DISABLED
        )
        self.log_area.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        self.evidence_label = tk.Label(
            self.root, text="Last evidence", width=30, height=12, bg="gray20", fg="white"
        )
        self.evidence_label.grid(row=2, column=2, padx=5, pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- logging ---------------------------------------------------------
    def _log(self, message: str) -> None:
        tk = self._tk
        self.log_area.configure(state=tk.NORMAL)
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state=tk.DISABLED)

    # -- control ---------------------------------------------------------
    def start(self) -> None:
        if self._monitoring:
            return
        self._monitoring = True
        self.start_button.configure(state=self._tk.DISABLED)
        self.stop_button.configure(state=self._tk.NORMAL)
        self.status_var.set("Monitoring")
        self._log("Monitoring started (loading model on first detection)...")

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._capture_thread.start()
        self._detect_thread.start()
        self.root.after(_DISPLAY_INTERVAL_MS, self._update_display)

    def stop(self) -> None:
        if not self._monitoring:
            return
        self._monitoring = False
        self.start_button.configure(state=self._tk.NORMAL)
        self.stop_button.configure(state=self._tk.DISABLED)
        self.status_var.set("Idle")
        self._log("Monitoring stopped.")

    # -- worker threads --------------------------------------------------
    def _capture_loop(self) -> None:
        """Continuously read frames and keep only the most recent one."""
        # Capture as fast as the camera allows so the displayed feed is smooth;
        # detection throttling happens independently in the detection loop.
        capture_config = dataclasses.replace(self.config.video, fps_limit=None)
        try:
            with VideoStream(capture_config) as stream:
                for frame in stream.frames():
                    if not self._monitoring:
                        break
                    with self._lock:
                        self._latest_frame = frame
        except Exception as exc:  # noqa: BLE001 - surface errors in the UI
            logger.exception("Capture loop failed")
            message = f"ERROR: {exc}"
            self.root.after(0, lambda: self._log(message))
        finally:
            self.root.after(0, self.stop)

    def _detect_loop(self) -> None:
        """Run detection on the latest frame as fast as the CPU/GPU allows."""
        try:
            while self._monitoring:
                with self._lock:
                    frame = None if self._latest_frame is None else self._latest_frame.copy()
                if frame is None:
                    time.sleep(0.01)
                    continue
                result = self.app.process_frame(frame)
                with self._lock:
                    self._latest_detections = result.detections
                if result.has_weapon:
                    self.root.after(0, lambda r=result: self._on_detection(r))
        except Exception as exc:  # noqa: BLE001 - surface errors in the UI
            logger.exception("Detection loop failed")
            message = f"ERROR: {exc}"
            self.root.after(0, lambda: self._log(message))

    # -- rendering (main thread) -----------------------------------------
    def _update_display(self) -> None:
        with self._lock:
            frame = self._latest_frame
            detections = list(self._latest_detections)
        if frame is not None:
            display = frame
            if detections:
                display = draw_detections(frame, detections)
                display = draw_banner(display, "WEAPON DETECTED")
            self._photo = self._to_photo(display, 640, 480)
            # width/height were set in text units for the placeholder; once an
            # image is shown tk treats them as pixels, so size to the image.
            self.video_label.configure(
                image=self._photo,
                text="",
                width=self._photo.width(),
                height=self._photo.height(),
            )
        if self._monitoring:
            self.root.after(_DISPLAY_INTERVAL_MS, self._update_display)

    def _on_detection(self, result: FrameResult) -> None:
        now = time.monotonic()
        if now - self._last_ui_event < _UI_EVENT_INTERVAL:
            return
        self._last_ui_event = now
        labels = ", ".join(sorted({d.label for d in result.detections}))
        conf = max(d.confidence for d in result.detections)
        self._log(f"Weapon detected: {labels} ({conf:.2f})")
        self._evidence_photo = self._to_photo(result.frame, 240, 180)
        self.evidence_label.configure(
            image=self._evidence_photo,
            text="",
            width=self._evidence_photo.width(),
            height=self._evidence_photo.height(),
        )

    def _to_photo(self, frame: NDArray[np.uint8], width: int, height: int):
        import cv2
        from PIL import Image, ImageTk

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((width, height))
        return ImageTk.PhotoImage(image)

    def _on_close(self) -> None:
        self.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch(config: AppConfig) -> None:
    WeaponDetectorGUI(config).run()
