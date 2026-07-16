"""Tkinter GUI for the weapon detection system.

Provides a live video panel, Start/Stop controls, a scrolling status/log area
and a panel showing the most recent evidence image. Tkinter and Pillow are only
imported when the GUI is actually launched so the rest of the package can be
used headlessly (and tested) without a display.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

from .app import FrameResult, WeaponDetectionApp
from .config import AppConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class WeaponDetectorGUI:
    """A simple Tkinter front-end around :class:`WeaponDetectionApp`."""

    def __init__(self, config: AppConfig, app: WeaponDetectionApp | None = None) -> None:
        import tkinter as tk
        from tkinter import scrolledtext

        self.config = config
        self.app = app or WeaponDetectionApp(config)

        self._tk = tk
        self._monitoring = False
        self._worker: threading.Thread | None = None
        self._result_queue: queue.Queue[FrameResult] = queue.Queue(maxsize=4)
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
        self._log("Monitoring started.")
        self._worker = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker.start()
        self.root.after(30, self._drain_queue)

    def stop(self) -> None:
        if not self._monitoring:
            return
        self._monitoring = False
        self.start_button.configure(state=self._tk.NORMAL)
        self.stop_button.configure(state=self._tk.DISABLED)
        self.status_var.set("Idle")
        self._log("Monitoring stopped.")

    def _capture_loop(self) -> None:
        def on_frame(result: FrameResult) -> bool:
            if not self._monitoring:
                return False
            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                pass  # drop frame if UI can't keep up
            return self._monitoring

        try:
            self.app.run(on_frame=on_frame)
        except Exception as exc:  # noqa: BLE001 - surface errors in the UI
            logger.exception("Capture loop failed")
            message = f"ERROR: {exc}"
            self.root.after(0, lambda: self._log(message))
        finally:
            self.root.after(0, self.stop)

    # -- rendering -------------------------------------------------------
    def _drain_queue(self) -> None:
        try:
            while True:
                result = self._result_queue.get_nowait()
                self._render(result)
        except queue.Empty:
            pass
        if self._monitoring:
            self.root.after(30, self._drain_queue)

    def _to_photo(self, frame: NDArray[np.uint8], width: int, height: int):
        import cv2
        from PIL import Image, ImageTk

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((width, height))
        return ImageTk.PhotoImage(image)

    def _render(self, result: FrameResult) -> None:
        self._photo = self._to_photo(result.frame, 640, 480)
        # width/height were set in text units for the placeholder; once an image
        # is shown tk treats them as pixels, so size the label to the image.
        self.video_label.configure(
            image=self._photo,
            text="",
            width=self._photo.width(),
            height=self._photo.height(),
        )
        if result.has_weapon:
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

    def _on_close(self) -> None:
        self.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch(config: AppConfig) -> None:
    WeaponDetectorGUI(config).run()
