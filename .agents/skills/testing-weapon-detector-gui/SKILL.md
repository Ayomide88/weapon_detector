---
name: testing-weapon-detector-gui
description: End-to-end runtime test of the Real-Time Weapon Detection Tkinter GUI (and headless CLI). Use when verifying detection/alert/evidence/logging changes at runtime.
---

# Testing the Weapon Detector GUI end-to-end

The app ingests video -> YOLOv8 detect -> draw boxes/banner -> save evidence JPEG -> log CSV -> dispatch alerts. Frontends: `weapon-detector` (Tkinter GUI, default) and `--headless` CLI.

## Environment prerequisites
- Python venv at `.venv` (see blueprint). `tkinter` is required for the GUI and is NOT installed by default: `sudo apt-get install -y python3-tk`, then verify `.venv/bin/python -c "import tkinter"`.
- GUI needs an X display (`DISPLAY=:0` on the VM desktop).
- Lint/test: `.venv/bin/ruff check .`, `.venv/bin/mypy src`, `.venv/bin/python -m pytest`.

## Key constraint: no webcam, no firearm model on the VM
The default `yolov8n.pt` (COCO) detects `knife` but NOT firearms, and there's no camera. To exercise the FULL pipeline deterministically, use a **proxy weapon class** that pretrained YOLO detects reliably (`person`) against a sample video. This is a **pipeline/runtime test, not firearm-accuracy validation** — always report it as such.

Create a test config (e.g. `testdata/test_config.yaml`):
```yaml
model: {weights: yolov8n.pt, confidence_threshold: 0.5, device: cpu, weapon_classes: [person]}
video: {source: "testdata/sample.mp4", fps_limit: 6, process_every_n_frames: 1}
alerts: {cooldown_seconds: 2, sound_enabled: true, desktop_enabled: true, twilio: {enabled: false}, email: {enabled: false}}
storage: {evidence_dir: output/evidence, log_file: output/detections.csv}
```
Generate a sample video with people (ultralytics `bus.jpg` works well). Prefer a **landscape** canvas (e.g. 1280x720) and enough frames (~150) at a low `fps_limit` so the GUI's live feed is large and monitoring lasts long enough to record. Source URL: `https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg` (the short `ultralytics.com/images/bus.jpg` 308-redirects).

## Run
- GUI: `DISPLAY=:0 setsid .venv/bin/weapon-detector --config testdata/test_config.yaml` (use `setsid ... & disown`; do NOT `pkill -f weapon-detector` — the pattern matches your own shell command and kills it. Kill by PID: `for p in $(pgrep -f '[w]eapon-detector'); do kill $p; done`, but note even the bracket regex can match a shell command containing the literal string, so prefer killing the known PID).
- Headless (deterministic pipeline check, no display): `.venv/bin/weapon-detector --headless --config testdata/test_config.yaml --max-frames 5`.

## Assertions (primary flow)
1. Idle state: status `Idle`, Start enabled, Stop disabled, empty feed/log/evidence.
2. Click Start Monitoring -> live feed shows red boxes + `<label> 0.xx` + red `WEAPON DETECTED` banner; status `Monitoring`; log shows `Weapon detected: <label> (0.xx)`; "Last evidence" panel updates.
3. Click Stop Monitoring -> status `Idle`, `Monitoring stopped.` logged.
4. On disk: `output/evidence/*.jpg` (annotated) and `output/detections.csv` (header `timestamp,source,label,confidence,evidence_path` + rows).

## Known gotchas
- **tk.Label image sizing:** a Label created with `width=`/`height=` in text units treats them as PIXELS once an image is assigned, clipping the image to a tiny box. When showing frames, set the label width/height to the image's pixel size (`photo.width()/height()`). Regression-check that the live feed fills its panel, not a thin strip.
- **Desktop notifications** (`plyer`) fail in a headless VM (no `notify-send`/dbus) — expected; `AlertManager` isolates the failure and the rest of the pipeline continues. Don't treat that logged traceback as a test failure.
- **Twilio/SMTP** delivery can't be tested without credentials; channel wiring is covered by unit tests. If added later, secrets live in env vars (`TWILIO_AUTH_TOKEN`, `WD_SMTP_PASSWORD`, etc.).

## Devin Secrets Needed
- None for the GUI/pipeline runtime test.
- Optional (only to test real alert delivery): `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `TWILIO_TO_NUMBER`; SMTP `WD_SMTP_HOST/PORT/USERNAME/WD_SMTP_PASSWORD`.
