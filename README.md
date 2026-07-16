# Real-Time Weapon Detection and Alert System

An intelligent surveillance prototype that detects weapons (firearms and knives)
in live video streams using **YOLOv8** and **OpenCV**, then raises **audio,
visual, desktop, email and SMS** alerts and logs evidence — reducing reliance on
manual monitoring and improving response times in places like schools, malls and
transport hubs.

> **Ethical use notice.** This project is for research and legitimate security
> purposes only. It has real limitations (false positives/negatives) and must be
> used responsibly and lawfully. See [Ethical Considerations](#ethical-considerations).

## Features

- **Live video capture** from a webcam or an IP camera over RTSP/HTTP.
- **YOLOv8 weapon detection** with configurable confidence threshold and a
  configurable set of weapon class names (guns, pistols, rifles, knives, ...).
- **On-screen localization** — bounding boxes, labels and confidence scores, plus
  an alert banner.
- **Real-time alerts** above a threshold with a cooldown to avoid flooding:
  - audible alarm (sound file or terminal bell fallback),
  - desktop pop-up notification (`plyer`),
  - SMS via **Twilio**,
  - email via **SMTP** (with the evidence image attached).
- **Evidence capture** — annotated screenshots saved to disk on every detection.
- **Event logging** to CSV (timestamp, source, label, confidence, evidence path).
- **Tkinter GUI** with a live feed, Start/Stop controls, a status/log area and a
  last-evidence panel — plus a **headless CLI** mode.

## Architecture

```
src/weapon_detector/
  config.py         # YAML + env-var configuration (dataclasses)
  detection.py      # WeaponDetector: YOLOv8 wrapper + weapon-class filtering
  video.py          # VideoStream: webcam / RTSP capture (context manager + iterator)
  visualization.py  # bounding boxes, labels, alert banner
  evidence.py       # EvidenceStore (screenshots) + EventLogger (CSV)
  alerts/           # AlertManager + sound / desktop / SMS / email channels
  app.py            # WeaponDetectionApp: ties detection -> evidence -> logging -> alerts
  gui.py            # Tkinter front-end
  cli.py            # `weapon-detector` command-line entry point
```

The pipeline is UI-agnostic: `WeaponDetectionApp.process_frame()` runs the full
detect → annotate → save evidence → log → alert flow for a single frame, and both
the GUI and the CLI drive it.

## Requirements

- Python 3.9+
- A webcam or an RTSP/HTTP IP camera (for live monitoring)
- A YOLOv8 model. The default `yolov8n.pt` (COCO) recognizes **`knife`** but
  **not firearms** — for gun/pistol/rifle detection you need a
  **weapon-fine-tuned** model (see [Training](#training-a-weapon-model)).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU-only PyTorch (skip the index URL if you have CUDA and want GPU wheels):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
# ...or install the package (editable) with dev extras:
pip install -e ".[dev]"
```

Optional file-based alarm playback: `pip install "playsound==1.2.2"`
(if not installed, the sound alert falls back to the terminal bell).

## Configuration

Configuration is layered: **defaults → YAML file → environment variables**
(env vars win, which is where secrets belong).

```bash
cp configs/config.example.yaml configs/config.yaml
cp .env.example .env               # fill in Twilio / SMTP secrets, then `source .env`
```

Key settings (`configs/config.yaml`):

| Section | Setting | Meaning |
|---|---|---|
| `model` | `weights` | Path to YOLO weights (`best.pt` for weapons) |
| `model` | `confidence_threshold` | Alert threshold, e.g. `0.6` |
| `model` | `weapon_classes` | Class names treated as weapons |
| `video` | `source` | Webcam index `"0"` or `rtsp://...` URL |
| `alerts` | `cooldown_seconds` | Minimum gap between alerts |
| `alerts.twilio` / `alerts.email` | `enabled`, credentials | Notification channels |
| `storage` | `evidence_dir`, `log_file` | Where evidence & logs are written |

Secrets are best supplied via env vars (see `.env.example`):
`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`,
`TWILIO_TO_NUMBER`, `WD_SMTP_HOST`, `WD_SMTP_USERNAME`, `WD_SMTP_PASSWORD`, etc.

## Usage

### GUI (default)

```bash
weapon-detector --config configs/config.yaml
# or: python -m weapon_detector.cli --config configs/config.yaml
```

Click **Start Monitoring** to begin; detections show bounding boxes, append to the
log area and update the evidence panel.

### Headless / CLI

```bash
# Webcam, custom weapons model, console output only:
weapon-detector --headless --source 0 --weights models/best.pt --confidence 0.6

# IP camera:
weapon-detector --headless --source "rtsp://user:pass@192.168.1.10:554/stream1"

# Process a fixed number of frames (handy for smoke tests):
weapon-detector --headless --max-frames 100
```

## Training a weapon model

The default COCO model **cannot detect firearms** (it only has a generic `knife`
class). To detect guns/pistols/rifles/knives reliably you must fine-tune YOLOv8
on a labelled weapon dataset. Training needs a **GPU** — if you don't have one,
use the included Google Colab notebook (free GPU).

### Option A — Google Colab (recommended, no local GPU needed)

Open `notebooks/train_weapon_model.ipynb` in
[Google Colab](https://colab.research.google.com/), select a GPU runtime, and run
the cells. It installs dependencies, downloads a dataset from Roboflow, trains,
reports accuracy, and downloads `best.pt`. Copy that file into the project (e.g.
`models/best.pt`).

### Option B — Local machine (needs an NVIDIA GPU)

1. **Get a dataset.** Grab a free labelled weapon dataset from
   [Roboflow Universe](https://universe.roboflow.com) (search "weapon detection",
   "pistol", "knife"). With a free [Roboflow API key](https://app.roboflow.com):

   ```bash
   pip install -e ".[train]"          # installs the roboflow client
   export ROBOFLOW_API_KEY=...        # from https://app.roboflow.com -> Settings -> API
   # A verified public pistol dataset (~2,970 images, class name `pistol`):
   python scripts/download_dataset.py \
     --workspace joseph-nelson --project pistols --version 1 \
     --location datasets/weapons
   ```

   This produces `datasets/weapons/data.yaml`. (Or point `--url` at any other
   Roboflow Universe dataset, or prepare your own YOLO-format dataset + YAML —
   see `data/weapon_dataset.example.yaml`.)

2. **Train** (auto-detects GPU/CPU):

   ```bash
   python scripts/train.py --data datasets/weapons/data.yaml --epochs 50 --imgsz 640
   ```

3. **Use the model.** Point the app at the resulting
   `runs/detect/weapon_yolov8/weights/best.pt`:

   ```bash
   weapon-detector --weights runs/detect/weapon_yolov8/weights/best.pt
   ```

   or set `model.weights` (or `WD_MODEL_WEIGHTS`) in your config. Make sure
   `model.weapon_classes` matches your dataset's class names.

## Testing

```bash
pip install -e ".[dev]"
pytest                 # unit tests (detector/alerts/video/etc. are mocked)
ruff check .           # lint
mypy src               # type-check
```

The unit tests mock the YOLO model, webcam and alert transports, so they run
without a GPU, camera, display or network.

### Performance validation

For accuracy metrics (precision / recall / mAP50), run `model.val()` on a held-out
test set (the training script prints these). Measure real-time performance (FPS,
latency) on your target hardware; use `video.process_every_n_frames` and
`video.fps_limit` to trade accuracy for throughput on CPU-only machines.

## Ethical considerations

- **Research/security use only.** Do not use this to harm, harass or unlawfully
  surveil people.
- **Not a safety-certified system.** It produces false positives and false
  negatives; a human should verify every alert before acting.
- **Privacy & law.** Comply with local laws and privacy regulations for video
  surveillance and data retention (evidence images and logs).
- Tune `confidence_threshold` and the weapon model to your environment to balance
  missed detections against false alarms.

## License

MIT — see `LICENSE`.
