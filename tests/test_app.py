from __future__ import annotations

from weapon_detector.alerts import AlertManager
from weapon_detector.app import WeaponDetectionApp
from weapon_detector.config import ModelConfig
from weapon_detector.detection import WeaponDetector
from weapon_detector.evidence import EventLogger, EvidenceStore


class SpyChannel:
    name = "spy"
    enabled = True

    def __init__(self):
        self.sent = []

    def send(self, alert):
        self.sent.append(alert)


def _build_app(config, model):
    detector = WeaponDetector(ModelConfig(confidence_threshold=0.6), model=model)
    spy = SpyChannel()
    manager = AlertManager([spy], cooldown_seconds=0)
    app = WeaponDetectionApp(
        config,
        detector=detector,
        alert_manager=manager,
        evidence_store=EvidenceStore(config.storage.evidence_dir),
        event_logger=EventLogger(config.storage.log_file),
    )
    return app, spy


def test_process_frame_no_weapon(config, make_yolo, blank_frame):
    model = make_yolo([(0, 0.99, (0, 0, 10, 10))])  # person only
    app, spy = _build_app(config, model)
    result = app.process_frame(blank_frame)
    assert result.has_weapon is False
    assert result.alerted is False
    assert spy.sent == []


def test_process_frame_with_weapon_triggers_pipeline(config, make_yolo, blank_frame):
    model = make_yolo([(1, 0.95, (10, 10, 100, 120))])  # pistol
    app, spy = _build_app(config, model)
    result = app.process_frame(blank_frame)

    assert result.has_weapon is True
    assert result.alerted is True
    assert len(spy.sent) == 1
    # Evidence image + CSV log written.
    evidence_dir = app.evidence_store.evidence_dir
    assert any(evidence_dir.iterdir())
    assert app.event_logger.log_file.exists()


def test_process_every_n_frames_skips(config, make_yolo, blank_frame):
    config.video.process_every_n_frames = 2
    model = make_yolo([(1, 0.95, (10, 10, 100, 120))])
    app, spy = _build_app(config, model)
    first = app.process_frame(blank_frame)   # index 1 -> skipped
    assert first.has_weapon is False
    second = app.process_frame(blank_frame)  # index 2 -> processed
    assert second.has_weapon is True


def test_process_frame_survives_detector_error(config, blank_frame):
    class ExplodingDetector:
        def detect(self, frame):
            raise RuntimeError("bad frame")

    spy = SpyChannel()
    app = WeaponDetectionApp(
        config,
        detector=ExplodingDetector(),
        alert_manager=AlertManager([spy], cooldown_seconds=0),
        evidence_store=EvidenceStore(config.storage.evidence_dir),
        event_logger=EventLogger(config.storage.log_file),
    )
    result = app.process_frame(blank_frame)
    assert result.has_weapon is False
    assert result.alerted is False
    assert spy.sent == []


def test_run_consumes_stream(config, make_yolo, blank_frame, monkeypatch):
    import weapon_detector.app as app_module

    model = make_yolo([(1, 0.95, (0, 0, 10, 10))])
    app, spy = _build_app(config, model)

    frames = [blank_frame, blank_frame, blank_frame]

    class FakeStream:
        def __init__(self, cfg):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def frames(self):
            yield from frames

    monkeypatch.setattr(app_module, "VideoStream", FakeStream)

    seen = []
    app.run(on_frame=lambda r: seen.append(r) or True, max_frames=2)
    assert len(seen) == 2
