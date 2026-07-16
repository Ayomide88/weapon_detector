from __future__ import annotations

import textwrap

from weapon_detector.config import load_config


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.model.confidence_threshold == 0.6
    assert cfg.model.weights == "yolov8n.pt"
    assert cfg.video.source == "0"
    assert "knife" in {c.lower() for c in cfg.model.weapon_classes}


def test_loads_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            """
            model:
              weights: models/best.pt
              confidence_threshold: 0.8
            video:
              source: "rtsp://cam/stream"
            alerts:
              cooldown_seconds: 5
              twilio:
                enabled: true
                account_sid: AC123
            """
        ),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.model.weights == "models/best.pt"
    assert cfg.model.confidence_threshold == 0.8
    assert cfg.video.source == "rtsp://cam/stream"
    assert cfg.alerts.cooldown_seconds == 5
    assert cfg.alerts.twilio.enabled is True
    assert cfg.alerts.twilio.account_sid == "AC123"


def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("WD_CONFIDENCE_THRESHOLD", "0.42")
    monkeypatch.setenv("WD_VIDEO_SOURCE", "2")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACfromenv")
    monkeypatch.setenv("WD_EMAIL_TO", "a@x.com, b@x.com")
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.model.confidence_threshold == 0.42
    assert cfg.video.source == "2"
    assert cfg.alerts.twilio.account_sid == "ACfromenv"
    assert cfg.alerts.email.to_addresses == ["a@x.com", "b@x.com"]


def test_unknown_keys_ignored(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  bogus_key: 1\n  weights: foo.pt\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.model.weights == "foo.pt"
