"""Configuration loading for the weapon detection system.

Settings are read from a YAML file and can be overridden by environment
variables (useful for secrets such as Twilio / SMTP credentials). Environment
variables always take precedence over the YAML file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

DEFAULT_WEAPON_CLASSES = ["gun", "pistol", "rifle", "knife", "handgun", "weapon"]


def _env(name: str, default: Any = None) -> Any:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ModelConfig:
    """YOLOv8 model settings."""

    weights: str = "yolov8n.pt"
    confidence_threshold: float = 0.6
    device: str = "cpu"
    image_size: int = 640
    # Names (case-insensitive) of classes that should be treated as weapons.
    weapon_classes: list[str] = field(default_factory=lambda: list(DEFAULT_WEAPON_CLASSES))


@dataclass
class VideoConfig:
    """Video ingestion settings.

    ``source`` may be an integer webcam index (as a string, e.g. ``"0"``) or an
    RTSP / HTTP URL for an IP camera.
    """

    source: str = "0"
    frame_width: int | None = None
    frame_height: int | None = None
    fps_limit: float | None = None
    # Only run detection on every Nth frame to save CPU (1 = every frame).
    process_every_n_frames: int = 1


@dataclass
class TwilioConfig:
    enabled: bool = False
    account_sid: str = ""
    auth_token: str = ""
    from_number: str = ""
    to_number: str = ""


@dataclass
class EmailConfig:
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    use_tls: bool = True
    username: str = ""
    password: str = ""
    from_address: str = ""
    to_addresses: list[str] = field(default_factory=list)


@dataclass
class AlertConfig:
    # Minimum seconds between successive alerts to avoid flooding.
    cooldown_seconds: float = 30.0
    sound_enabled: bool = True
    sound_file: str | None = None
    desktop_enabled: bool = True
    twilio: TwilioConfig = field(default_factory=TwilioConfig)
    email: EmailConfig = field(default_factory=EmailConfig)


@dataclass
class StorageConfig:
    evidence_dir: str = "output/evidence"
    log_file: str = "output/detections.csv"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a (possibly nested) dataclass from a plain dict, ignoring unknown keys."""
    # ``from __future__ import annotations`` turns field types into strings, so
    # resolve them to real types to detect nested dataclasses.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = hints.get(f.name, f.type)
        if isinstance(field_type, type) and is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(field_type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    """Override secrets and common tunables from environment variables."""
    model = config.model
    model.weights = _env("WD_MODEL_WEIGHTS", model.weights)
    model.device = _env("WD_DEVICE", model.device)
    conf = _env("WD_CONFIDENCE_THRESHOLD")
    if conf is not None:
        model.confidence_threshold = float(conf)

    video = config.video
    video.source = _env("WD_VIDEO_SOURCE", video.source)

    twilio = config.alerts.twilio
    twilio.enabled = _env_bool("WD_TWILIO_ENABLED", twilio.enabled)
    twilio.account_sid = _env("TWILIO_ACCOUNT_SID", twilio.account_sid)
    twilio.auth_token = _env("TWILIO_AUTH_TOKEN", twilio.auth_token)
    twilio.from_number = _env("TWILIO_FROM_NUMBER", twilio.from_number)
    twilio.to_number = _env("TWILIO_TO_NUMBER", twilio.to_number)

    email = config.alerts.email
    email.enabled = _env_bool("WD_EMAIL_ENABLED", email.enabled)
    email.smtp_host = _env("WD_SMTP_HOST", email.smtp_host)
    port = _env("WD_SMTP_PORT")
    if port is not None:
        email.smtp_port = int(port)
    email.username = _env("WD_SMTP_USERNAME", email.username)
    email.password = _env("WD_SMTP_PASSWORD", email.password)
    email.from_address = _env("WD_EMAIL_FROM", email.from_address)
    to_addrs = _env("WD_EMAIL_TO")
    if to_addrs is not None:
        email.to_addresses = [a.strip() for a in to_addrs.split(",") if a.strip()]

    return config


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load configuration from ``path`` (YAML), then apply environment overrides.

    If ``path`` is ``None`` or does not exist, defaults are used (still subject
    to environment overrides).
    """
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file {p} must contain a mapping at the top level")
            data = loaded

    config = _from_dict(AppConfig, data) if data else AppConfig()
    return _apply_env_overrides(config)
