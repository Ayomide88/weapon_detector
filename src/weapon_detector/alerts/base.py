"""Base types for alert channels."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ..detection import Detection

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path


@dataclass(frozen=True)
class Alert:
    """A weapon-detection alert ready to be dispatched to channels."""

    timestamp: datetime
    source: str
    detections: Sequence[Detection]
    evidence_path: Path | None = None

    @property
    def max_confidence(self) -> float:
        return max((d.confidence for d in self.detections), default=0.0)

    @property
    def labels(self) -> list[str]:
        # Preserve order while de-duplicating.
        seen: dict[str, None] = {}
        for d in self.detections:
            seen.setdefault(d.label, None)
        return list(seen)

    def summary(self) -> str:
        labels = ", ".join(self.labels) or "weapon"
        return (
            f"Weapon detected ({labels}) on source '{self.source}' "
            f"at {self.timestamp.isoformat()} "
            f"with confidence {self.max_confidence:.2f}."
        )


class AlertChannel(abc.ABC):
    """A single delivery mechanism for alerts (SMS, email, sound, ...)."""

    name: str = "channel"

    @property
    def enabled(self) -> bool:
        return True

    @abc.abstractmethod
    def send(self, alert: Alert) -> None:
        """Deliver ``alert``. Implementations should raise on failure."""
