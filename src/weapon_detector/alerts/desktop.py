"""Desktop pop-up notification alert channel."""

from __future__ import annotations

from .base import Alert, AlertChannel


class DesktopAlertChannel(AlertChannel):
    """Show an OS desktop notification via :mod:`plyer` when available."""

    name = "desktop"

    def __init__(self, enabled: bool = True, timeout: int = 10) -> None:
        self._enabled = enabled
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send(self, alert: Alert) -> None:
        from plyer import notification

        notification.notify(
            title="Weapon Detected",
            message=alert.summary(),
            timeout=self.timeout,
        )
