"""Coordinate multiple alert channels with de-duplication / cooldown."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable

from ..config import AlertConfig
from .base import Alert, AlertChannel
from .desktop import DesktopAlertChannel
from .email_alert import EmailAlertChannel
from .sms import TwilioSMSChannel
from .sound import SoundAlertChannel

logger = logging.getLogger(__name__)


class AlertManager:
    """Dispatch alerts to all enabled channels, honoring a cooldown window.

    A failure in one channel is logged but never prevents the other channels
    from being tried.
    """

    def __init__(
        self,
        channels: Iterable[AlertChannel],
        cooldown_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._channels = [c for c in channels if c.enabled]
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._last_dispatch: float | None = None

    @classmethod
    def from_config(cls, config: AlertConfig) -> AlertManager:
        channels: list[AlertChannel] = [
            SoundAlertChannel(enabled=config.sound_enabled, sound_file=config.sound_file),
            DesktopAlertChannel(enabled=config.desktop_enabled),
            TwilioSMSChannel(config.twilio),
            EmailAlertChannel(config.email),
        ]
        return cls(channels, cooldown_seconds=config.cooldown_seconds)

    @property
    def active_channels(self) -> list[str]:
        return [c.name for c in self._channels]

    def _in_cooldown(self) -> bool:
        if self._last_dispatch is None:
            return False
        return (self._clock() - self._last_dispatch) < self.cooldown_seconds

    def dispatch(self, alert: Alert, force: bool = False) -> bool:
        """Send ``alert`` to all channels.

        Returns ``True`` if the alert was dispatched, ``False`` if it was
        suppressed by the cooldown window. Set ``force=True`` to bypass cooldown.
        """
        if not force and self._in_cooldown():
            logger.debug("Alert suppressed by cooldown window")
            return False
        self._last_dispatch = self._clock()
        for channel in self._channels:
            try:
                channel.send(alert)
            except Exception:  # noqa: BLE001 - one channel must not break others
                logger.exception("Alert channel %r failed", channel.name)
        return True
