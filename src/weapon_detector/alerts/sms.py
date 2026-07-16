"""SMS alert channel backed by the Twilio API."""

from __future__ import annotations

from ..config import TwilioConfig
from .base import Alert, AlertChannel


class TwilioSMSChannel(AlertChannel):
    """Send an SMS to security personnel via Twilio."""

    name = "sms"

    def __init__(self, config: TwilioConfig, client: object | None = None) -> None:
        self.config = config
        self._client = client

    @property
    def enabled(self) -> bool:
        cfg = self.config
        return bool(
            cfg.enabled
            and cfg.account_sid
            and cfg.auth_token
            and cfg.from_number
            and cfg.to_number
        )

    def _get_client(self) -> object:
        if self._client is None:
            from twilio.rest import Client

            self._client = Client(self.config.account_sid, self.config.auth_token)
        return self._client

    def send(self, alert: Alert) -> None:
        client = self._get_client()
        client.messages.create(  # type: ignore[attr-defined]
            body=alert.summary(),
            from_=self.config.from_number,
            to=self.config.to_number,
        )
