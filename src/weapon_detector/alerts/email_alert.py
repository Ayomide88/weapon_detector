"""Email alert channel backed by SMTP."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Callable

from ..config import EmailConfig
from .base import Alert, AlertChannel

SMTPFactory = Callable[[str, int], smtplib.SMTP]


class EmailAlertChannel(AlertChannel):
    """Send an email (optionally with the evidence image attached) via SMTP."""

    name = "email"

    def __init__(
        self, config: EmailConfig, smtp_factory: SMTPFactory | None = None
    ) -> None:
        self.config = config
        self._smtp_factory = smtp_factory or (lambda host, port: smtplib.SMTP(host, port))

    @property
    def enabled(self) -> bool:
        cfg = self.config
        return bool(
            cfg.enabled and cfg.smtp_host and cfg.from_address and cfg.to_addresses
        )

    def _build_message(self, alert: Alert) -> EmailMessage:
        message = EmailMessage()
        message["Subject"] = "ALERT: Weapon Detected"
        message["From"] = self.config.from_address
        message["To"] = ", ".join(self.config.to_addresses)
        message.set_content(alert.summary())
        if alert.evidence_path is not None:
            path = Path(alert.evidence_path)
            if path.exists():
                message.add_attachment(
                    path.read_bytes(),
                    maintype="image",
                    subtype="jpeg",
                    filename=path.name,
                )
        return message

    def send(self, alert: Alert) -> None:
        message = self._build_message(alert)
        smtp = self._smtp_factory(self.config.smtp_host, self.config.smtp_port)
        try:
            if self.config.use_tls:
                smtp.starttls()
            if self.config.username:
                smtp.login(self.config.username, self.config.password)
            smtp.send_message(message)
        finally:
            smtp.quit()
