from __future__ import annotations

from datetime import datetime, timezone

from weapon_detector.alerts import Alert, AlertChannel, AlertManager
from weapon_detector.alerts.email_alert import EmailAlertChannel
from weapon_detector.alerts.sms import TwilioSMSChannel
from weapon_detector.config import EmailConfig, TwilioConfig
from weapon_detector.detection import Detection


def _alert():
    return Alert(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source="0",
        detections=[Detection(label="pistol", confidence=0.9, box=(0, 0, 1, 1))],
    )


class RecordingChannel(AlertChannel):
    name = "recording"

    def __init__(self, enabled=True, fail=False):
        self._enabled = enabled
        self.fail = fail
        self.sent = []

    @property
    def enabled(self):
        return self._enabled

    def send(self, alert):
        if self.fail:
            raise RuntimeError("boom")
        self.sent.append(alert)


def test_manager_dispatches_to_enabled_channels():
    good = RecordingChannel()
    disabled = RecordingChannel(enabled=False)
    manager = AlertManager([good, disabled], cooldown_seconds=0)
    assert manager.active_channels == ["recording"]
    assert manager.dispatch(_alert()) is True
    assert len(good.sent) == 1
    assert disabled.sent == []


def test_manager_cooldown_suppresses():
    clock = {"t": 100.0}
    ch = RecordingChannel()
    manager = AlertManager([ch], cooldown_seconds=30, clock=lambda: clock["t"])
    assert manager.dispatch(_alert()) is True
    clock["t"] = 110.0
    assert manager.dispatch(_alert()) is False  # within cooldown
    clock["t"] = 200.0
    assert manager.dispatch(_alert()) is True
    assert len(ch.sent) == 2


def test_manager_force_bypasses_cooldown():
    ch = RecordingChannel()
    manager = AlertManager([ch], cooldown_seconds=1000)
    manager.dispatch(_alert())
    assert manager.dispatch(_alert(), force=True) is True
    assert len(ch.sent) == 2


def test_channel_failure_does_not_break_others():
    failing = RecordingChannel(fail=True)
    ok = RecordingChannel()
    manager = AlertManager([failing, ok], cooldown_seconds=0)
    assert manager.dispatch(_alert()) is True
    assert len(ok.sent) == 1


def test_twilio_channel_disabled_without_credentials():
    channel = TwilioSMSChannel(TwilioConfig(enabled=True))
    assert channel.enabled is False


def test_twilio_channel_sends_with_client():
    class FakeMessages:
        def __init__(self):
            self.created = []

        def create(self, **kwargs):
            self.created.append(kwargs)

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

    client = FakeClient()
    cfg = TwilioConfig(
        enabled=True,
        account_sid="AC",
        auth_token="tok",
        from_number="+1",
        to_number="+2",
    )
    channel = TwilioSMSChannel(cfg, client=client)
    assert channel.enabled is True
    channel.send(_alert())
    assert client.messages.created[0]["to"] == "+2"
    assert "Weapon detected" in client.messages.created[0]["body"]


def test_email_channel_builds_message_and_sends():
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port):
            sent["host"] = host

        def starttls(self):
            sent["tls"] = True

        def login(self, user, password):
            sent["user"] = user

        def send_message(self, message):
            sent["message"] = message

        def quit(self):
            sent["quit"] = True

    cfg = EmailConfig(
        enabled=True,
        smtp_host="smtp.test",
        username="u",
        password="p",
        from_address="from@test",
        to_addresses=["to@test"],
    )
    channel = EmailAlertChannel(cfg, smtp_factory=lambda h, p: FakeSMTP(h, p))
    assert channel.enabled is True
    channel.send(_alert())
    assert sent["host"] == "smtp.test"
    assert sent["tls"] is True
    assert sent["message"]["To"] == "to@test"
    assert sent["quit"] is True


def test_alert_summary_lists_labels():
    alert = Alert(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source="cam1",
        detections=[
            Detection(label="pistol", confidence=0.9, box=(0, 0, 1, 1)),
            Detection(label="knife", confidence=0.7, box=(0, 0, 1, 1)),
        ],
    )
    summary = alert.summary()
    assert "pistol" in summary and "knife" in summary
    assert alert.max_confidence == 0.9
