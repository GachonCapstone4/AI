import csv
import importlib.util
import json
import sys
import types

import pytest


class _BasicProperties:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _stub_missing_batch_dependencies():
    if "boto3" not in sys.modules and importlib.util.find_spec("boto3") is None:
        sys.modules["boto3"] = types.ModuleType("boto3")

    if "mysql" not in sys.modules and importlib.util.find_spec("mysql") is None:
        mysql = types.ModuleType("mysql")
        mysql.connector = types.ModuleType("mysql.connector")
        sys.modules["mysql"] = mysql
        sys.modules["mysql.connector"] = mysql.connector

    if "pika" not in sys.modules and importlib.util.find_spec("pika") is None:
        pika = types.ModuleType("pika")
        pika.PlainCredentials = object
        pika.ConnectionParameters = object
        pika.BlockingConnection = object
        pika.BasicProperties = _BasicProperties
        sys.modules["pika"] = pika


class _FakeChannel:
    def __init__(self):
        self.published = None
        self.exchange_declarations = []
        self.queue_declarations = []
        self.bindings = []

    def exchange_declare(self, **kwargs):
        self.exchange_declarations.append(kwargs)

    def queue_declare(self, **kwargs):
        self.queue_declarations.append(kwargs)

    def queue_bind(self, **kwargs):
        self.bindings.append(kwargs)

    def basic_publish(self, **kwargs):
        self.published = kwargs


def test_create_csv_adds_regenerated_email_text(tmp_path):
    _stub_missing_batch_dependencies()
    from batch.dataset_batch import create_csv

    csv_path = tmp_path / "dataset.csv"
    rows = [
        {
            "emailId": "train_1",
            "threadId": "thread-1",
            "from": "sender@example.com",
            "subject": "Invoice request",
            "body": "Please send the invoice.",
            "email_text": "stale value",
            "domain": "Finance",
            "intent": "Invoice Request",
        },
        {
            "emailId": "train_2",
            "threadId": None,
            "from": "empty@example.com",
            "subject": None,
            "body": "",
            "domain": "General",
            "intent": "Question",
        },
    ]

    create_csv(rows, str(csv_path))

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        written_rows = list(reader)

    assert reader.fieldnames == [
        "emailId",
        "threadId",
        "from",
        "subject",
        "body",
        "email_text",
        "domain",
        "intent",
    ]
    assert written_rows[0]["email_text"] == "Invoice request\nPlease send the invoice."
    assert written_rows[1]["email_text"] == ""


def test_dataset_sse_log_uses_collecting_sse_type(monkeypatch):
    _stub_missing_batch_dependencies()
    import batch.dataset_batch as dataset_batch

    monkeypatch.setattr(dataset_batch, "ADMIN_USER_ID", "1")
    channel = _FakeChannel()

    dataset_batch.publish_sse_log(
        channel,
        "[INFO] DB 데이터 추출 시작",
        sse_type=dataset_batch.DATASET_SSE_TYPE,
    )

    payload = json.loads(channel.published["body"])
    assert channel.published["exchange"] == "x.sse.fanout"
    assert channel.published["routing_key"] == ""
    assert payload == {
        "user_id": 1,
        "sse_type": "ai-collecting-updated",
        "data": "[INFO] DB 데이터 추출 시작",
    }
    assert isinstance(payload["user_id"], int)


def test_dataset_training_event_uses_admin_status_route(monkeypatch):
    _stub_missing_batch_dependencies()
    import batch.dataset_batch as dataset_batch

    monkeypatch.setattr(dataset_batch, "JOB_ID", "collect-1")
    channel = _FakeChannel()

    dataset_batch.publish_training_event(
        channel,
        status="COMPLETED",
        dataset_version="v2026-05-12-120000",
    )

    payload = json.loads(channel.published["body"])
    assert channel.exchange_declarations == [
        {
            "exchange": "x.ai2app.direct",
            "exchange_type": "direct",
            "durable": True,
        }
    ]
    assert channel.queue_declarations == [
        {"queue": "q.2app.training", "durable": True}
    ]
    assert channel.bindings == [
        {
            "queue": "q.2app.training",
            "exchange": "x.ai2app.direct",
            "routing_key": "app.training",
        }
    ]
    assert channel.published["exchange"] == "x.ai2app.direct"
    assert channel.published["routing_key"] == "app.training"
    assert payload["job_id"] == "collect-1"
    assert payload["status"] == "COMPLETED"
    assert payload["dataset_version"] == "v2026-05-12-120000"


def test_dataset_sse_log_rejects_non_integer_user_id(monkeypatch):
    _stub_missing_batch_dependencies()
    import batch.dataset_batch as dataset_batch

    monkeypatch.setattr(dataset_batch, "ADMIN_USER_ID", "admin")
    channel = _FakeChannel()

    with pytest.raises(ValueError, match="ADMIN_USER_ID must be an integer"):
        dataset_batch.publish_sse_log(
            channel,
            "[INFO] DB 데이터 추출 시작",
            sse_type=dataset_batch.DATASET_SSE_TYPE,
        )

    assert channel.published is None
