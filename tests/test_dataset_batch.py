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


def _dataset_rows(start: int, count: int, domain_prefix: str = "Domain"):
    rows = []
    for offset in range(count):
        idx = start + offset
        rows.append(
            {
                "emailId": f"train_{idx}",
                "threadId": f"thread-{idx}",
                "from": f"sender{idx}@example.com",
                "subject": f"Subject {idx}",
                "body": f"Body {idx}",
                "domain": f"{domain_prefix}{idx % 2}",
                "intent": f"Intent{idx % 3}",
            }
        )
    return rows


def test_merge_dataset_rows_keeps_existing_dataset_new_and_appends_new_rows_near_1050():
    _stub_missing_batch_dependencies()
    from batch.dataset_batch import merge_dataset_rows

    existing_rows = _dataset_rows(0, 1000)
    new_rows = _dataset_rows(1000, 50)

    result = merge_dataset_rows(existing_rows, new_rows)

    assert result["existing_rows"] == 1000
    assert result["new_rows"] == 50
    assert len(result["merged_rows"]) == 1050


def test_merge_dataset_rows_deduplicates_by_email_id_with_latest_data_winning():
    _stub_missing_batch_dependencies()
    from batch.dataset_batch import merge_dataset_rows

    existing_rows = _dataset_rows(1, 1)
    new_rows = [
        {
            **_dataset_rows(1, 1, domain_prefix="UpdatedDomain")[0],
            "subject": "Updated subject",
            "domain": "UpdatedDomain",
            "intent": "UpdatedIntent",
        }
    ]

    result = merge_dataset_rows(existing_rows, new_rows)

    assert len(result["merged_rows"]) == 1
    assert result["merged_rows"][0]["subject"] == "Updated subject"
    assert result["merged_rows"][0]["domain"] == "UpdatedDomain"
    assert result["merged_rows"][0]["intent"] == "UpdatedIntent"


def test_write_csv_rows_preserves_commas_newlines_and_quotes(tmp_path):
    _stub_missing_batch_dependencies()
    from batch.dataset_batch import _read_csv_rows, _write_csv_rows

    csv_path = tmp_path / "dataset_new.csv"
    rows = [
        {
            "emailId": "train_csv",
            "threadId": "thread,csv",
            "from": "sender@example.com",
            "subject": 'Subject, with "quote"',
            "body": 'Line 1\nLine, 2\n"Line 3"',
            "domain": "Finance",
            "intent": "Invoice",
        }
    ]

    _write_csv_rows(rows, str(csv_path))
    loaded_rows = _read_csv_rows(str(csv_path))

    assert loaded_rows[0]["subject"] == 'Subject, with "quote"'
    assert loaded_rows[0]["body"] == 'Line 1\nLine, 2\n"Line 3"'
    assert loaded_rows[0]["email_text"] == 'Subject, with "quote"\nLine 1\nLine, 2\n"Line 3"'


def test_validate_dataset_logs_distribution_and_rejects_too_small(caplog):
    _stub_missing_batch_dependencies()
    from batch.dataset_batch import validate_dataset

    caplog.set_level("INFO")
    rows = _dataset_rows(0, 4)

    with pytest.raises(ValueError, match="Dataset is too small"):
        validate_dataset(rows, min_samples=5)

    assert "domain distribution" in caplog.text
    assert "domain missing count" in caplog.text
    assert "intent missing count" in caplog.text
    assert "Domain0" in caplog.text


def test_dataset_merge_uploads_only_dataset_new_key(monkeypatch, tmp_path):
    _stub_missing_batch_dependencies()
    import batch.dataset_batch as dataset_batch

    uploaded_keys = []

    class FakeS3Client:
        def download_file(self, _bucket, key, destination):
            if key == "dataset/dataset_new.csv":
                dataset_batch._write_csv_rows(_dataset_rows(0, 1000), destination)
                return
            raise FileNotFoundError(key)

        def upload_file(self, _filepath, _bucket, key):
            uploaded_keys.append(key)

    monkeypatch.setattr(dataset_batch.boto3, "client", lambda *args, **kwargs: FakeS3Client(), raising=False)

    existing_path = tmp_path / "dataset_new_existing.csv"
    merged_path = tmp_path / "dataset_new.csv"

    dataset_batch.download_from_s3_if_exists("dataset/dataset_new.csv", str(existing_path))
    result = dataset_batch.merge_dataset_rows(
        dataset_batch._read_csv_rows(str(existing_path)),
        _dataset_rows(1000, 50),
    )
    dataset_batch._write_csv_rows(result["merged_rows"], str(merged_path))
    dataset_batch.upload_to_s3(str(merged_path), "dataset/dataset_new.csv")

    assert uploaded_keys == ["dataset/dataset_new.csv"]


def test_validation_failure_does_not_upload_dataset(monkeypatch):
    _stub_missing_batch_dependencies()
    import batch.dataset_batch as dataset_batch

    uploaded = []

    class FakeConnection:
        def close(self):
            pass

    monkeypatch.setattr(dataset_batch, "validate_required_env", lambda names=dataset_batch.REQUIRED_ENV_VARS: None)
    monkeypatch.setattr(dataset_batch, "connect_rabbitmq", lambda: (FakeConnection(), _FakeChannel()))
    monkeypatch.setattr(dataset_batch, "publish_sse_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(dataset_batch, "publish_training_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(dataset_batch, "fetch_training_data", lambda: _dataset_rows(0, 1, domain_prefix="Only"))
    monkeypatch.setattr(dataset_batch, "download_from_s3_if_exists", lambda _key, _path: False)
    monkeypatch.setattr(dataset_batch, "upload_to_s3", lambda *args, **kwargs: uploaded.append(args))

    with pytest.raises(ValueError, match="Dataset is too small"):
        dataset_batch.main()

    assert uploaded == []


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
        dataset_s3_uri="s3://capstone-gachon/dataset/dataset_new.csv",
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
    assert payload["dataset_s3_uri"] == "s3://capstone-gachon/dataset/dataset_new.csv"


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
