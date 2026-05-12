import pytest

from src.mlops.training_events import publish_sse_log, publish_training_status


def _clear_training_env(monkeypatch):
    for name in (
        "RABBITMQ_HOST",
        "RABBITMQ_PORT",
        "RABBITMQ_USERNAME",
        "RABBITMQ_PASSWORD",
        "TRAINING_STATUS_QUEUE",
        "TRAINING_STATUS_ROUTING_KEY",
        "AI2APP_EXCHANGE",
        "SSE_EXCHANGE",
        "ADMIN_USER_ID",
        "USER_ID",
        "RABBITMQ_DRY_RUN",
    ):
        monkeypatch.delenv(name, raising=False)


def test_training_status_running_is_summary_event_only(monkeypatch):
    _clear_training_env(monkeypatch)

    result = publish_training_status(
        job_id="train-1",
        status="RUNNING",
        model_version="training-final-004",
        dry_run=True,
    )

    assert result["queue"] == "q.2app.training"
    assert result["exchange"] == "x.ai2app.direct"
    assert result["routing_key"] == "app.training"
    assert result["payload"] == {
        "job_id": "train-1",
        "status": "RUNNING",
        "model_version": "training-final-004",
        "finished_at": None,
        "metrics": {
            "intent_f1": None,
            "domain_accuracy": None,
        },
        "error_message": None,
    }
    assert "data" not in result["payload"]
    assert "message" not in result["payload"]
    assert "stdout" not in result["payload"]
    assert "stderr" not in result["payload"]


def test_training_status_completed_filters_metrics(monkeypatch):
    _clear_training_env(monkeypatch)

    result = publish_training_status(
        job_id="train-1",
        status="COMPLETED",
        model_version="training-final-004",
        metrics={
            "intent_f1": 0.91,
            "domain_accuracy": 0.88,
            "warning": "long text",
        },
        finished_at="2026-05-04T10:30:05Z",
        dry_run=True,
    )

    assert result["payload"] == {
        "job_id": "train-1",
        "status": "COMPLETED",
        "model_version": "training-final-004",
        "finished_at": "2026-05-04T10:30:05Z",
        "metrics": {
            "intent_f1": 0.91,
            "domain_accuracy": 0.88,
        },
        "error_message": None,
    }


def test_training_status_failed_has_finished_at_and_error(monkeypatch):
    _clear_training_env(monkeypatch)

    result = publish_training_status(
        job_id="train-1",
        status="FAILED",
        model_version="training-final-004",
        error_message="training failed",
        dry_run=True,
    )

    assert result["payload"]["status"] == "FAILED"
    assert result["payload"]["model_version"] == "training-final-004"
    assert result["payload"]["finished_at"]
    assert result["payload"]["error_message"] == "training failed"


def test_sse_log_uses_fanout_exchange_and_log_payload(monkeypatch):
    _clear_training_env(monkeypatch)
    monkeypatch.setenv("ADMIN_USER_ID", "1")

    result = publish_sse_log("[INFO] SBERT 학습 시작", dry_run=True)

    assert result["exchange"] == "x.sse.fanout"
    assert result["payload"] == {
        "user_id": 1,
        "sse_type": "ai-training-updated",
        "data": "[INFO] SBERT 학습 시작",
    }
    assert isinstance(result["payload"]["user_id"], int)


def test_sse_log_converts_user_id_env_string_to_int(monkeypatch):
    _clear_training_env(monkeypatch)
    monkeypatch.setenv("USER_ID", "1")

    result = publish_sse_log("[INFO] Domain Logistic Regression 학습 시작", dry_run=True)

    assert result["payload"]["user_id"] == 1
    assert isinstance(result["payload"]["user_id"], int)
    assert result["payload"]["sse_type"] == "ai-training-updated"


def test_sse_log_rejects_non_integer_user_id(monkeypatch):
    _clear_training_env(monkeypatch)
    monkeypatch.setenv("ADMIN_USER_ID", "admin")

    with pytest.raises(ValueError, match="ADMIN_USER_ID must be an integer"):
        publish_sse_log("[INFO] SBERT 학습 시작", dry_run=True)
