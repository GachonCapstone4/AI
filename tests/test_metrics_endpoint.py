import sys
import types

sys.modules.setdefault("joblib", types.SimpleNamespace(load=lambda *_args, **_kwargs: None))
sys.modules.setdefault(
    "sentence_transformers",
    types.SimpleNamespace(SentenceTransformer=object),
)

from fastapi.testclient import TestClient

from api.main import app
from src.metrics import record_active_model, record_classify_error, record_classify_success


def test_metrics_endpoint_exposes_inference_metrics_only():
    record_classify_success(
        model_version="training-final-004",
        domain="업무",
        intent="문의",
        latency_seconds=0.123,
        confidence_score=0.91,
        schedule_detected=True,
    )
    record_classify_error(
        model_version="training-final-004",
        error_type="ValueError",
    )
    record_active_model(model_version="training-final-004")

    response = TestClient(app).get("/metrics")
    body = response.text

    assert response.status_code == 200
    assert "ai_classify_requests_total" in body
    assert "ai_classify_latency_seconds_bucket" in body
    assert "ai_classify_confidence_score_bucket" in body
    assert "ai_schedule_detected_total" in body
    assert "ai_classify_errors_total" in body
    assert "ai_active_model_info" in body
    assert "ai_training_intent_f1" not in body
    assert "ai_training_domain_accuracy" not in body

    assert "email_id=" not in body
    assert "outbox_id=" not in body
    assert "request_id=" not in body
