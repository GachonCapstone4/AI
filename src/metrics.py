from __future__ import annotations

from typing import Any

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except ImportError:  # pragma: no cover - keeps local imports working before dependencies are installed.
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = Gauge = Histogram = None
    generate_latest = None


_CLASSIFY_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)
_CONFIDENCE_BUCKETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

if Counter is not None:
    ai_classify_requests_total = Counter(
        "ai_classify_requests_total",
        "Total email classification requests.",
        ("model_version", "domain", "intent", "status"),
    )
    ai_classify_latency_seconds = Histogram(
        "ai_classify_latency_seconds",
        "Email classification latency in seconds.",
        ("model_version", "domain", "intent"),
        buckets=_CLASSIFY_LATENCY_BUCKETS,
    )
    ai_classify_confidence_score = Histogram(
        "ai_classify_confidence_score",
        "Email classification confidence score.",
        ("model_version", "domain", "intent"),
        buckets=_CONFIDENCE_BUCKETS,
    )
    ai_schedule_detected_total = Counter(
        "ai_schedule_detected_total",
        "Total classifications where schedule information was detected.",
        ("model_version", "domain", "intent"),
    )
    ai_classify_errors_total = Counter(
        "ai_classify_errors_total",
        "Total email classification errors.",
        ("model_version", "error_type"),
    )
    ai_active_model_info = Gauge(
        "ai_active_model_info",
        "Active model marker. The active model version has value 1.",
        ("model_version",),
    )
else:
    ai_classify_requests_total = None
    ai_classify_latency_seconds = None
    ai_classify_confidence_score = None
    ai_schedule_detected_total = None
    ai_classify_errors_total = None
    ai_active_model_info = None


_active_model_versions: set[str] = set()


def _label(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text or "unknown"


def record_classify_success(
    *,
    model_version: str | None,
    domain: str | None,
    intent: str | None,
    latency_seconds: float,
    confidence_score: float | None,
    schedule_detected: bool,
) -> None:
    try:
        if ai_classify_requests_total is None:
            return

        labels = {
            "model_version": _label(model_version),
            "domain": _label(domain),
            "intent": _label(intent),
        }
        ai_classify_requests_total.labels(**labels, status="success").inc()
        ai_classify_latency_seconds.labels(**labels).observe(float(latency_seconds))

        if confidence_score is not None:
            ai_classify_confidence_score.labels(**labels).observe(float(confidence_score))

        if schedule_detected:
            ai_schedule_detected_total.labels(**labels).inc()
    except Exception:
        pass


def record_classify_error(*, model_version: str | None, error_type: str | None) -> None:
    try:
        if ai_classify_errors_total is None:
            return
        ai_classify_errors_total.labels(
            model_version=_label(model_version),
            error_type=_label(error_type),
        ).inc()
    except Exception:
        pass


def record_active_model(*, model_version: str | None) -> None:
    try:
        if ai_active_model_info is None:
            return

        version = _label(model_version)
        _active_model_versions.add(version)
        for known_version in _active_model_versions:
            ai_active_model_info.labels(model_version=known_version).set(
                1 if known_version == version else 0
            )
    except Exception:
        pass


def metrics_content() -> tuple[bytes, str]:
    if generate_latest is None:
        return b"# prometheus_client is not installed\n", CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
