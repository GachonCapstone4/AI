# ============================================================
# training consumer
#
# Consume : q.2ai.training
# Publish : q.2app.training
# ============================================================

import sys
import os
import json
import time
from datetime import datetime, timezone

import pika
from pydantic import ValidationError
from sklearn.metrics import accuracy_score, f1_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from api.schemas import TrainingJobRequest, TrainingJobResult, TrainingMetrics
from messaging.publisher import publish
from messaging.structured_log import get_logger
from data_utils import load_dataset, generate_contrastive_pairs, save_pairs_csv
from train_sbert import run_sbert_finetuning, generate_embeddings
from train_domain import train_domain_classifier
from train_intent import train_intent_classifiers
from src.settings import get_settings

# ── 설정 ─────────────────────────────────────────────────────
CONSUME_QUEUE = "q.2ai.training"
PUBLISH_ROUTING_KEY = "q.2app.training"
PREFETCH_COUNT = 1

log = get_logger("consumer.training")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _build_model_version() -> str:
    return "v" + datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")


def _run_training_pipeline() -> dict:
    safe_mode = get_settings().TRAINING_SAFE_MODE

    if safe_mode:
        return {
            "model_version": _build_model_version(),
            "metrics": TrainingMetrics(intent_f1=0.0, domain_accuracy=0.0),
        }

    df = load_dataset()
    pairs = generate_contrastive_pairs(df)
    save_pairs_csv(pairs)

    run_sbert_finetuning()
    X = generate_embeddings(df["email_text"].tolist())

    domain_clf, le_domain = train_domain_classifier(X, df["domain"].values)
    intent_classifiers, intent_encoders = train_intent_classifiers(X, df)

    domain_pred_enc = domain_clf.predict(X)
    domain_pred = le_domain.inverse_transform(domain_pred_enc)
    domain_accuracy = accuracy_score(df["domain"].values, domain_pred)

    intent_true = []
    intent_pred = []
    for domain in df["domain"].unique():
        if domain not in intent_classifiers:
            continue
        mask = df["domain"] == domain
        if not mask.any():
            continue
        clf = intent_classifiers[domain]
        le = intent_encoders[domain]
        X_domain = X[mask]
        y_true = df.loc[mask, "intent"].values
        y_pred_enc = clf.predict(X_domain)
        y_pred = le.inverse_transform(y_pred_enc)
        intent_true.extend(y_true.tolist())
        intent_pred.extend(y_pred.tolist())

    intent_f1 = (
        f1_score(intent_true, intent_pred, average="weighted")
        if intent_true and intent_pred else 0.0
    )

    return {
        "model_version": _build_model_version(),
        "metrics": TrainingMetrics(
            intent_f1=round(float(intent_f1), 4),
            domain_accuracy=round(float(domain_accuracy), 4),
        ),
    }


def _build_success(job_id: str, model_version: str, metrics: TrainingMetrics) -> TrainingJobResult:
    return TrainingJobResult(
        job_id=job_id,
        status="completed",
        model_version=model_version,
        finished_at=_utc_now(),
        metrics=metrics,
        error_message=None,
    )


def _build_failure(job_id: str, error_message: str) -> TrainingJobResult:
    return TrainingJobResult(
        job_id=job_id,
        status="failed",
        model_version=None,
        finished_at=_utc_now(),
        metrics=TrainingMetrics(),
        error_message=error_message,
    )


def _safe_ack(ch, delivery_tag, job_id: str) -> None:
    try:
        ch.basic_ack(delivery_tag=delivery_tag)
        log.info("ack_sent", queue=CONSUME_QUEUE, job_id=job_id, delivery_tag=delivery_tag)
    except Exception as e:
        log.error(
            "ack_failed",
            queue=CONSUME_QUEUE,
            job_id=job_id,
            delivery_tag=delivery_tag,
            error=str(e),
        )


def _safe_nack(ch, delivery_tag, job_id: str, requeue: bool = False) -> None:
    try:
        ch.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
        log.info(
            "nack_sent",
            queue=CONSUME_QUEUE,
            job_id=job_id,
            delivery_tag=delivery_tag,
            requeue=requeue,
        )
    except Exception as e:
        log.error(
            "nack_failed",
            queue=CONSUME_QUEUE,
            job_id=job_id,
            delivery_tag=delivery_tag,
            error=str(e),
        )


def _safe_publish_result(ch, result: TrainingJobResult, job_id: str) -> bool:
    try:
        print("[training.publish] sending to q.2app.training")
        sys.stdout.flush()

        ch.basic_publish(
            exchange="",
            routing_key=PUBLISH_ROUTING_KEY,
            body=json.dumps(result.model_dump(), ensure_ascii=False).encode("utf-8"),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
            ),
        )
        log.info(
            "result_published",
            queue=CONSUME_QUEUE,
            publish_routing_key=PUBLISH_ROUTING_KEY,
            job_id=job_id,
            status=result.status,
        )
        return True
    except Exception as e:
        log.error(
            "result_publish_failed",
            queue=CONSUME_QUEUE,
            publish_routing_key=PUBLISH_ROUTING_KEY,
            job_id=job_id,
            status=result.status,
            error=str(e),
        )
        return False


def _callback(ch, method, _properties, body):
    job_id = "(unknown)"
    t0 = time.perf_counter()

    try:
        print(f"[training.callback] entered delivery_tag={method.delivery_tag}")
        print(f"[training.callback] raw_body={body!r}")
        sys.stdout.flush()

        log.info(
            "callback_entered",
            queue=CONSUME_QUEUE,
            delivery_tag=method.delivery_tag,
            raw_body_preview=body.decode("utf-8", errors="replace")[:500],
        )

        data = json.loads(body)
        job_id = data.get("job_id", job_id)
        log.info("message_parsed", queue=CONSUME_QUEUE, job_id=job_id)

        log.info("job_received", queue=CONSUME_QUEUE, job_id=job_id)
        payload = TrainingJobRequest(**data)

        if payload.job_type != "training" or payload.task_type != "training":
            raise ValueError("unsupported job_type/task_type")

        log.info(
            "job_started",
            queue=CONSUME_QUEUE,
            job_id=payload.job_id,
            dataset_version=payload.dataset_version,
            requested_by=payload.requested_by,
            safe_mode=get_settings().TRAINING_SAFE_MODE,
        )

        log.info("training_before_run", queue=CONSUME_QUEUE, job_id=payload.job_id)
        result_data = _run_training_pipeline()
        log.info("training_after_run", queue=CONSUME_QUEUE, job_id=payload.job_id)

        result = _build_success(
            job_id=payload.job_id,
            model_version=result_data["model_version"],
            metrics=result_data["metrics"],
        )

        publish_ok = _safe_publish_result(ch, result, payload.job_id)
        if publish_ok:
            _safe_ack(ch, method.delivery_tag, payload.job_id)
        else:
            _safe_nack(ch, method.delivery_tag, payload.job_id, requeue=False)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.info(
            "job_completed",
            queue=CONSUME_QUEUE,
            job_id=payload.job_id,
            status=result.status,
            model_version=result.model_version,
            elapsed_ms=elapsed_ms,
        )

    except json.JSONDecodeError as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.error(
            "job_failed",
            queue=CONSUME_QUEUE,
            job_id=job_id,
            elapsed_ms=elapsed_ms,
            error=str(e),
        )
        result = _build_failure(job_id=job_id, error_message=str(e))
        publish_ok = _safe_publish_result(ch, result, job_id)
        if publish_ok:
            _safe_ack(ch, method.delivery_tag, job_id)
        else:
            _safe_nack(ch, method.delivery_tag, job_id, requeue=False)

    except (ValidationError, ValueError) as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.error(
            "job_failed",
            queue=CONSUME_QUEUE,
            job_id=job_id,
            elapsed_ms=elapsed_ms,
            error=str(e),
        )
        result = _build_failure(job_id=job_id, error_message=str(e))
        publish_ok = _safe_publish_result(ch, result, job_id)
        if publish_ok:
            _safe_ack(ch, method.delivery_tag, job_id)
        else:
            _safe_nack(ch, method.delivery_tag, job_id, requeue=False)

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        print(f"[training.callback] unhandled_error={e}")
        sys.stdout.flush()
        log.error(
            "job_failed",
            queue=CONSUME_QUEUE,
            job_id=job_id,
            elapsed_ms=elapsed_ms,
            error=str(e),
        )
        result = _build_failure(job_id=job_id, error_message=str(e))
        publish_ok = _safe_publish_result(ch, result, job_id)
        if publish_ok:
            _safe_ack(ch, method.delivery_tag, job_id)
        else:
            _safe_nack(ch, method.delivery_tag, job_id, requeue=False)


def main():
    settings = get_settings()

    while True:
        try:
            conn = pika.BlockingConnection(pika.URLParameters(settings.RABBITMQ_URL))
            ch = conn.channel()
            ch.basic_qos(prefetch_count=PREFETCH_COUNT)
            ch.basic_consume(
                queue=CONSUME_QUEUE,
                on_message_callback=_callback,
                auto_ack=False,
            )
            log.info("consuming", queue=CONSUME_QUEUE, safe_mode=settings.TRAINING_SAFE_MODE)
            ch.start_consuming()

        except pika.exceptions.AMQPConnectionError as e:
            log.warning("connection_lost", queue=CONSUME_QUEUE, error=str(e), retry_in_sec=5)
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("shutdown", queue=CONSUME_QUEUE)
            break


if __name__ == "__main__":
    main()
