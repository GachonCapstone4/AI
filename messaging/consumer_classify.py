# ============================================================
# classify consumer
#
# Consume : q.2ai.classify   (x.app2ai.direct)
# Publish : q.2app.classify  (x.ai2app.direct)
#
# ack / nack 정책
# ---------------
# 성공                → ack
# JSON/Pydantic 실패  → q.dlx.failed publish 후 ack
# 처리 실패           → x-death count < 3 이면 nack(requeue=False) 로 retry queue 이동
# 처리 실패           → x-death count >= 3 또는 영구 실패면 q.dlx.failed publish 후 ack
# ============================================================

import sys
import os
import json
import time
import threading

import pika
from pydantic import ValidationError

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from api.schemas import ClassifyRequest, ResponseMeta
from api.services.classify_service import run_classify
from messaging.publisher import AI2APP_EXCHANGE, enable_delivery_confirms, publish
from messaging.structured_log import get_logger
from inference import load_classify_pipeline, predict_email
from src.settings import get_settings

# ── 설정 ─────────────────────────────────────────────────────
CONSUME_QUEUE        = "q.2ai.classify"
PUBLISH_QUEUE        = "q.2app.classify"
PUBLISH_ROUTING_KEY  = "2app.classify"
FAILED_QUEUE         = "q.dlx.failed"
PREFETCH_COUNT       = 1
MAX_RETRY_COUNT      = 3

log = get_logger("consumer.classify")

_classify_pipeline: dict = {}

PERMANENT_ERROR_MARKERS = (
    "400",
    "401",
    "403",
    "invalid api key",
    "authentication",
    "unauthorized",
    "forbidden",
    "bad request",
)
TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "temporarily unavailable",
    "service unavailable",
    "server disconnected",
    "remote end closed connection",
    "network is unreachable",
)


def _build_backend_classify_payload(result) -> dict:
    schedule_info = result.schedule_info
    schedule_detected = schedule_info is not None
    entities_json = (
        json.dumps(schedule_info, ensure_ascii=False)
        if schedule_detected
        else "{}"
    )
    model_version = getattr(result.meta, "model_version", None) or "unknown"

    return {
        "outbox_id": result.outbox_id,
        "email_id": result.email_id,
        "domain": result.classification.domain,
        "intent": result.classification.intent,
        "confidence_score": result.confidence_score,
        "summary_text": result.summary,
        "schedule_detected": schedule_detected,
        "entities_json": entities_json,
        "model_version": model_version,
    }


def _resolve_model_version(pipeline: dict) -> str:
    settings_version = get_settings().ACTIVE_MODEL_VERSION
    if settings_version:
        return settings_version

    model_metadata = (pipeline.get("model") or {}).get("metadata") or {}
    return (
        model_metadata.get("model_version")
        or model_metadata.get("modelVersion")
        or "unknown"
    )


class ClassifyConsumerRunner:
    def __init__(self, pipeline: dict) -> None:
        self._pipeline = pipeline
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._connection = None
        self._channel = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._run,
            name="rabbitmq-classify-consumer",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()

        if self._channel and getattr(self._channel, "is_open", False):
            try:
                self._connection.add_callback_threadsafe(self._channel.stop_consuming)
            except Exception:
                pass

        if self._connection and getattr(self._connection, "is_open", False):
            try:
                self._connection.add_callback_threadsafe(self._connection.close)
            except Exception:
                pass

        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        global _classify_pipeline
        _classify_pipeline = self._pipeline

        settings = get_settings()
        log.info("consumer_starting", queue=CONSUME_QUEUE)

        while not self._stop_event.is_set():
            try:
                conn = pika.BlockingConnection(pika.URLParameters(settings.RABBITMQ_URL))
                ch = conn.channel()

                self._connection = conn
                self._channel = ch

                enable_delivery_confirms(ch)
                ch.basic_qos(prefetch_count=PREFETCH_COUNT)
                ch.basic_consume(queue=CONSUME_QUEUE, on_message_callback=_callback)
                log.info(
                    "consuming",
                    queue=CONSUME_QUEUE,
                    source_exchange="x.app2ai.direct",
                    source_routing_key="2ai.classify",
                )
                ch.start_consuming()
            except pika.exceptions.AMQPConnectionError as e:
                if self._stop_event.is_set():
                    break
                log.warning(
                    "connection_lost",
                    queue=CONSUME_QUEUE,
                    error=str(e),
                    retry_in_sec=5,
                )
                time.sleep(5)
            except Exception as e:
                if self._stop_event.is_set():
                    break
                log.error(
                    "consumer_crashed",
                    queue=CONSUME_QUEUE,
                    exception_type=type(e).__name__,
                    error=str(e),
                    retry_in_sec=5,
                )
                time.sleep(5)
            finally:
                self._channel = None
                if self._connection and getattr(self._connection, "is_open", False):
                    try:
                        self._connection.close()
                    except Exception:
                        pass
                self._connection = None

        log.info("consumer_stopped", queue=CONSUME_QUEUE)


def _safe_ack(ch, delivery_tag, outbox_id, email_id) -> None:
    ch.basic_ack(delivery_tag=delivery_tag)
    log.info("ack_sent", queue=CONSUME_QUEUE, outbox_id=outbox_id, email_id=email_id)


def _safe_nack(ch, delivery_tag, outbox_id, email_id, requeue: bool) -> None:
    ch.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
    log.info(
        "nack_sent",
        queue=CONSUME_QUEUE,
        outbox_id=outbox_id,
        email_id=email_id,
        requeue=requeue,
    )


def _get_retry_count(properties) -> int:
    headers = getattr(properties, "headers", None) or {}
    x_death = headers.get("x-death", [])

    if not isinstance(x_death, list):
        return 0

    for death in x_death:
        if isinstance(death, dict) and death.get("queue") == CONSUME_QUEUE:
            count = death.get("count", 0)
            try:
                return int(count)
            except (TypeError, ValueError):
                return 0
    return 0


def _publish_failed_message(
    ch,
    *,
    outbox_id: str,
    email_id: str,
    body: bytes,
    error: str,
    retry_count: int,
    exception_type: str,
) -> None:
    failed_message = {
        "source_queue": CONSUME_QUEUE,
        "outbox_id": outbox_id,
        "email_id": email_id,
        "retry_count": retry_count,
        "exception_type": exception_type,
        "error": error,
        "body": body.decode("utf-8", errors="replace"),
    }

    ch.basic_publish(
        exchange="",
        routing_key=FAILED_QUEUE,
        body=json.dumps(failed_message, ensure_ascii=False).encode("utf-8"),
        properties=pika.BasicProperties(
            content_type="application/json",
            delivery_mode=2,
        ),
    )
    log.error(
        "failed_message_published",
        queue=CONSUME_QUEUE,
        failed_queue=FAILED_QUEUE,
        outbox_id=outbox_id,
        email_id=email_id,
        retry_count=retry_count,
        exception_type=exception_type,
        error=error,
    )


def _is_permanent_processing_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in PERMANENT_ERROR_MARKERS)


def _is_transient_processing_error(exc: Exception) -> bool:
    if isinstance(exc, (pika.exceptions.AMQPConnectionError, TimeoutError)):
        return True

    message = str(exc).lower()
    return any(marker in message for marker in TRANSIENT_ERROR_MARKERS)


# ── 콜백 ─────────────────────────────────────────────────────
def _callback(ch, method, properties, body):
    outbox_id = "(unknown)"
    email_id  = "(unknown)"
    t0 = time.perf_counter()
    retry_count = _get_retry_count(properties)

    try:
        log.info(
            "received_message",
            queue=CONSUME_QUEUE,
            delivery_tag=method.delivery_tag,
            routing_key=method.routing_key,
            exchange=method.exchange,
            redelivered=method.redelivered,
            retry_count=retry_count,
            content_type=getattr(properties, "content_type", None),
            body_size=len(body),
        )

        data      = json.loads(body)
        outbox_id = data.get("outbox_id", outbox_id)
        email_id  = data.get("email_id", data.get("emailId", email_id))

        log.info("message_parsed",
                 queue=CONSUME_QUEUE, outbox_id=outbox_id, email_id=email_id)

        payload = ClassifyRequest(**data)
        log.info("processing_started",
                 queue=CONSUME_QUEUE, outbox_id=payload.outbox_id, email_id=payload.email_id)
        result  = run_classify(payload, _classify_pipeline)
        log.info("processing_succeeded",
                 queue=CONSUME_QUEUE, outbox_id=payload.outbox_id, email_id=payload.email_id)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        result.meta = ResponseMeta(
            elapsed_ms=elapsed_ms,
            source="consumer.classify",
            model_version=_resolve_model_version(_classify_pipeline),
        )

        log.info("publish_attempt",
                 queue=CONSUME_QUEUE,
                 target_exchange=AI2APP_EXCHANGE,
                 target_routing_key=PUBLISH_ROUTING_KEY,
                 outbox_id=payload.outbox_id,
                 email_id=payload.email_id)
        try:
            publish(ch, PUBLISH_ROUTING_KEY, _build_backend_classify_payload(result))
            log.info("publish_succeeded",
                     queue=CONSUME_QUEUE,
                     target_exchange=AI2APP_EXCHANGE,
                     target_routing_key=PUBLISH_ROUTING_KEY,
                     outbox_id=payload.outbox_id,
                     email_id=payload.email_id)
        except Exception as e:
            log.error("publish_failed",
                      queue=CONSUME_QUEUE,
                      target_exchange=AI2APP_EXCHANGE,
                      target_routing_key=PUBLISH_ROUTING_KEY,
                      outbox_id=payload.outbox_id,
                      email_id=payload.email_id,
                      exception_type=type(e).__name__,
                      error=str(e))
            raise

        _safe_ack(ch, method.delivery_tag, payload.outbox_id, payload.email_id)

        log.info("processed",
                 queue=CONSUME_QUEUE, outbox_id=payload.outbox_id, email_id=payload.email_id,
                 success=True, elapsed_ms=elapsed_ms)

    except json.JSONDecodeError as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.error("json_parse_failed",
                  queue=CONSUME_QUEUE, outbox_id=outbox_id, email_id=email_id,
                  success=False, elapsed_ms=elapsed_ms, error=str(e), retry_count=retry_count)
        _publish_failed_message(
            ch,
            outbox_id=outbox_id,
            email_id=email_id,
            body=body,
            error=str(e),
            retry_count=retry_count,
            exception_type=type(e).__name__,
        )
        _safe_ack(ch, method.delivery_tag, outbox_id, email_id)

    except ValidationError as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.error("schema_validation_failed",
                  queue=CONSUME_QUEUE, outbox_id=outbox_id, email_id=email_id,
                  success=False, elapsed_ms=elapsed_ms, error=str(e), retry_count=retry_count)
        _publish_failed_message(
            ch,
            outbox_id=outbox_id,
            email_id=email_id,
            body=body,
            error=str(e),
            retry_count=retry_count,
            exception_type=type(e).__name__,
        )
        _safe_ack(ch, method.delivery_tag, outbox_id, email_id)

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        is_permanent = _is_permanent_processing_error(e)
        should_retry = (not is_permanent) and retry_count < MAX_RETRY_COUNT
        log.error("processing_failed",
                  queue=CONSUME_QUEUE, outbox_id=outbox_id, email_id=email_id,
                  success=False, elapsed_ms=elapsed_ms,
                  exception_type=type(e).__name__, error=str(e),
                  retry_count=retry_count, should_retry=should_retry)

        if should_retry:
            _safe_nack(ch, method.delivery_tag, outbox_id, email_id, requeue=False)
            return

        _publish_failed_message(
            ch,
            outbox_id=outbox_id,
            email_id=email_id,
            body=body,
            error=str(e),
            retry_count=retry_count,
            exception_type=type(e).__name__,
        )
        _safe_ack(ch, method.delivery_tag, outbox_id, email_id)


# ── 메인 ─────────────────────────────────────────────────────
def main():
    global _classify_pipeline
    log.info("pipeline_loading", queue=CONSUME_QUEUE, path_role="classify-core")
    model = load_classify_pipeline()
    _classify_pipeline = {"model": model, "predict": predict_email}
    log.info("pipeline_ready", queue=CONSUME_QUEUE, path_role="classify-core")

    runner = ClassifyConsumerRunner(_classify_pipeline)
    runner.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("shutdown", queue=CONSUME_QUEUE)
        runner.stop()


if __name__ == "__main__":
    main()
