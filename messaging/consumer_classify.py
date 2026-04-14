# ============================================================
# classify consumer
#
# Consume : q.2ai.classify   (x.app2ai.direct)
# Publish : q.2app.classify  (x.ai2app.direct)
#
# ack / nack 정책
# ---------------
# 성공              → ack
# JSON 파싱 실패    → nack(requeue=False)   손상 메시지 → DLQ
# Pydantic 검증 실패→ nack(requeue=False)   스키마 불일치 → DLQ
# 일시적 오류       → nack(requeue=True)    재시도 가능
#                    ※ 무한 루프 방지: DLQ + x-death 카운팅 권장
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
PREFETCH_COUNT       = 1

log = get_logger("consumer.classify")

_classify_pipeline: dict = {}


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


# ── 콜백 ─────────────────────────────────────────────────────
def _callback(ch, method, properties, body):
    outbox_id = "(unknown)"
    email_id  = "(unknown)"
    t0 = time.perf_counter()

    try:
        log.info(
            "received_message",
            queue=CONSUME_QUEUE,
            delivery_tag=method.delivery_tag,
            routing_key=method.routing_key,
            exchange=method.exchange,
            redelivered=method.redelivered,
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
        result.meta = ResponseMeta(elapsed_ms=elapsed_ms, source="consumer.classify")

        log.info("publish_attempt",
                 queue=CONSUME_QUEUE,
                 target_exchange=AI2APP_EXCHANGE,
                 target_routing_key=PUBLISH_ROUTING_KEY,
                 outbox_id=payload.outbox_id,
                 email_id=payload.email_id)
        try:
            publish(ch, PUBLISH_ROUTING_KEY, result.model_dump())
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
                  success=False, elapsed_ms=elapsed_ms, error=str(e))
        _safe_nack(ch, method.delivery_tag, outbox_id, email_id, requeue=False)

    except ValidationError as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.error("schema_validation_failed",
                  queue=CONSUME_QUEUE, outbox_id=outbox_id, email_id=email_id,
                  success=False, elapsed_ms=elapsed_ms, error=str(e))
        _safe_nack(ch, method.delivery_tag, outbox_id, email_id, requeue=False)

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.error("processing_failed",
                  queue=CONSUME_QUEUE, outbox_id=outbox_id, email_id=email_id,
                  success=False, elapsed_ms=elapsed_ms,
                  exception_type=type(e).__name__, error=str(e))
        _safe_nack(ch, method.delivery_tag, outbox_id, email_id, requeue=True)


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
