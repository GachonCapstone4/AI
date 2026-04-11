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

import pika
from pydantic import ValidationError

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from api.schemas import ClassifyRequest, ResponseMeta
from api.services.classify_service import run_classify
from messaging.publisher import publish, AI2APP_EXCHANGE
from messaging.structured_log import get_logger
from inference import load_classify_pipeline, predict_email

# ── 설정 ─────────────────────────────────────────────────────
RABBITMQ_URL    = os.getenv("RABBITMQ_URL", "amqp://admin:admin1234!@192.168.2.20:30672/")
CONSUME_QUEUE        = "q.2ai.classify"
PUBLISH_QUEUE        = "q.2app.classify"
PUBLISH_ROUTING_KEY  = "2app.classify"
PREFETCH_COUNT       = 1

log = get_logger("consumer.classify")

_classify_pipeline: dict = {}


# ── 콜백 ─────────────────────────────────────────────────────
def _callback(ch, method, _properties, body):
    outbox_id = "(unknown)"
    email_id  = "(unknown)"
    t0 = time.perf_counter()

    try:
        data      = json.loads(body)
        outbox_id = data.get("outbox_id", outbox_id)
        email_id  = data.get("email_id",  email_id)

        log.info("received",
                 queue=CONSUME_QUEUE, outbox_id=outbox_id, email_id=email_id)

        payload = ClassifyRequest(**data)
        result  = run_classify(payload, _classify_pipeline)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        result.meta = ResponseMeta(elapsed_ms=elapsed_ms, source="consumer.classify")

        publish(ch, PUBLISH_ROUTING_KEY, result.model_dump())
        ch.basic_ack(delivery_tag=method.delivery_tag)

        log.info("processed",
                 queue=CONSUME_QUEUE, outbox_id=outbox_id, email_id=email_id,
                 success=True, elapsed_ms=elapsed_ms)

    except json.JSONDecodeError as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.error("json_parse_failed",
                  queue=CONSUME_QUEUE, outbox_id=outbox_id, email_id=email_id,
                  success=False, elapsed_ms=elapsed_ms, error=str(e))
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except ValidationError as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.error("schema_validation_failed",
                  queue=CONSUME_QUEUE, outbox_id=outbox_id, email_id=email_id,
                  success=False, elapsed_ms=elapsed_ms, error=str(e))
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.error("processing_failed",
                  queue=CONSUME_QUEUE, outbox_id=outbox_id, email_id=email_id,
                  success=False, elapsed_ms=elapsed_ms, error=str(e))
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


# ── 메인 ─────────────────────────────────────────────────────
def main():
    global _classify_pipeline

    log.info("pipeline_loading", queue=CONSUME_QUEUE, path_role="classify-core")
    model     = load_classify_pipeline()
    _classify_pipeline = {"model": model, "predict": predict_email}
    log.info("pipeline_ready",   queue=CONSUME_QUEUE, path_role="classify-core")

    while True:
        try:
            conn = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
            ch   = conn.channel()
            ch.basic_qos(prefetch_count=PREFETCH_COUNT)
            ch.basic_consume(queue=CONSUME_QUEUE, on_message_callback=_callback)
            log.info("consuming", queue=CONSUME_QUEUE)
            ch.start_consuming()

        except pika.exceptions.AMQPConnectionError as e:
            log.warning("connection_lost", queue=CONSUME_QUEUE, error=str(e),
                        retry_in_sec=5)
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("shutdown", queue=CONSUME_QUEUE)
            break


if __name__ == "__main__":
    main()
