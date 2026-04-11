# ============================================================
# draft consumer
#
# 내부 실험 / fallback 용 deprecated candidate 경로.
# classify 코어 경로와 분리하기 위해 최소 파이프라인만 사용한다.
# TODO: draft 가 RAG 서버로 이전되면 이 consumer 는 중계/종료 여부를 재평가한다.
#
# Consume : q.2ai.draft   (x.app2ai.direct)
# Publish : q.2app.draft  (x.ai2app.direct)
#
# ack / nack 정책 + 에러 처리
# ---------------------------
# 성공                   → ack
# JSON 파싱 실패          → nack(requeue=False)   DLQ
# Pydantic 검증 실패      → nack(requeue=False)   DLQ
#
# ValueError (비즈니스 검증, e.g. regenerate + previous_draft 누락)
#   → 재시도해도 동일 실패, 앱이 피드백 받아야 함
#   → ErrorResponse publish → ack
#   → error_code: VALIDATION_ERROR
#
# 일시적 오류 (Claude API 다운 등)
#   → nack(requeue=True)   재시도
#   ※ 무한 루프 방지: DLQ + x-death 카운팅 권장
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

from api.schemas import DraftRequest, ErrorResponse, ResponseMeta
from api.services.draft_service import run_draft
from messaging.publisher import publish, AI2APP_EXCHANGE
from messaging.structured_log import get_logger
from inference import load_draft_pipeline

# ── 설정 ─────────────────────────────────────────────────────
RABBITMQ_URL    = os.getenv("RABBITMQ_URL", "amqp://admin:admin1234!@192.168.2.20:30672/")
CONSUME_QUEUE        = "q.2ai.draft"
PUBLISH_QUEUE        = "q.2app.draft"
PUBLISH_ROUTING_KEY  = "2app.draft"
PREFETCH_COUNT       = 1

log = get_logger("consumer.draft")

_draft_pipeline: dict = {}


# ── 에러 응답 publish ────────────────────────────────────────
def _publish_error(ch, request_id: str, email_id: str,
                   error_code: str, error_message: str,
                   elapsed_ms: float) -> None:
    err = ErrorResponse(
        request_id=request_id,
        emailId=email_id,
        error_code=error_code,
        error_message=error_message,
        meta=ResponseMeta(elapsed_ms=elapsed_ms, source="consumer.draft"),
    )
    publish(ch, PUBLISH_ROUTING_KEY, err.model_dump())


# ── 콜백 ─────────────────────────────────────────────────────
def _callback(ch, method, _properties, body):
    request_id = "(unknown)"
    email_id   = "(unknown)"
    mode       = "(unknown)"
    t0 = time.perf_counter()

    try:
        data       = json.loads(body)
        request_id = data.get("request_id", request_id)
        email_id   = data.get("emailId",    email_id)
        mode       = data.get("mode",       mode)

        log.info("received",
                 queue=CONSUME_QUEUE, request_id=request_id,
                 emailId=email_id, mode=mode)

        payload = DraftRequest(**data)
        result  = run_draft(payload, _draft_pipeline)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        result.meta = ResponseMeta(elapsed_ms=elapsed_ms, source="consumer.draft")

        publish(ch, PUBLISH_ROUTING_KEY, result.model_dump())
        ch.basic_ack(delivery_tag=method.delivery_tag)

        log.info("processed",
                 queue=CONSUME_QUEUE, request_id=request_id,
                 emailId=email_id, mode=mode, success=True, elapsed_ms=elapsed_ms)

    except json.JSONDecodeError as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.error("json_parse_failed",
                  queue=CONSUME_QUEUE, request_id=request_id, emailId=email_id,
                  mode=mode, success=False, elapsed_ms=elapsed_ms, error=str(e))
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except ValidationError as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.error("schema_validation_failed",
                  queue=CONSUME_QUEUE, request_id=request_id, emailId=email_id,
                  mode=mode, success=False, elapsed_ms=elapsed_ms, error=str(e))
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except ValueError as e:
        # 비즈니스 로직 오류 (regenerate + previous_draft 누락 등)
        # 재시도해도 동일 실패 → error 응답 publish + ack
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.warning("validation_error",
                    queue=CONSUME_QUEUE, request_id=request_id, emailId=email_id,
                    mode=mode, success=False, elapsed_ms=elapsed_ms, error=str(e))
        _publish_error(ch, request_id, email_id,
                       error_code="VALIDATION_ERROR",
                       error_message=str(e),
                       elapsed_ms=elapsed_ms)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.error("processing_failed",
                  queue=CONSUME_QUEUE, request_id=request_id, emailId=email_id,
                  mode=mode, success=False, elapsed_ms=elapsed_ms, error=str(e))
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


# ── 메인 ─────────────────────────────────────────────────────
def main():
    global _draft_pipeline

    log.info("pipeline_loading", queue=CONSUME_QUEUE, path_role="draft-deprecated")
    model     = load_draft_pipeline()
    _draft_pipeline = {"model": model}
    log.info("pipeline_ready",   queue=CONSUME_QUEUE, path_role="draft-deprecated")

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
