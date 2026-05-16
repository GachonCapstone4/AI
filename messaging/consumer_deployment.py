# ============================================================
# deployment consumer
#
# Consume : q.2ai.deployment  (x.app2ai.direct, routing key: 2ai.deployment)
# Publish : q.2app.training    (x.ai2app.direct, routing key: app.training)
#
# The consumer reuses the same ModelManager instance as the HTTP
# /deployment/preload, /validate, /switch API.
# ============================================================

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

import pika
from pydantic import ValidationError

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from api.schemas import DeploymentRequest
from messaging.publisher import enable_delivery_confirms, publish_deployment_status
from messaging.structured_log import get_logger
from src.settings import get_settings

CONSUME_QUEUE = "q.2ai.deployment"
PUBLISH_QUEUE = "q.2app.training"
SOURCE_EXCHANGE = "x.app2ai.direct"
SOURCE_ROUTING_KEY = "2ai.deployment"
PUBLISH_EXCHANGE = "x.ai2app.direct"
PUBLISH_ROUTING_KEY = "app.training"
SSE_EXCHANGE = "x.sse.fanout"
SSE_TYPE = "ai-training-updated"
SSE_SUCCESS_MESSAGE = "모델 교체가 완료되었습니다."
SSE_FAILURE_MESSAGE = "모델 교체에 실패했습니다."
CONSUME_QUEUE_ARGUMENTS = {
    "x-dead-letter-exchange": "x.retry.direct",
    "x-dead-letter-routing-key": "2ai.deployment.retry",
}
PREFETCH_COUNT = 1

log = get_logger("consumer.deployment")


class DeploymentStageError(RuntimeError):
    def __init__(self, stage: str, original: Exception) -> None:
        super().__init__(str(original))
        self.stage = stage
        self.original = original


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fallback_job_id(user_id: int | str | None) -> str:
    user_part = str(user_id) if user_id not in (None, "") else "unknown"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"deployment-{user_part}-{timestamp}-{uuid4().hex[:8]}"


def normalize_deployment_payload(payload: DeploymentRequest) -> DeploymentRequest:
    if payload.task_type is not None and payload.task_type != "deployment":
        raise ValueError(f"Unsupported deployment task_type: {payload.task_type}")
    if payload.job_type is not None and payload.job_type != "model":
        raise ValueError(f"Unsupported deployment job_type: {payload.job_type}")

    if payload.job_id:
        return payload

    return payload.model_copy(update={"job_id": _fallback_job_id(payload.user_id)})


def _running_event(payload: DeploymentRequest, stage: str, message: str) -> dict:
    return {
        "job_id": payload.job_id,
        "status": "RUNNING",
        "model_version": payload.model_version,
        "stage": stage,
        "error_message": None,
        "finished_at": None,
        "message": message,
        "timestamp": _utc_now(),
    }


def _completed_event(payload: DeploymentRequest, active_model_version: str) -> dict:
    return {
        "job_id": payload.job_id,
        "status": "COMPLETED",
        "model_version": payload.model_version,
        "active_model_version": active_model_version,
        "stage": "COMPLETED",
        "error_message": None,
        "finished_at": _utc_now(),
        "message": "Deployment completed",
    }


def _failed_event(job_id: str, model_version: str, stage: str, error_message: str) -> dict:
    return {
        "job_id": job_id,
        "status": "FAILED",
        "model_version": model_version,
        "stage": stage,
        "error_message": error_message,
        "finished_at": _utc_now(),
    }


def _sse_payload(payload: DeploymentRequest | None, message: str) -> dict:
    user_id = getattr(payload, "user_id", None)
    return {
        "user_id": user_id,
        "sse_type": SSE_TYPE,
        "data": message,
    }


def _safe_publish_sse(channel, payload: DeploymentRequest | None, message: str) -> None:
    try:
        channel.basic_publish(
            exchange=SSE_EXCHANGE,
            routing_key="",
            body=json.dumps(_sse_payload(payload, message), ensure_ascii=False).encode("utf-8"),
            properties=pika.BasicProperties(content_type="application/json"),
            mandatory=False,
        )
        log.info("sse_published", exchange=SSE_EXCHANGE, message=message)
    except Exception as exc:
        log.warning(
            "sse_publish_failed",
            exchange=SSE_EXCHANGE,
            exception_type=type(exc).__name__,
            error=str(exc),
        )


def declare_deployment_topology(channel) -> None:
    channel.exchange_declare(exchange=SOURCE_EXCHANGE, exchange_type="direct", durable=True)
    channel.queue_declare(
        queue=CONSUME_QUEUE,
        durable=True,
        arguments=CONSUME_QUEUE_ARGUMENTS,
    )
    channel.queue_bind(
        queue=CONSUME_QUEUE,
        exchange=SOURCE_EXCHANGE,
        routing_key=SOURCE_ROUTING_KEY,
    )
    channel.exchange_declare(exchange=PUBLISH_EXCHANGE, exchange_type="direct", durable=True)
    channel.queue_declare(queue=PUBLISH_QUEUE, durable=True)
    channel.queue_bind(
        queue=PUBLISH_QUEUE,
        exchange=PUBLISH_EXCHANGE,
        routing_key=PUBLISH_ROUTING_KEY,
    )


def process_deployment_message(channel, manager, payload: DeploymentRequest) -> dict:
    payload = normalize_deployment_payload(payload)
    stage = "PRELOAD"
    try:
        publish_deployment_status(
            channel,
            _running_event(payload, stage, "Preloading deployment model"),
        )
        manager.preload(payload.model_version)
    except Exception as exc:
        raise DeploymentStageError(stage, exc) from exc

    stage = "VALIDATE"
    try:
        publish_deployment_status(
            channel,
            _running_event(payload, stage, "Validating staging model"),
        )
        manager.validate()
    except Exception as exc:
        raise DeploymentStageError(stage, exc) from exc

    stage = "SWITCH"
    try:
        publish_deployment_status(
            channel,
            _running_event(payload, stage, "Switching active model"),
        )
        switch_result = manager.switch()
        event = _completed_event(payload, switch_result["model_version"])
        publish_deployment_status(channel, event)
        _safe_publish_sse(channel, payload, SSE_SUCCESS_MESSAGE)
    except Exception as exc:
        raise DeploymentStageError(stage, exc) from exc
    return event


class DeploymentConsumerRunner:
    def __init__(self, model_manager) -> None:
        self._model_manager = model_manager
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._connection = None
        self._channel = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._run,
            name="rabbitmq-deployment-consumer",
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
        settings = get_settings()
        log.info("consumer_starting", queue=CONSUME_QUEUE)

        while not self._stop_event.is_set():
            try:
                conn = pika.BlockingConnection(pika.URLParameters(settings.RABBITMQ_URL))
                ch = conn.channel()

                self._connection = conn
                self._channel = ch

                declare_deployment_topology(ch)
                enable_delivery_confirms(ch)
                ch.basic_qos(prefetch_count=PREFETCH_COUNT)
                ch.basic_consume(queue=CONSUME_QUEUE, on_message_callback=self._callback)
                log.info(
                    "consuming",
                    queue=CONSUME_QUEUE,
                    source_exchange=SOURCE_EXCHANGE,
                    source_routing_key=SOURCE_ROUTING_KEY,
                )
                ch.start_consuming()
            except pika.exceptions.AMQPConnectionError as exc:
                if self._stop_event.is_set():
                    break
                log.warning(
                    "connection_lost",
                    queue=CONSUME_QUEUE,
                    error=str(exc),
                    retry_in_sec=5,
                )
                time.sleep(5)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                log.error(
                    "consumer_crashed",
                    queue=CONSUME_QUEUE,
                    exception_type=type(exc).__name__,
                    error=str(exc),
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

    def _callback(self, ch, method, properties, body) -> None:
        del properties

        job_id = "(unknown)"
        model_version = None
        user_id = None
        stage = "PARSE"
        t0 = time.perf_counter()

        try:
            data = json.loads(body)
            if isinstance(data, dict):
                job_id = data.get("job_id", data.get("jobId", job_id))
                user_id = data.get("user_id", data.get("userId"))
                if job_id == "(unknown)":
                    if user_id not in (None, ""):
                        job_id = _fallback_job_id(user_id)
                model_version = data.get(
                    "model_version",
                    data.get("modelVersion"),
                )

            payload = DeploymentRequest(**data)
            payload = normalize_deployment_payload(payload)
            job_id = payload.job_id
            model_version = payload.model_version

            log.info(
                "processing_started",
                queue=CONSUME_QUEUE,
                job_id=job_id,
                model_version=model_version,
            )
            process_deployment_message(ch, self._model_manager, payload)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            log.info(
                "processed",
                queue=CONSUME_QUEUE,
                job_id=job_id,
                model_version=model_version,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except json.JSONDecodeError as exc:
            self._publish_failed_and_ack(ch, method.delivery_tag, job_id, model_version, user_id, stage, exc)
        except ValidationError as exc:
            self._publish_failed_and_ack(ch, method.delivery_tag, job_id, model_version, user_id, stage, exc)
        except DeploymentStageError as exc:
            self._publish_failed_and_ack(
                ch,
                method.delivery_tag,
                job_id,
                model_version,
                user_id,
                exc.stage,
                exc.original,
            )
        except Exception as exc:
            self._publish_failed_and_ack(ch, method.delivery_tag, job_id, model_version, user_id, stage, exc)

    def _publish_failed_and_ack(
        self,
        ch,
        delivery_tag,
        job_id: str,
        model_version: str | None,
        user_id: int | str | None,
        stage: str,
        exc: Exception,
    ) -> None:
        event = _failed_event(job_id, model_version, stage, str(exc))
        publish_deployment_status(ch, event)
        failure_payload = DeploymentRequest(
            job_id=job_id,
            model_version=model_version,
            user_id=user_id,
        )
        _safe_publish_sse(ch, failure_payload, SSE_FAILURE_MESSAGE)
        ch.basic_ack(delivery_tag=delivery_tag)
        log.error(
            "failed_event_published",
            queue=CONSUME_QUEUE,
            target_queue=PUBLISH_QUEUE,
            job_id=job_id,
            model_version=model_version,
            stage=stage,
            exception_type=type(exc).__name__,
            error=str(exc),
        )


def main() -> None:
    from src.model_manager import ModelManager

    model_manager = ModelManager()
    model_manager.load_initial_model()
    runner = DeploymentConsumerRunner(model_manager)
    runner.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("shutdown", queue=CONSUME_QUEUE)
        runner.stop()


if __name__ == "__main__":
    main()
