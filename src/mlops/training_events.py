from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any


logger = logging.getLogger(__name__)

DEFAULT_RABBITMQ_PORT = 30672
DEFAULT_RABBITMQ_USERNAME = "admin"
DEFAULT_RABBITMQ_PASSWORD = "admin1234!"
DEFAULT_TRAINING_STATUS_QUEUE = "q.2app.training"
DEFAULT_TRAINING_STATUS_ROUTING_KEY = "app.training"
DEFAULT_AI2APP_EXCHANGE = "x.ai2app.direct"
DEFAULT_SSE_EXCHANGE = "x.sse.fanout"
DEFAULT_SSE_TYPE = "ai-training-updated"


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _env_bool(name: str) -> bool:
    return (_env(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _sse_user_id_from_env() -> tuple[str | None, str]:
    admin_user_id = _env("ADMIN_USER_ID")
    if admin_user_id:
        return admin_user_id, "ADMIN_USER_ID"
    return _env("USER_ID"), "USER_ID"


def _parse_sse_user_id(value: str | int | None, source: str) -> int:
    if value is None or value == "":
        message = f"{source} is required for SSE log publish."
        logger.error(message)
        raise ValueError(message)

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        message = f"{source} must be an integer for SSE log publish: {value!r}"
        logger.error(message)
        raise ValueError(message) from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _training_metrics_payload(metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "intent_f1": metrics.get("intent_f1"),
        "domain_accuracy": metrics.get("domain_accuracy"),
    }


def _training_status_payload(
    *,
    job_id: str,
    status: str,
    model_version: str | None,
    finished_at: str | None,
    metrics: dict[str, Any] | None,
    error_message: str | None,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": status,
        "model_version": model_version,
        "finished_at": finished_at,
        "metrics": _training_metrics_payload(metrics),
        "error_message": error_message,
    }


def _rabbitmq_config() -> dict:
    user_id, user_id_source = _sse_user_id_from_env()
    return {
        "host": _env("RABBITMQ_HOST"),
        "port": int(_env("RABBITMQ_PORT") or DEFAULT_RABBITMQ_PORT),
        "username": _env("RABBITMQ_USERNAME") or DEFAULT_RABBITMQ_USERNAME,
        "password": _env("RABBITMQ_PASSWORD") or DEFAULT_RABBITMQ_PASSWORD,
        "training_status_queue": (
            _env("TRAINING_STATUS_QUEUE") or DEFAULT_TRAINING_STATUS_QUEUE
        ),
        "training_status_routing_key": (
            _env("TRAINING_STATUS_ROUTING_KEY") or DEFAULT_TRAINING_STATUS_ROUTING_KEY
        ),
        "ai2app_exchange": _env("AI2APP_EXCHANGE") or DEFAULT_AI2APP_EXCHANGE,
        "sse_exchange": _env("SSE_EXCHANGE") or DEFAULT_SSE_EXCHANGE,
        "user_id": user_id,
        "user_id_source": user_id_source,
        "dry_run": _env_bool("RABBITMQ_DRY_RUN"),
    }


def _publish_queue_message(
    config: dict,
    queue_name: str,
    routing_key: str,
    payload: dict,
) -> None:
    if not config["host"]:
        raise ValueError("RABBITMQ_HOST is required for RabbitMQ publish.")

    import pika

    credentials = pika.PlainCredentials(config["username"], config["password"])
    parameters = pika.ConnectionParameters(
        host=config["host"],
        port=config["port"],
        credentials=credentials,
    )
    connection = pika.BlockingConnection(parameters)
    try:
        channel = connection.channel()
        channel.exchange_declare(
            exchange=config["ai2app_exchange"],
            exchange_type="direct",
            durable=True,
        )
        channel.queue_declare(queue=queue_name, durable=True)
        channel.queue_bind(
            queue=queue_name,
            exchange=config["ai2app_exchange"],
            routing_key=routing_key,
        )
        channel.basic_publish(
            exchange=config["ai2app_exchange"],
            routing_key=routing_key,
            body=json.dumps(payload, ensure_ascii=False),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
            ),
        )
    finally:
        connection.close()


def _publish_exchange_message(config: dict, exchange_name: str, payload: dict) -> None:
    if not config["host"]:
        raise ValueError("RABBITMQ_HOST is required for RabbitMQ publish.")

    import pika

    credentials = pika.PlainCredentials(config["username"], config["password"])
    parameters = pika.ConnectionParameters(
        host=config["host"],
        port=config["port"],
        credentials=credentials,
    )
    connection = pika.BlockingConnection(parameters)
    try:
        channel = connection.channel()
        channel.exchange_declare(exchange=exchange_name, exchange_type="fanout", durable=True)
        channel.basic_publish(
            exchange=exchange_name,
            routing_key="",
            body=json.dumps(payload, ensure_ascii=False),
            properties=pika.BasicProperties(content_type="application/json"),
        )
    finally:
        connection.close()


def _print_dry_run(kind: str, target: str, payload: dict) -> None:
    print(
        json.dumps(
            {
                "dry_run": True,
                "kind": kind,
                "target": target,
                "payload": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def publish_training_status(
    job_id: str,
    status: str,
    model_version: str | None = None,
    metrics: dict[str, Any] | None = None,
    error_message: str | None = None,
    finished_at: str | None = None,
    dry_run: bool | None = None,
) -> dict:
    config = _rabbitmq_config()
    effective_dry_run = config["dry_run"] if dry_run is None else dry_run
    normalized_status = status.upper()

    if normalized_status == "RUNNING":
        payload = _training_status_payload(
            job_id=job_id,
            status="RUNNING",
            model_version=model_version,
            finished_at=finished_at,
            metrics=metrics,
            error_message=error_message,
        )
    elif normalized_status == "COMPLETED":
        payload = _training_status_payload(
            job_id=job_id,
            status="COMPLETED",
            model_version=model_version,
            finished_at=finished_at or _utc_now(),
            metrics=metrics,
            error_message=error_message,
        )
    elif normalized_status == "FAILED":
        payload = _training_status_payload(
            job_id=job_id,
            status="FAILED",
            model_version=model_version,
            finished_at=finished_at or _utc_now(),
            metrics=metrics,
            error_message=error_message or "",
        )
    else:
        raise ValueError(f"Unsupported training status: {status}")

    queue_name = config["training_status_queue"]
    routing_key = config["training_status_routing_key"]
    if effective_dry_run:
        _print_dry_run("direct_exchange", config["ai2app_exchange"], payload)
    else:
        _publish_queue_message(config, queue_name, routing_key, payload)

    return {
        "published": not effective_dry_run,
        "dry_run": effective_dry_run,
        "queue": queue_name,
        "exchange": config["ai2app_exchange"],
        "routing_key": routing_key,
        "payload": payload,
    }


def publish_sse_log(
    message: str,
    user_id: str | int | None = None,
    sse_type: str = DEFAULT_SSE_TYPE,
    dry_run: bool | None = None,
) -> dict:
    config = _rabbitmq_config()
    effective_dry_run = config["dry_run"] if dry_run is None else dry_run
    raw_user_id = user_id if user_id is not None else config["user_id"]
    user_id_source = "user_id" if user_id is not None else config["user_id_source"]
    effective_user_id = _parse_sse_user_id(raw_user_id, user_id_source)

    payload = {
        "user_id": effective_user_id,
        "sse_type": sse_type,
        "data": message,
    }

    exchange_name = config["sse_exchange"]
    if effective_dry_run:
        _print_dry_run("fanout_exchange", exchange_name, payload)
    else:
        _publish_exchange_message(config, exchange_name, payload)

    return {
        "published": not effective_dry_run,
        "dry_run": effective_dry_run,
        "exchange": exchange_name,
        "payload": payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish training status and SSE log events for the training container."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--user-id", default=_env("ADMIN_USER_ID") or _env("USER_ID"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    publish_training_status(
        job_id=args.job_id,
        status="RUNNING",
        dry_run=args.dry_run or _env_bool("RABBITMQ_DRY_RUN"),
    )
    publish_sse_log(
        user_id=args.user_id,
        message=f"[INFO] Training job {args.job_id} is running.",
        dry_run=args.dry_run or _env_bool("RABBITMQ_DRY_RUN"),
    )


if __name__ == "__main__":
    main()
