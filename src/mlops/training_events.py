from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any


DEFAULT_RABBITMQ_PORT = 30672
DEFAULT_RABBITMQ_USERNAME = "admin"
DEFAULT_RABBITMQ_PASSWORD = "admin1234!"
DEFAULT_TRAINING_STATUS_QUEUE = "q.2app.training"
DEFAULT_SSE_EXCHANGE = "x.sse.fanout"
DEFAULT_SSE_TYPE = "ai-training-updated"


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _env_bool(name: str) -> bool:
    return (_env(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rabbitmq_config() -> dict:
    return {
        "host": _env("RABBITMQ_HOST"),
        "port": int(_env("RABBITMQ_PORT") or DEFAULT_RABBITMQ_PORT),
        "username": _env("RABBITMQ_USERNAME") or DEFAULT_RABBITMQ_USERNAME,
        "password": _env("RABBITMQ_PASSWORD") or DEFAULT_RABBITMQ_PASSWORD,
        "training_status_queue": (
            _env("TRAINING_STATUS_QUEUE") or DEFAULT_TRAINING_STATUS_QUEUE
        ),
        "sse_exchange": _env("SSE_EXCHANGE") or DEFAULT_SSE_EXCHANGE,
        "user_id": _env("ADMIN_USER_ID") or _env("USER_ID"),
        "dry_run": _env_bool("RABBITMQ_DRY_RUN"),
    }


def _publish_queue_message(config: dict, queue_name: str, payload: dict) -> None:
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
        channel.queue_declare(queue=queue_name, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
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

    if status == "running":
        payload = {
            "job_id": job_id,
            "status": "running",
        }
    elif status == "completed":
        payload = {
            "job_id": job_id,
            "status": "completed",
            "model_version": model_version,
            "finished_at": finished_at or _utc_now(),
            "metrics": metrics or {},
        }
    elif status == "failed":
        payload = {
            "job_id": job_id,
            "status": "failed",
            "error_message": error_message or "",
            "finished_at": finished_at or _utc_now(),
        }
    else:
        raise ValueError(f"Unsupported training status: {status}")

    queue_name = config["training_status_queue"]
    if effective_dry_run:
        _print_dry_run("queue", queue_name, payload)
    else:
        _publish_queue_message(config, queue_name, payload)

    return {
        "published": not effective_dry_run,
        "dry_run": effective_dry_run,
        "queue": queue_name,
        "payload": payload,
    }


def publish_sse_log(
    message: str,
    user_id: str | None = None,
    sse_type: str = DEFAULT_SSE_TYPE,
    dry_run: bool | None = None,
) -> dict:
    config = _rabbitmq_config()
    effective_dry_run = config["dry_run"] if dry_run is None else dry_run
    effective_user_id = user_id or config["user_id"]
    if not effective_user_id:
        raise ValueError("ADMIN_USER_ID or USER_ID is required for SSE log publish.")

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
    parser.add_argument("--user-id", default=_env("ADMIN_USER_ID") or _env("USER_ID") or "admin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    publish_training_status(
        job_id=args.job_id,
        status="running",
        dry_run=args.dry_run or _env_bool("RABBITMQ_DRY_RUN"),
    )
    publish_sse_log(
        user_id=args.user_id,
        message=f"[INFO] Training job {args.job_id} is running.",
        dry_run=args.dry_run or _env_bool("RABBITMQ_DRY_RUN"),
    )


if __name__ == "__main__":
    main()
