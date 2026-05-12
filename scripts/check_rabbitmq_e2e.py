#!/usr/bin/env python
# ============================================================
# RabbitMQ topology E2E check
#
# 실행:
#   python scripts/check_rabbitmq_e2e.py
#
# 환경변수:
#   RABBITMQ_URL
#   또는 RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, RABBITMQ_PASSWORD
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

import pika


APP2AI_EXCHANGE = "x.app2ai.direct"
AI2APP_EXCHANGE = "x.ai2app.direct"
SSE_EXCHANGE = "x.sse.fanout"

CLASSIFY_IN_QUEUE = "q.2ai.classify"
CLASSIFY_OUT_QUEUE = "q.2app.classify"
DEPLOYMENT_IN_QUEUE = "q.ai.deployment"
JOB_STATUS_QUEUE = "q.2app.training"
DEPLOYMENT_OUT_QUEUE = JOB_STATUS_QUEUE
TRAINING_QUEUE = "q.2app.training"

CLASSIFY_IN_RK = "2ai.classify"
CLASSIFY_OUT_RK = "2app.classify"
DEPLOYMENT_IN_RK = "deployment"
DEPLOYMENT_OUT_RK = "app.training"
TRAINING_RK = "app.training"

LONG_LOG_FIELDS = {"data", "message", "stdout", "stderr"}

PROPS = pika.BasicProperties(
    content_type="application/json",
    delivery_mode=2,
)


class CheckFailure(Exception):
    pass


def _rabbitmq_url() -> str:
    explicit_url = os.getenv("RABBITMQ_URL")
    if explicit_url:
        return explicit_url

    host = os.getenv("RABBITMQ_HOST", "192.168.2.20")
    port = os.getenv("RABBITMQ_PORT", "30672")
    username = os.getenv("RABBITMQ_USERNAME", "admin")
    password = os.getenv("RABBITMQ_PASSWORD", "admin1234!")
    return f"amqp://{username}:{password}@{host}:{port}/"


def _connect(url: str):
    params = pika.URLParameters(url)
    conn = pika.BlockingConnection(params)
    return conn, conn.channel()


def _declare_topology(conn, ch):
    ch.exchange_declare(exchange=APP2AI_EXCHANGE, exchange_type="direct", durable=True)
    ch.exchange_declare(exchange=AI2APP_EXCHANGE, exchange_type="direct", durable=True)
    ch.exchange_declare(exchange=SSE_EXCHANGE, exchange_type="fanout", durable=True)

    ch = _ensure_queue_binding(conn, ch, CLASSIFY_IN_QUEUE, APP2AI_EXCHANGE, CLASSIFY_IN_RK)
    ch = _ensure_queue_binding(conn, ch, CLASSIFY_OUT_QUEUE, AI2APP_EXCHANGE, CLASSIFY_OUT_RK)
    ch = _ensure_queue_binding(conn, ch, DEPLOYMENT_IN_QUEUE, APP2AI_EXCHANGE, DEPLOYMENT_IN_RK)
    ch = _ensure_queue_binding(conn, ch, TRAINING_QUEUE, AI2APP_EXCHANGE, TRAINING_RK)
    return ch


def _ensure_queue(conn, ch, queue: str):
    try:
        ch.queue_declare(queue=queue, passive=True)
    except pika.exceptions.ChannelClosedByBroker as exc:
        if exc.reply_code != 404:
            raise CheckFailure(
                f"queue={queue} exists with incompatible arguments or cannot be checked: {exc}"
            ) from exc
        ch = conn.channel()
        ch.queue_declare(queue=queue, durable=True)
    return ch


def _ensure_queue_binding(conn, ch, queue: str, exchange: str, routing_key: str):
    ch = _ensure_queue(conn, ch, queue)
    ch.queue_bind(queue=queue, exchange=exchange, routing_key=routing_key)
    return ch


def _publish_json(ch, exchange: str, routing_key: str, payload: dict) -> None:
    ch.basic_publish(
        exchange=exchange,
        routing_key=routing_key,
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        properties=PROPS,
        mandatory=True,
    )


def _check_publish_routable(ch, exchange: str, routing_key: str, payload: dict, label: str) -> None:
    try:
        _publish_json(ch, exchange, routing_key, payload)
    except pika.exceptions.UnroutableError as exc:
        raise CheckFailure(
            f"{label} is not routable: exchange={exchange}, routing_key={routing_key}, error={exc}"
        ) from exc


def _consume_matching(ch, queue: str, predicate, timeout: float) -> dict:
    deadline = time.time() + timeout
    last_payload = None
    while time.time() < deadline:
        method, _properties, body = ch.basic_get(queue=queue, auto_ack=False)
        if method is None:
            time.sleep(0.2)
            continue

        try:
            payload = json.loads(body.decode("utf-8"))
            last_payload = payload
            if predicate(payload):
                ch.basic_ack(method.delivery_tag)
                return payload
        except json.JSONDecodeError:
            last_payload = body.decode("utf-8", errors="replace")

        ch.basic_nack(method.delivery_tag, requeue=True)
        time.sleep(0.2)

    raise CheckFailure(f"timeout waiting for queue={queue}; last_payload={last_payload!r}")


def _publish_and_consume_matching(ch, queue: str, publish_func, predicate, timeout: float) -> dict:
    state = {
        "payload": None,
        "last_payload": None,
        "done": False,
    }

    def _callback(channel, method, _properties, body):
        try:
            payload = json.loads(body.decode("utf-8"))
            state["last_payload"] = payload
            if predicate(payload):
                state["payload"] = payload
                state["done"] = True
                channel.basic_ack(method.delivery_tag)
                channel.basic_cancel(consumer_tag)
                return
        except json.JSONDecodeError:
            state["last_payload"] = body.decode("utf-8", errors="replace")

        channel.basic_nack(method.delivery_tag, requeue=True)

    consumer_tag = ch.basic_consume(queue=queue, on_message_callback=_callback, auto_ack=False)
    publish_func()

    deadline = time.time() + timeout
    try:
        while time.time() < deadline and not state["done"]:
            ch.connection.process_data_events(time_limit=0.2)
    finally:
        if not state["done"] and ch.is_open:
            try:
                ch.basic_cancel(consumer_tag)
            except Exception:
                pass

    if state["payload"] is None:
        queue_state = _queue_state(ch, queue)
        raise CheckFailure(
            f"timeout waiting for queue={queue}; "
            f"messages={queue_state.get('message_count')}, "
            f"consumers={queue_state.get('consumer_count')}; "
            f"last_payload={state['last_payload']!r}; "
            "if consumers > 0, an existing AI/Backend consumer may have consumed the test message first"
        )
    return state["payload"]


def _queue_state(ch, queue: str) -> dict:
    try:
        result = ch.queue_declare(queue=queue, passive=True)
        return {
            "message_count": result.method.message_count,
            "consumer_count": result.method.consumer_count,
        }
    except Exception as exc:
        return {
            "message_count": "unknown",
            "consumer_count": "unknown",
            "error": str(exc),
        }


def _pass(label: str) -> None:
    print(f"[PASS] {label}")


def _info(label: str) -> None:
    print(f"[INFO] {label}")


def _fail(label: str, detail: str) -> None:
    print(f"[FAIL] {label}")
    print(f"       {detail}")


def _assert_fields(payload: dict, required: set[str], label: str) -> None:
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise CheckFailure(f"{label} missing fields: {', '.join(missing)}")


def _classify_request_payload(test_id: str) -> dict:
    return {
        "outbox_id": f"e2e-in-{test_id}",
        "email_id": f"email-{test_id}",
        "sender_email": "sender@example.com",
        "sender_name": "테스트 발신자",
        "subject": "납품 일정 문의",
        "body_clean": "이번 달 납품 일정과 회의 가능 시간을 확인 부탁드립니다.",
        "received_at": "2026-05-04T10:30:00Z",
        "_e2e_id": test_id,
    }


def _classify_result_payload(test_id: str) -> dict:
    return {
        "outbox_id": f"e2e-out-{test_id}",
        "email_id": f"email-{test_id}",
        "domain": "업무",
        "intent": "문의",
        "confidence_score": 0.91,
        "summary_text": "납품 일정 확인 요청 이메일입니다.",
        "schedule_detected": True,
        "entities_json": {
            "date": "2026-05-10",
            "time": "14:00",
            "location": "회의실 A",
        },
        "model_version": "training-final-004",
        "_e2e_id": test_id,
    }


def _training_status_payload(test_id: str, status: str = "RUNNING") -> dict:
    return {
        "job_id": f"train-e2e-{test_id}",
        "status": status,
        "model_version": "training-final-004",
        "finished_at": None if status == "RUNNING" else "2026-05-04T10:30:05Z",
        "metrics": {
            "intent_f1": None if status == "RUNNING" else 0.91,
            "domain_accuracy": None if status == "RUNNING" else 0.88,
        },
        "error_message": None,
        "_e2e_id": test_id,
    }


def _deployment_request_payload(test_id: str) -> dict:
    return {
        "job_id": f"deploy-e2e-{test_id}",
        "model_version": "training-final-004",
        "artifact_s3_uri": "s3://capstone-gachon/models/training-final-004/",
        "requested_by": "admin",
        "requested_at": "2026-05-04T10:30:00Z",
        "_e2e_id": test_id,
    }


def _deployment_status_payload(test_id: str) -> dict:
    return {
        "job_id": f"deploy-e2e-{test_id}",
        "status": "COMPLETED",
        "model_version": "training-final-004",
        "active_model_version": "training-final-004",
        "finished_at": "2026-05-04T10:30:05Z",
        "message": "Deployment completed",
        "_e2e_id": test_id,
    }


def _sse_payload(test_id: str) -> dict:
    return {
        "user_id": "admin",
        "sse_type": "ai-training-updated",
        "data": "[INFO] Training started",
        "_e2e_id": test_id,
    }


def check_classify_request(ch, test_id: str, timeout: float) -> None:
    label = f"{APP2AI_EXCHANGE} -> {CLASSIFY_IN_QUEUE}"
    payload = _classify_request_payload(test_id)
    received = _publish_and_consume_matching(
        ch,
        CLASSIFY_IN_QUEUE,
        lambda: _publish_json(ch, APP2AI_EXCHANGE, CLASSIFY_IN_RK, payload),
        lambda item: item.get("_e2e_id") == test_id,
        timeout,
    )
    _assert_fields(
        received,
        {"outbox_id", "email_id", "sender_email", "sender_name", "subject", "body_clean", "received_at"},
        label,
    )
    _pass(label)


def check_classify_result(ch, test_id: str, timeout: float) -> None:
    label = f"{AI2APP_EXCHANGE} -> {CLASSIFY_OUT_QUEUE}"
    payload = _classify_result_payload(test_id)
    received = _publish_and_consume_matching(
        ch,
        CLASSIFY_OUT_QUEUE,
        lambda: _publish_json(ch, AI2APP_EXCHANGE, CLASSIFY_OUT_RK, payload),
        lambda item: item.get("_e2e_id") == test_id,
        timeout,
    )
    _assert_fields(
        received,
        {
            "outbox_id",
            "email_id",
            "domain",
            "intent",
            "confidence_score",
            "summary_text",
            "schedule_detected",
            "entities_json",
            "model_version",
        },
        label,
    )
    _pass(label)

    if not isinstance(received.get("entities_json"), dict):
        raise CheckFailure(
            f"q.2app.classify entities_json must be object; got {type(received.get('entities_json')).__name__}"
        )
    _pass("entities_json is object")


def check_training_status(ch, test_id: str, timeout: float) -> None:
    label = f"{AI2APP_EXCHANGE} -> {TRAINING_QUEUE}"
    payload = _training_status_payload(test_id, "RUNNING")
    received = _publish_and_consume_matching(
        ch,
        TRAINING_QUEUE,
        lambda: _publish_json(ch, AI2APP_EXCHANGE, TRAINING_RK, payload),
        lambda item: item.get("_e2e_id") == test_id,
        timeout,
    )
    _assert_fields(
        received,
        {"job_id", "status", "model_version", "finished_at", "metrics", "error_message"},
        label,
    )
    if received.get("status") not in {"RUNNING", "COMPLETED", "FAILED"}:
        raise CheckFailure(f"q.2app.training status is invalid: {received.get('status')}")
    _pass(label)

    long_fields = sorted(LONG_LOG_FIELDS & set(received.keys()))
    if long_fields:
        raise CheckFailure(
            f"q.2app.training payload must not include long log fields: {', '.join(long_fields)}"
        )
    _pass("training status payload has no long log fields")


def check_deployment_request(ch, test_id: str, timeout: float) -> None:
    label = f"{APP2AI_EXCHANGE} -> {DEPLOYMENT_IN_QUEUE}"
    payload = _deployment_request_payload(test_id)
    received = _publish_and_consume_matching(
        ch,
        DEPLOYMENT_IN_QUEUE,
        lambda: _publish_json(ch, APP2AI_EXCHANGE, DEPLOYMENT_IN_RK, payload),
        lambda item: item.get("_e2e_id") == test_id,
        timeout,
    )
    _assert_fields(received, {"job_id", "model_version"}, label)
    _pass(label)


def check_deployment_status(ch, test_id: str, timeout: float) -> None:
    label = f"{AI2APP_EXCHANGE} -> {DEPLOYMENT_OUT_QUEUE}"
    payload = _deployment_status_payload(test_id)
    received = _publish_and_consume_matching(
        ch,
        DEPLOYMENT_OUT_QUEUE,
        lambda: _publish_json(ch, AI2APP_EXCHANGE, DEPLOYMENT_OUT_RK, payload),
        lambda item: item.get("_e2e_id") == test_id,
        timeout,
    )
    _assert_fields(
        received,
        {"job_id", "status", "model_version", "active_model_version", "finished_at"},
        label,
    )
    _pass(label)


def check_sse_fanout(ch, test_id: str, timeout: float) -> None:
    label = f"{SSE_EXCHANGE} -> temporary SSE queue"
    result = ch.queue_declare(queue="", exclusive=True, auto_delete=True)
    temp_queue = result.method.queue
    ch.queue_bind(queue=temp_queue, exchange=SSE_EXCHANGE, routing_key="")

    payload = _sse_payload(test_id)
    _publish_json(ch, SSE_EXCHANGE, "", payload)
    received = _consume_matching(
        ch,
        temp_queue,
        lambda item: item.get("_e2e_id") == test_id,
        timeout,
    )
    _assert_fields(received, {"user_id", "sse_type", "data"}, label)
    _pass(label)


def _print_queue_state(ch, queue: str) -> None:
    state = _queue_state(ch, queue)
    if "error" in state:
        _info(f"{queue}: message_count={state['message_count']} consumer_count={state['consumer_count']} error={state['error']}")
        return
    _info(f"{queue}: message_count={state['message_count']} consumer_count={state['consumer_count']}")


def check_classify_request_topology(ch, test_id: str) -> None:
    label = f"{APP2AI_EXCHANGE} -> {CLASSIFY_IN_QUEUE}"
    _check_publish_routable(
        ch,
        APP2AI_EXCHANGE,
        CLASSIFY_IN_RK,
        _classify_request_payload(test_id),
        label,
    )
    _print_queue_state(ch, CLASSIFY_IN_QUEUE)
    _pass(label)


def check_classify_result_topology(ch, test_id: str) -> None:
    label = f"{AI2APP_EXCHANGE} -> {CLASSIFY_OUT_QUEUE}"
    payload = _classify_result_payload(test_id)
    _check_publish_routable(ch, AI2APP_EXCHANGE, CLASSIFY_OUT_RK, payload, label)
    _print_queue_state(ch, CLASSIFY_OUT_QUEUE)
    _pass(label)

    if not isinstance(payload.get("entities_json"), dict):
        raise CheckFailure(
            f"q.2app.classify entities_json must be object; got {type(payload.get('entities_json')).__name__}"
        )
    _pass("entities_json is object")


def check_training_status_topology(ch, test_id: str) -> None:
    label = f"{AI2APP_EXCHANGE} -> {TRAINING_QUEUE}"
    payload = _training_status_payload(test_id, "RUNNING")
    _check_publish_routable(ch, AI2APP_EXCHANGE, TRAINING_RK, payload, label)
    _print_queue_state(ch, TRAINING_QUEUE)

    if payload.get("status") not in {"RUNNING", "COMPLETED", "FAILED"}:
        raise CheckFailure(f"q.2app.training status is invalid: {payload.get('status')}")
    _pass(label)

    long_fields = sorted(LONG_LOG_FIELDS & set(payload.keys()))
    if long_fields:
        raise CheckFailure(
            f"q.2app.training payload must not include long log fields: {', '.join(long_fields)}"
        )
    _pass("training status payload has no long log fields")


def check_deployment_request_topology(ch, test_id: str) -> None:
    label = f"{APP2AI_EXCHANGE} -> {DEPLOYMENT_IN_QUEUE}"
    payload = _deployment_request_payload(test_id)
    _check_publish_routable(ch, APP2AI_EXCHANGE, DEPLOYMENT_IN_RK, payload, label)
    _print_queue_state(ch, DEPLOYMENT_IN_QUEUE)
    _pass(label)


def check_deployment_status_topology(ch, test_id: str) -> None:
    label = f"{AI2APP_EXCHANGE} -> {DEPLOYMENT_OUT_QUEUE}"
    payload = _deployment_status_payload(test_id)
    _check_publish_routable(ch, AI2APP_EXCHANGE, DEPLOYMENT_OUT_RK, payload, label)
    _print_queue_state(ch, DEPLOYMENT_OUT_QUEUE)
    _pass(label)


def main() -> int:
    parser = argparse.ArgumentParser(description="RabbitMQ topology E2E check")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--topology-only",
        action="store_true",
        help="Check exchanges, queues, bindings, routable publishes, and SSE fanout without consuming target queues.",
    )
    args = parser.parse_args()

    url = _rabbitmq_url()
    safe_url = url
    if "@" in safe_url and "://" in safe_url:
        scheme, rest = safe_url.split("://", 1)
        if "@" in rest:
            safe_url = f"{scheme}://***:***@{rest.split('@', 1)[1]}"

    print(f"RabbitMQ: {safe_url}")
    print(f"Timeout : {args.timeout}s")

    if args.topology_only:
        checks = [
            check_classify_request_topology,
            check_classify_result_topology,
            check_deployment_request_topology,
            check_deployment_status_topology,
            check_training_status_topology,
        ]
    else:
        checks = [
            check_classify_request,
            check_classify_result,
            check_deployment_request,
            check_deployment_status,
            check_training_status,
        ]
    test_id = uuid.uuid4().hex
    failures = 0

    try:
        conn, ch = _connect(url)
    except Exception as exc:
        _fail("RabbitMQ connection", str(exc))
        return 1

    try:
        try:
            ch = _declare_topology(conn, ch)
            _pass("exchange/queue/binding declare")
        except Exception as exc:
            _fail("exchange/queue/binding declare", str(exc))
            return 1

        for check in checks:
            try:
                if args.topology_only:
                    check(ch, test_id)
                else:
                    check(ch, test_id, args.timeout)
            except Exception as exc:
                failures += 1
                _fail(check.__name__, str(exc))

        try:
            check_sse_fanout(ch, test_id, args.timeout)
        except Exception as exc:
            failures += 1
            _fail("check_sse_fanout", str(exc))

        if failures:
            print(f"\nFAIL: {failures} check(s) failed")
            return 1

        mode = "topology-only" if args.topology_only else "full E2E"
        print(f"\nPASS: RabbitMQ {mode} check completed")
        return 0
    finally:
        if not conn.is_closed:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
