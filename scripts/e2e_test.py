#!/usr/bin/env python
# ============================================================
# RabbitMQ End-to-End 테스트 스크립트
#
# 전제 조건
# ----------
# 1. RabbitMQ 실행 중 (기본 amqp://guest:guest@localhost:5672/)
# 2. consumer_classify.py 실행 중
#
# 실행
# ----
#   python scripts/e2e_test.py
#   RABBITMQ_URL=amqp://user:pass@host:5672/ python scripts/e2e_test.py
#
# 옵션
# ----
#   --timeout 30     응답 대기 최대 초 (기본 30)
#   classify consumer / publisher 계약만 테스트
# ============================================================

import sys
import os
import json
import time
import uuid
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import pika

RABBITMQ_URL    = os.getenv("RABBITMQ_URL", "amqp://admin:admin1234!@192.168.2.20:30672/")
APP2AI_EXCHANGE = "x.app2ai.direct"
AI2APP_EXCHANGE = "x.ai2app.direct"

CLASSIFY_IN  = "q.2ai.classify"
CLASSIFY_OUT = "q.2app.classify"

CLASSIFY_IN_RK  = "2ai.classify"

_PROPS = pika.BasicProperties(
    content_type="application/json",
    delivery_mode=2,
)


# ── 헬퍼 ─────────────────────────────────────────────────────

def _connect() -> tuple:
    conn = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
    ch   = conn.channel()
    return conn, ch



def _publish(ch, exchange: str, routing_key: str, message: dict):
    ch.basic_publish(
        exchange=exchange,
        routing_key=routing_key,
        body=json.dumps(message, ensure_ascii=False).encode(),
        properties=_PROPS,
    )


def _poll(ch, queue: str, match_key: str, match_val, timeout: int) -> dict | None:
    """
    지정 큐를 polling 하여 match_key == match_val 인 메시지 반환.
    timeout 초 안에 없으면 None.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        method, _, body = ch.basic_get(queue=queue, auto_ack=True)
        if body:
            msg = json.loads(body)
            if msg.get(match_key) == match_val:
                return msg
        time.sleep(0.5)
    return None


def _print_result(label: str, ok: bool, elapsed: float, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}  ({elapsed:.0f} ms)"
          + (f"\n         {detail}" if detail else ""))


# ── classify E2E ─────────────────────────────────────────────

def test_classify(ch, timeout: int) -> dict | None:
    print("\n── classify E2E ──────────────────────────────────────")
    outbox_id = int(uuid.uuid4().int % 100000)
    payload = {
        "outbox_id":    outbox_id,
        "email_id":     1,
        "sender_email": "test@example.com",
        "sender_name":  "테스트",
        "subject":      "납품 일정 문의",
        "body_clean":   "이번 달 납품 일정을 알려주시겠어요? 빠른 확인 부탁드립니다.",
        "received_at":  "2026-04-06T10:00:00",
    }

    print(f"  Publish  → {CLASSIFY_IN}  outbox_id={outbox_id}")
    t0 = time.perf_counter()
    _publish(ch, APP2AI_EXCHANGE, CLASSIFY_IN_RK, payload)

    resp = _poll(ch, CLASSIFY_OUT, "outbox_id", outbox_id, timeout)
    elapsed = (time.perf_counter() - t0) * 1000

    if resp is None:
        _print_result("classify response", False, elapsed,
                      f"timeout({timeout}s) — consumer 실행 중인지 확인")
        return None

    # 필수 필드 검증
    errors = []
    for f in ["outbox_id", "email_id", "classification", "summary", "email_embedding"]:
        if f not in resp:
            errors.append(f"missing field: {f}")
    if resp.get("outbox_id") != outbox_id:
        errors.append(f"outbox_id mismatch: {resp.get('outbox_id')}")
    if not isinstance(resp.get("email_embedding"), list):
        errors.append("email_embedding is not a list")
    if resp.get("status") == "error":
        errors.append(f"error response: {resp.get('error_message')}")

    ok = len(errors) == 0
    _print_result("classify response",  ok, elapsed,
                  " | ".join(errors) if errors else "")
    if ok:
        print(f"         domain={resp['classification'].get('domain')} "
              f"intent={resp['classification'].get('intent')}")
        meta = resp.get("meta") or {}
        if meta:
            print(f"         meta.elapsed_ms={meta.get('elapsed_ms')} "
                  f"source={meta.get('source')}")
    return resp

# ── 메인 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI 서버 E2E 테스트")
    parser.add_argument("--timeout",       type=int, default=30)
    args = parser.parse_args()

    print(f"RabbitMQ: {RABBITMQ_URL}")
    print(f"Timeout : {args.timeout}s")

    try:
        conn, ch = _connect()
    except pika.exceptions.AMQPConnectionError as e:
        print(f"\n[ERROR] RabbitMQ 연결 실패: {e}")
        sys.exit(1)

    try:
        test_classify(ch, args.timeout)

    finally:
        if not conn.is_closed:
            conn.close()

    print("\n──────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
