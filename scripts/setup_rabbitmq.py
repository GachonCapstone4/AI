#!/usr/bin/env python
# ============================================================
# [로컬 테스트 전용] RabbitMQ Exchange / Queue / Binding 초기화
#
# 목적
# ----
#   Docker로 띄운 로컬 RabbitMQ에 테스트용 리소스를 1회 생성한다.
#   consumer_classify.py / consumer_training.py 실행 전에 한 번만 돌리면 된다.
#
# 주의
# ----
#   이 스크립트는 로컬 개발/테스트 환경 전용이다.
#   운영 환경에서는 절대 실행하지 않는다.
#   운영 리소스(exchange, queue, binding)는 인프라 팀 또는
#   별도 IaC(Terraform, Ansible 등)에서 관리한다.
#
# 전제 조건
# ---------
#   RabbitMQ Docker 컨테이너가 localhost:5672 에서 실행 중이어야 한다.
#   (실행 방법은 이 파일 상단 주석 또는 README 참조)
#
# 실행
# ----
#   python scripts/setup_rabbitmq.py
#   RABBITMQ_URL=amqp://guest:guest@localhost:5672/ python scripts/setup_rabbitmq.py
# ============================================================

import os
import sys
import pika

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://admin:admin1234!@192.168.2.20:30672/")

# ── 생성할 리소스 정의 ────────────────────────────────────────

EXCHANGES = [
    ("x.app2ai.direct", "direct"),   # App → AI
    ("x.ai2app.direct", "direct"),   # AI → App
]

# (queue_name, exchange, binding_key)
QUEUES = [
    ("q.2ai.classify",  "x.app2ai.direct", "2ai.classify"),
    ("q.2ai.training",  "x.app2ai.direct", "q.2ai.training"),
    ("q.2app.classify", "x.ai2app.direct", "2app.classify"),
    ("q.2app.training", "x.ai2app.direct", "q.2app.training"),
]


def main():
    print("=" * 55)
    print("[로컬 테스트 전용] RabbitMQ 리소스 초기화")
    print("=" * 55)
    print(f"URL: {RABBITMQ_URL}\n")

    try:
        conn = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
    except pika.exceptions.AMQPConnectionError as e:
        print(f"[ERROR] RabbitMQ 연결 실패: {e}")
        print("Docker 컨테이너가 실행 중인지 확인하세요.")
        sys.exit(1)

    ch = conn.channel()

    # Exchange 선언
    print("── Exchange 선언 ──────────────────────────────────")
    for name, etype in EXCHANGES:
        ch.exchange_declare(exchange=name, exchange_type=etype, durable=True)
        print(f"  [OK] {name}  (type={etype}, durable=True)")

    # Queue 선언 + Binding
    print("\n── Queue 선언 + Binding ────────────────────────────")
    for queue, exchange, binding_key in QUEUES:
        ch.queue_declare(queue=queue, durable=True)
        ch.queue_bind(queue=queue, exchange=exchange, routing_key=binding_key)
        print(f"  [OK] {queue}")
        print(f"       bind: exchange={exchange}  binding_key={binding_key}")

    print("\n── Self-check 요약 ─────────────────────────────────")
    for queue, exchange, binding_key in QUEUES:
        print(f"  queue={queue}")
        print(f"    exchange={exchange}")
        print(f"    binding_key={binding_key}")

    conn.close()

    print("\n── 완료 ────────────────────────────────────────────")
    print("이제 consumer를 실행할 수 있습니다.")
    print("  python messaging/consumer_classify.py")
    print("  python messaging/consumer_training.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
