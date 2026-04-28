"""
데이터 수집 배치 스크립트
- DB에서 학습 데이터 추출
- CSV 파일 생성
- S3 업로드
- SSE 로그 발행 (x.sse.fanout)
- 완료 이벤트 발행 (q.2app.training)

환경변수:
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    S3_BUCKET, S3_DATASET_KEY
    RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, RABBITMQ_PASSWORD
    JOB_ID, ADMIN_USER_ID
"""

import os
import sys
import csv
import json
import logging
import tempfile
from datetime import datetime, timezone

import boto3
import mysql.connector
import pika

# ============================================================
# 로깅 설정
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# 환경변수
# ============================================================
DB_HOST     = os.environ.get("DB_HOST", "192.168.3.10")
DB_PORT     = int(os.environ.get("DB_PORT", "3306"))
DB_USER     = os.environ.get("DB_USER", "capstone")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "capstone")
DB_NAME     = os.environ.get("DB_NAME", "email_agent")

AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.environ.get("AWS_REGION", "ap-northeast-2")
S3_BUCKET             = os.environ.get("S3_BUCKET", "capstone-gachon")
S3_DATASET_KEY        = os.environ.get("S3_DATASET_KEY", "dataset/dataset_new.csv")

RABBITMQ_HOST     = os.environ.get("RABBITMQ_HOST", "192.168.2.20")
RABBITMQ_PORT     = int(os.environ.get("RABBITMQ_PORT", "30672"))
RABBITMQ_USERNAME = os.environ.get("RABBITMQ_USERNAME", "admin")
RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD", "admin1234!")

JOB_ID        = os.environ.get("JOB_ID", "unknown-job")
ADMIN_USER_ID = os.environ.get("ADMIN_USER_ID", "0")

# RabbitMQ 상수
EXCHANGE_SSE_FANOUT  = "x.sse.fanout"
QUEUE_TRAINING_RESULT = "q.2app.training"
SSE_TYPE = "ai-training-updated"


# ============================================================
# RabbitMQ 연결
# ============================================================
def connect_rabbitmq():
    credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials,
        heartbeat=60
    )
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    return connection, channel


# ============================================================
# SSE 로그 발행
# ============================================================
def publish_sse_log(channel, message: str):
    payload = {
        "user_id": ADMIN_USER_ID,
        "sse_type": SSE_TYPE,
        "data": message
    }
    try:
        channel.basic_publish(
            exchange=EXCHANGE_SSE_FANOUT,
            routing_key="",
            body=json.dumps(payload, ensure_ascii=False),
            properties=pika.BasicProperties(content_type="application/json")
        )
        logger.info(f"SSE 발행: {message}")
    except Exception as e:
        logger.warning(f"SSE 발행 실패: {e}")


# ============================================================
# 완료/실패 이벤트 발행
# ============================================================
def publish_training_event(channel, status: str, error_message: str = None, dataset_version: str = None):
    payload = {
        "job_id": JOB_ID,
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat()
    }
    if dataset_version:
        payload["dataset_version"] = dataset_version
    if error_message:
        payload["error_message"] = error_message

    try:
        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_TRAINING_RESULT,
            body=json.dumps(payload, ensure_ascii=False),
            properties=pika.BasicProperties(content_type="application/json")
        )
        logger.info(f"학습 이벤트 발행: status={status}")
    except Exception as e:
        logger.warning(f"학습 이벤트 발행 실패: {e}")


# ============================================================
# DB에서 학습 데이터 추출
# ============================================================
def fetch_training_data():
    logger.info("DB 연결 중...")
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4"
    )
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT
            CONCAT('train_', e.email_id) AS emailId,
            e.external_msg_id            AS threadId,
            e.sender_email               AS `from`,
            e.subject                    AS subject,
            e.body_clean                 AS body,
            ear.domain                   AS domain,
            ear.intent                   AS intent
        FROM emails e
        INNER JOIN email_analysis_results ear
            ON e.email_id = ear.email_id
        WHERE ear.domain IS NOT NULL
          AND ear.intent IS NOT NULL
          AND e.body_clean IS NOT NULL
          AND e.body_clean != ''
        ORDER BY e.email_id ASC
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    logger.info(f"총 {len(rows)}건 추출 완료")
    return rows


# ============================================================
# CSV 파일 생성
# ============================================================
def create_csv(rows: list, filepath: str):
    fieldnames = ["emailId", "threadId", "from", "subject", "body", "domain", "intent"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV 파일 생성 완료: {filepath} ({len(rows)}행)")


# ============================================================
# S3 업로드
# ============================================================
def upload_to_s3(filepath: str):
    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    s3_client.upload_file(filepath, S3_BUCKET, S3_DATASET_KEY)
    s3_uri = f"s3://{S3_BUCKET}/{S3_DATASET_KEY}"
    logger.info(f"S3 업로드 완료: {s3_uri}")
    return s3_uri


# ============================================================
# 메인
# ============================================================
def main():
    logger.info(f"===== 데이터 수집 배치 시작 — job_id={JOB_ID} =====")

    # RabbitMQ 연결
    connection, channel = connect_rabbitmq()

    try:
        # 1. DB 데이터 추출
        publish_sse_log(channel, "[INFO] DB 데이터 추출 시작")
        rows = fetch_training_data()
        publish_sse_log(channel, f"[INFO] {len(rows)}건 추출 완료")

        if len(rows) == 0:
            raise ValueError("추출된 데이터가 없습니다. domain/intent 분류된 이메일을 확인해주세요.")

        # 2. CSV 생성
        publish_sse_log(channel, "[INFO] CSV 파일 변환 중")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name

        create_csv(rows, tmp_path)
        publish_sse_log(channel, "[INFO] CSV 파일 생성 완료")

        # 3. S3 업로드
        publish_sse_log(channel, "[INFO] S3 업로드 시작")
        s3_uri = upload_to_s3(tmp_path)
        publish_sse_log(channel, f"[INFO] S3 업로드 완료 — {s3_uri}")

        # 4. dataset_version 생성
        dataset_version = datetime.now(timezone.utc).strftime("v%Y-%m-%d-%H%M%S")
        publish_sse_log(channel, f"[INFO] 데이터 수집 완료 — dataset_version: {dataset_version}")

        # 5. 완료 이벤트 발행
        publish_training_event(
            channel,
            status="COMPLETED",
            dataset_version=dataset_version
        )

        logger.info("===== 데이터 수집 배치 완료 =====")

    except Exception as e:
        logger.error(f"배치 실패: {e}", exc_info=True)
        publish_sse_log(channel, f"[ERROR] 데이터 수집 실패: {e}")
        publish_training_event(channel, status="FAILED", error_message=str(e))
        raise

    finally:
        try:
            connection.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)  # 성공
    except Exception:
        sys.exit(1)  # 실패