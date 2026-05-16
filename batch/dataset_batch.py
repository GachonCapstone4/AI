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
    ADMIN_USER_ID
"""

import os
import sys
import csv
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import mysql.connector
import pika

# ============================================================
# 로깅 설정
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ============================================================
# 환경변수
# ============================================================
DB_HOST     = os.environ.get("DB_HOST")
DB_PORT     = os.environ.get("DB_PORT")
DB_USER     = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME     = os.environ.get("DB_NAME")

AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.environ.get("AWS_REGION", "ap-northeast-2")
S3_BUCKET             = os.environ.get("S3_BUCKET", "capstone-gachon")
S3_DATASET_KEY        = os.environ.get("S3_DATASET_KEY", "dataset/dataset_new.csv")
MIN_DATASET_SIZE      = int(os.environ.get("MIN_DATASET_SIZE", "100"))

RABBITMQ_HOST     = os.environ.get("RABBITMQ_HOST")
RABBITMQ_PORT     = os.environ.get("RABBITMQ_PORT")
RABBITMQ_USERNAME = os.environ.get("RABBITMQ_USERNAME")
RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD")

JOB_ID        = datetime.now(timezone.utc).strftime("collecting-job-%Y%m%d-%H%M%S")
ADMIN_USER_ID = os.environ.get("ADMIN_USER_ID")

REQUIRED_ENV_VARS = (
    "ADMIN_USER_ID",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "RABBITMQ_HOST",
    "RABBITMQ_PORT",
    "RABBITMQ_USERNAME",
    "RABBITMQ_PASSWORD",
    "DB_HOST",
    "DB_PORT",
    "DB_USER",
    "DB_PASSWORD",
    "DB_NAME",
)

RABBITMQ_REQUIRED_ENV_VARS = (
    "ADMIN_USER_ID",
    "RABBITMQ_HOST",
    "RABBITMQ_PORT",
    "RABBITMQ_USERNAME",
    "RABBITMQ_PASSWORD",
)

# RabbitMQ 상수
EXCHANGE_SSE_FANOUT   = "x.sse.fanout"
EXCHANGE_TRAINING_DIRECT = "x.ai2app.direct"
QUEUE_TRAINING_RESULT = "q.2app.training"
ROUTING_KEY_TRAINING = "app.training"
DEFAULT_SSE_TYPE      = "ai-training-updated"
DATASET_SSE_TYPE      = "ai-collecting-updated"
CSV_FIELDNAMES = ["emailId", "threadId", "from", "subject", "body", "email_text", "domain", "intent"]


# ============================================================
# 환경변수 검증
# ============================================================
def validate_required_env(names=REQUIRED_ENV_VARS):
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"필수 환경변수가 누락되었습니다: {', '.join(missing)}")


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


# ============================================================
# RabbitMQ 연결
# ============================================================
def connect_rabbitmq():
    credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=int(RABBITMQ_PORT),
        credentials=credentials,
        heartbeat=60
    )
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    return connection, channel


# ============================================================
# SSE 로그 발행
# ============================================================
def publish_sse_log(channel, message: str, sse_type: str = DEFAULT_SSE_TYPE):
    payload = {
        "user_id": _parse_sse_user_id(ADMIN_USER_ID, "ADMIN_USER_ID"),
        "sse_type": sse_type,
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
def publish_training_event(
    channel,
    status: str,
    error_message: str = None,
    dataset_version: str = None,
    dataset_s3_uri: str = None,
):
    payload = {
        "job_id": JOB_ID,
        "job_type": "dataset",
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat()
    }
    if dataset_version:
        payload["dataset_version"] = dataset_version
    if dataset_s3_uri:
        payload["dataset_s3_uri"] = dataset_s3_uri
    if error_message:
        payload["error_message"] = error_message

    try:
        channel.exchange_declare(
            exchange=EXCHANGE_TRAINING_DIRECT,
            exchange_type="direct",
            durable=True,
        )
        channel.queue_declare(queue=QUEUE_TRAINING_RESULT, durable=True)
        channel.queue_bind(
            queue=QUEUE_TRAINING_RESULT,
            exchange=EXCHANGE_TRAINING_DIRECT,
            routing_key=ROUTING_KEY_TRAINING,
        )
        channel.basic_publish(
            exchange=EXCHANGE_TRAINING_DIRECT,
            routing_key=ROUTING_KEY_TRAINING,
            body=json.dumps(payload, ensure_ascii=False),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
            ),
            mandatory=True,
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
        port=int(DB_PORT),
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
            ORDER BY e.email_id ASC \
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
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            subject = row.get("subject") or ""
            body = row.get("body") or ""
            csv_row = {
                fieldname: row.get(fieldname)
                for fieldname in CSV_FIELDNAMES
                if fieldname != "email_text"
            }
            # Always regenerate email_text from subject/body to keep training input consistent.
            csv_row["email_text"] = f"{subject}\n{body}".strip()
            writer.writerow(csv_row)
    logger.info(f"CSV 파일 생성 완료: {filepath} ({len(rows)}행)")


def _normalize_csv_row(row: dict) -> dict:
    subject = row.get("subject") or ""
    body = row.get("body") or ""
    normalized = {fieldname: row.get(fieldname) for fieldname in CSV_FIELDNAMES if fieldname != "email_text"}
    normalized["email_text"] = f"{subject}\n{body}".strip()
    return normalized


def _dedup_key(row: dict) -> str:
    email_id = str(row.get("emailId") or "").strip()
    if email_id:
        return f"emailId:{email_id}"
    thread_id = str(row.get("threadId") or "").strip()
    if thread_id:
        return f"threadId:{thread_id}"
    subject = str(row.get("subject") or "").strip()
    body = str(row.get("body") or "").strip()
    sender = str(row.get("from") or "").strip()
    return f"content:{sender}|{subject}|{body}"


def _read_csv_rows(filepath: str) -> list[dict]:
    path = Path(filepath)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [
            _normalize_csv_row(row)
            for row in csv.DictReader(f)
        ]


def _write_csv_rows(rows: list[dict], filepath: str) -> None:
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_csv_row(row))


def _deduplicate_rows(rows: list[dict]) -> tuple[list[dict], int]:
    latest_by_key = {}
    duplicate_count = 0
    for row in rows:
        key = _dedup_key(row)
        if key in latest_by_key:
            duplicate_count += 1
        # Latest row wins because later sources are appended later.
        latest_by_key[key] = _normalize_csv_row(row)
    return list(latest_by_key.values()), duplicate_count


def merge_dataset_rows(existing_rows: list[dict], new_rows: list[dict]) -> dict:
    normalized_new_rows = [_normalize_csv_row(row) for row in new_rows]
    merged_rows, duplicate_count = _deduplicate_rows([*existing_rows, *normalized_new_rows])

    return {
        "merged_rows": merged_rows,
        "existing_rows": len(existing_rows),
        "new_rows": len(normalized_new_rows),
        "duplicate_count": duplicate_count,
    }


def _distribution(rows: list[dict], column: str) -> dict:
    counts = {}
    for row in rows:
        value = str(row.get(column) or "").strip()
        if not value:
            value = "<missing>"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _missing_count(rows: list[dict], column: str) -> int:
    return sum(1 for row in rows if not str(row.get(column) or "").strip())


def validate_dataset(rows: list[dict], min_samples: int = MIN_DATASET_SIZE) -> None:
    domain_distribution = _distribution(rows, "domain")
    intent_distribution = _distribution(rows, "intent")
    total_samples = len(rows)
    logger.info(f"dataset total samples: {total_samples}")
    logger.info(f"domain missing count: {_missing_count(rows, 'domain')}")
    logger.info(f"intent missing count: {_missing_count(rows, 'intent')}")
    logger.info(f"domain distribution: {domain_distribution}")
    logger.info(f"intent distribution: {intent_distribution}")

    if total_samples < min_samples:
        raise ValueError(
            f"Dataset is too small: total_samples={total_samples}, minimum_required={min_samples}"
        )
    non_empty_domains = [domain for domain in domain_distribution if domain != "<missing>"]
    if len(non_empty_domains) < 2:
        raise ValueError(
            "Dataset must contain at least 2 domain classes; "
            f"domain_distribution={domain_distribution}"
        )


# ============================================================
# S3 업로드
# ============================================================
def _s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )


def download_from_s3_if_exists(key: str, filepath: str) -> bool:
    s3_client = _s3_client()
    try:
        s3_client.download_file(S3_BUCKET, key, filepath)
        logger.info(f"S3 다운로드 완료: s3://{S3_BUCKET}/{key} -> {filepath}")
        return True
    except Exception as exc:
        logger.warning(f"S3 다운로드 스킵: s3://{S3_BUCKET}/{key} ({exc})")
        return False


def upload_to_s3(filepath: str, key: str = None):
    key = key or S3_DATASET_KEY
    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    s3_client.upload_file(filepath, S3_BUCKET, key)
    s3_uri = f"s3://{S3_BUCKET}/{key}"
    logger.info(f"S3 업로드 완료: {s3_uri}")
    return s3_uri


# ============================================================
# 메인
# ============================================================
def main():
    logger.info(f"===== 데이터 수집 배치 시작 — job_id={JOB_ID} =====")

    validate_required_env(RABBITMQ_REQUIRED_ENV_VARS)
    connection, channel = connect_rabbitmq()

    try:
        validate_required_env()

        # 1. DB 데이터 추출
        publish_sse_log(channel, "[INFO] DB 데이터 추출 시작", sse_type=DATASET_SSE_TYPE)
        rows = fetch_training_data()
        publish_sse_log(channel, f"[INFO] {len(rows)}건 추출 완료", sse_type=DATASET_SSE_TYPE)

        if len(rows) == 0:
            raise ValueError("추출된 데이터가 없습니다. domain/intent 분류된 이메일을 확인해주세요.")

        # 2. 기존 dataset_new.csv 다운로드 및 새 수집분 merge
        publish_sse_log(channel, "[INFO] 기존 dataset_new.csv 확인 중", sse_type=DATASET_SSE_TYPE)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            existing_path = tmp_root / "dataset_new_existing.csv"
            merged_path = tmp_root / "dataset_new.csv"

            download_from_s3_if_exists(S3_DATASET_KEY, str(existing_path))

            existing_rows = _read_csv_rows(str(existing_path))
            logger.info(f"existing dataset_new.csv rows: {len(existing_rows)}")

            merge_result = merge_dataset_rows(existing_rows, rows)
            logger.info(
                "dataset merge result: "
                f"existing_rows={merge_result['existing_rows']}, "
                f"new_rows={merge_result['new_rows']}, "
                f"duplicates_overwritten={merge_result['duplicate_count']}, "
                f"merged_rows={len(merge_result['merged_rows'])}"
            )

            validate_dataset(merge_result["merged_rows"])

            _write_csv_rows(merge_result["merged_rows"], str(merged_path))
            publish_sse_log(
                channel,
                f"[INFO] dataset_new.csv merge 완료 — {len(merge_result['merged_rows'])}건",
                sse_type=DATASET_SSE_TYPE,
            )

            # 3. validation 성공 후 동일 key로 업로드한다.
            publish_sse_log(channel, "[INFO] S3 업로드 시작", sse_type=DATASET_SSE_TYPE)
            dataset_s3_uri = upload_to_s3(str(merged_path), S3_DATASET_KEY)
            publish_sse_log(
                channel,
                f"[INFO] S3 업로드 완료 — {dataset_s3_uri}",
                sse_type=DATASET_SSE_TYPE,
            )

        # 4. dataset_version 생성
        dataset_version = datetime.now(timezone.utc).strftime("v%Y-%m-%d-%H%M%S")
        publish_sse_log(
            channel,
            f"[INFO] 데이터 수집 완료 — dataset_version: {dataset_version}, dataset={dataset_s3_uri}",
            sse_type=DATASET_SSE_TYPE,
        )

        # 5. 완료 이벤트 발행
        publish_training_event(
            channel,
            status="COMPLETED",
            dataset_version=dataset_version,
            dataset_s3_uri=dataset_s3_uri,
        )

        logger.info("===== 데이터 수집 배치 완료 =====")

    except Exception as e:
        logger.error(f"배치 실패: {e}", exc_info=True)
        publish_sse_log(channel, f"[ERROR] 데이터 수집 실패: {e}", sse_type=DATASET_SSE_TYPE)
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
