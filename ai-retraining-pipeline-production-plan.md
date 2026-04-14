---

# 📦 AI Retraining Pipeline 운영 전환 액션 플랜 (v3 - School LLM + S3 Model)

## 🎯 목표

AI 서버를 **S3 기반 데이터 + S3 모델 + 학교 LLM API + RabbitMQ 이벤트 기반 운영 구조**로 전환한다.

---

# Phase 0. 기준 확정 (Design Freeze)

## 목적

운영 구조의 핵심 계약 및 외부 의존성을 확정한다.

## 작업

* [ ] retraining 데이터 source를 **S3 dataset artifact**로 확정

* [ ] 모델 로딩을 **S3 model artifact 기반**으로 전환

* [ ] LLM 호출을 **학교 LLM API로 통일**

* [ ] Backend ↔ AI 계약 고정

  * `jobId`
  * `datasetVersion`
  * `datasetUri`

* [ ] LLM 정책 확정

  * [ ] OpenAI 직접 호출 제거
  * [ ] Claude 제거
  * [ ] 학교 LLM(OpenAI-compatible API) 사용

* [ ] 토큰 사용 정책 정의 (100,000/day)

## 완료 기준

* [ ] 데이터 / 모델 / LLM 모두 외부 시스템 기반으로 고정됨

---

# Phase 1. 환경변수 외부 주입 구조 정리

## 목적

모든 설정을 코드에서 분리하고 운영 환경에서 주입 가능하도록 변경

## 작업

* [ ] 중앙 Settings 모듈 도입
* [ ] `.env.example` 재작성
* [ ] MOCK_MODE 관련 코드 완전 제거
* [ ] OpenAI key 기반 구조 제거
* [ ] Claude 코드 제거
* [ ] 모든 LLM 호출을 base_url 기반 구조로 변경
* [ ] 모델 경로 하드코딩 제거 (S3 기반으로 변경)
* [ ] FastAPI host/port env로 이동

---

## 필수 환경변수

```env
APP_ENV=prod
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# RabbitMQ
RABBITMQ_URL=...

# AWS
AWS_REGION=ap-northeast-2
S3_DATASET_BUCKET=...
S3_DATASET_PREFIX=datasets
S3_MODEL_BUCKET=...
S3_MODEL_PREFIX=models

# Model (S3 기반)
MODEL_SOURCE=s3
ACTIVE_MODEL_VERSION=2026-04-14-001
MODEL_LOCAL_CACHE_DIR=/tmp/model-cache

# School LLM (핵심)
LLM_BASE_URL=http://cellm.gachon.ac.kr:8000/v1
LLM_API_KEY=sk-vllm-xxxxxxxx
LLM_MODEL=chat-model

# Training
TRAINING_SAFE_MODE=0
```

---

## 완료 기준

* [ ] 코드 수정 없이 환경변수만으로 실행 환경 변경 가능
* [ ] 필수 env 누락 시 서버 실행 실패

---

# Phase 2. Dataset 구조 표준화 (S3 기반)

## 목적

운영용 dataset을 표준화하고 S3에 저장

## 작업

* [ ] JSON array → JSONL(.jsonl.gz) 구조로 변경
* [ ] chunk 기반 dataset 생성
* [ ] schema 정의

```json
{
  "emailId": 123,
  "threadId": "t1",
  "subject": "...",
  "body": "...",
  "emailText": "...",
  "domain": "HR",
  "intent": "Interview",
  "labelSource": "admin",
  "labeledAt": "...",
  "updatedAt": "..."
}
```

* [ ] manifest.json 생성

```json
{
  "jobId": "job_001",
  "datasetVersion": "v1",
  "schemaVersion": "1.0",
  "snapshotTime": "...",
  "partCount": 3,
  "totalRows": 1200,
  "parts": [
    {
      "s3Key": "data-00001.jsonl.gz",
      "rowCount": 400,
      "checksum": "..."
    }
  ]
}
```

---

## S3 구조

```
s3://bucket/datasets/{datasetVersion}/manifest.json
s3://bucket/datasets/{datasetVersion}/data-00001.jsonl.gz
```

---

## 완료 기준

* [ ] datasetVersion ↔ S3 데이터 연결됨
* [ ] AI가 manifest 기반으로 dataset 검증 가능

---

# Phase 3. Backend Retrain Job 구조

## 목적

retraining lifecycle 관리

## 작업

* [ ] `retrain_job` 테이블 생성

### 상태 머신

```
REQUESTED
EXPORTING_DATASET
DATASET_READY
TRAINING_QUEUED
TRAINING_STARTED
TRAINING_RUNNING
MODEL_UPLOADING
SUCCEEDED
FAILED
```

* [ ] job 필드 정의

  * job_id
  * dataset_version
  * dataset_s3_uri
  * model_version
  * status
  * error_message
  * timestamps

* [ ] 중복 retrain 방지

---

## 완료 기준

* [ ] jobId 기준 상태 추적 가능

---

# Phase 4. Dataset Export Pipeline (Backend)

## 목적

DB → S3 dataset 변환

## 작업

* [ ] email + outbox join
* [ ] streaming 조회
* [ ] JSONL 생성
* [ ] chunk 분할
* [ ] gzip 압축
* [ ] checksum 생성
* [ ] manifest 생성
* [ ] S3 업로드

---

## 완료 기준

* [ ] 대용량 데이터 안정 처리 가능

---

# Phase 5. Training Event Contract

## 이벤트 종류

* training.requested
* training.started
* training.progress
* training.completed
* training.failed

---

## 공통 필드

```json
{
  "eventId": "...",
  "eventType": "...",
  "occurredAt": "...",
  "jobId": "...",
  "schemaVersion": "1.0"
}
```

---

## 요청 이벤트

```json
{
  "jobId": "job_001",
  "datasetUri": "s3://...",
  "datasetVersion": "v1"
}
```

---

## 완료 기준

* [ ] MQ 기반 retraining 실행 가능

---

# Phase 6. AI Server Training Pipeline

## 목적

S3 dataset 기반 학습 수행

## 작업

* [ ] datasetUri 기반 S3 다운로드
* [ ] manifest 검증
* [ ] JSONL streaming 처리
* [ ] 학습 수행
* [ ] progress 이벤트 발행
* [ ] 완료 이벤트 발행

---

## 완료 기준

* [ ] 로컬 파일 없이 학습 가능

---

# Phase 7. Model Artifact (S3 기반)

## 목적

모델을 외부화하여 운영에서 관리

## 작업

* [ ] 학습 결과 S3 업로드

```
s3://bucket/models/{modelVersion}/
```

* [ ] metadata 저장

  * metrics.json
  * config.json
  * label mapping

* [ ] inference 시 ACTIVE_MODEL_VERSION 기반 로딩

---

## 완료 기준

* [ ] 코드 수정 없이 모델 교체 가능

---

# Phase 8. LLM Integration (School API)

## 목적

모든 LLM 호출을 학교 API로 통일

## 작업

* [ ] OpenAI client를 base_url 기반으로 사용
* [ ] 모든 LLM 호출을 학교 endpoint로 전환
* [ ] Authorization header 적용

---

## 호출 방식

```bash
curl http://cellm.gachon.ac.kr:8000/v1/chat/completions \
-H "Authorization: Bearer {API_KEY}" \
-H "Content-Type: application/json"
```

---

## Python 예시

```python
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
```

---

## Token 정책

* [ ] 1일 100,000 토큰 제한 고려
* [ ] 요청당 max token 제한
* [ ] 긴 입력 truncation

---

## 완료 기준

* [ ] 모든 LLM 호출이 학교 API 사용

---

# Phase 9. 장애 대응 / 안정성

## 작업

* [ ] retry 정책
* [ ] DLQ 설정
* [ ] idempotency 처리
* [ ] MQ 실패 처리

---

# Phase 10. 모니터링 / 운영

## 작업

* [ ] jobId 기반 logging
* [ ] 실패율 추적
* [ ] MQ 상태 모니터링
* [ ] admin 조회 API

---

# 🚀 최종 완료 기준

* [ ] 로컬 dataset 사용 없음
* [ ] 로컬 모델 사용 없음
* [ ] S3 기반 dataset + model 사용
* [ ] 학교 LLM API 사용
* [ ] retraining job 추적 가능
* [ ] token 제한 내 안정 운영 가능

---