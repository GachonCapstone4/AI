---

# 추천 구현 순서

## 1. 환경변수/설정 구조부터 먼저 정리

가장 먼저 해야 합니다.
이걸 안 하면 이후 작업이 전부 다시 깨집니다.

먼저 할 것:

* `Settings` 모듈 만들기
* `RABBITMQ_URL`, `S3_*`, `LLM_BASE_URL`, `LLM_API_KEY`, `ACTIVE_MODEL_VERSION` 정리
* `MOCK_MODE` 제거
* Claude/OpenAI 직접 키 구조 제거
* startup validation 추가

왜 먼저냐면:

* dataset S3 업로드도 설정이 필요하고
* model S3 로딩도 설정이 필요하고
* 학교 LLM 연동도 설정이 필요해서
* 제일 아래 기반이 됩니다

**산출물**

* `config/settings.py`
* `.env.example`
* startup validation

---

## 2. 학교 LLM 연동 코드 교체

그 다음은 LLM 호출부를 바꾸는 게 좋습니다.

먼저 할 것:

* `gpt_service.py` 같은 호출부를 학교 LLM 기준으로 통일
* `base_url=http://cellm.gachon.ac.kr:8000/v1` 구조 반영
* `LLM_API_KEY`, `LLM_MODEL` 사용
* 요청 길이 제한 / max token 제한 추가

이걸 초반에 하는 이유:

* 지금 mock를 더 안 쓴다고 했고
* `/summarize`, `/draft` 쪽 실제 동작 확인이 빨라야
* 전체 운영 전환 중 가장 눈에 보이는 기능이 먼저 살아납니다

**산출물**

* 공통 LLM client
* summarize/draft service 수정
* 토큰 사용 제한 로직

---

## 3. Dataset 계약 확정

이제 retraining 쪽으로 들어갑니다.

먼저 확정할 것:

* JSONL.gz 포맷
* manifest 구조
* S3 key 규칙
* `datasetVersion`, `datasetUri` 규칙
* row schema

이 단계는 “코드 작성”보다 “계약 고정”이 중요합니다.

예:

* `s3://bucket/datasets/{datasetVersion}/manifest.json`
* `s3://bucket/datasets/{datasetVersion}/data-00001.jsonl.gz`

**산출물**

* dataset schema 문서
* manifest schema 문서
* S3 naming rule

---

## 4. Backend retrain_job 테이블과 상태머신 구현

그 다음은 Backend입니다.

먼저 할 것:

* `retrain_job` 테이블 생성
* 상태 정의
* `jobId` 발급
* retrain 버튼 누르면 job row부터 생성
* 중복 실행 방지

이걸 dataset export보다 먼저 하는 이유:

* export 실패/성공 상태를 기록할 곳이 먼저 있어야 하고
* 나중에 MQ 이벤트도 job 기준으로 추적해야 하기 때문입니다

추천 상태:

* `REQUESTED`
* `EXPORTING_DATASET`
* `DATASET_READY`
* `TRAINING_QUEUED`
* `TRAINING_STARTED`
* `TRAINING_RUNNING`
* `MODEL_UPLOADING`
* `SUCCEEDED`
* `FAILED`

**산출물**

* DB migration
* status transition 로직
* job 조회 API 초안

---

## 5. Backend dataset export → S3 업로드 구현

이제 실제 데이터 파이프라인을 만듭니다.

먼저 할 것:

* `email + outbox` join 쿼리 작성
* pagination/streaming 조회
* JSONL chunk writer
* gzip 압축
* manifest 생성
* S3 업로드
* 업로드 성공 시 `DATASET_READY`

여기서 중요한 건:

* 전체 데이터를 한 번에 메모리에 올리지 않기
* part 단위로 생성/업로드하기
* checksum 남기기

**산출물**

* export service
* S3 uploader
* dataset manifest writer

---

## 6. RabbitMQ training.requested 이벤트 계약 확정 및 발행

dataset이 준비되면 이제 AI에 넘겨야 합니다.

먼저 할 것:

* `training.requested` schema 확정
* Backend가 `DATASET_READY` 이후에만 이벤트 발행
* payload에 최소한 아래 포함

  * `jobId`
  * `datasetVersion`
  * `datasetUri`

이 단계부터는 서비스 간 연동입니다.

**산출물**

* exchange/queue/routing key 정의
* request publisher
* event schema 문서

---

## 7. AI training consumer 구현

이제 AI 서버가 retrain 이벤트를 받도록 만듭니다.

먼저 할 것:

* `training.requested` consume
* `jobId` 중복 실행 방지
* `datasetUri`에서 manifest 읽기
* part 다운로드 및 검증
* training 시작 이벤트 발행

이 단계에서 꼭 넣어야 할 것:

* schema validation
* checksum 검증
* 최소 row 수 검증
* invalid dataset이면 즉시 실패 이벤트 발행

**산출물**

* training consumer
* dataset downloader/validator
* started/failed event 발행

---

## 8. 학습 완료 후 모델 S3 업로드

학습이 끝나면 바로 모델 저장 구조를 붙입니다.

먼저 할 것:

* 모델 산출물 디렉토리 구조 정의
* S3 업로드
* `modelVersion` 생성 규칙
* `metrics.json`, `config.json`, `label_mapping.json` 저장

예:

* `s3://bucket/models/{modelVersion}/...`

**산출물**

* model artifact uploader
* metadata writer
* completion event payload에 `modelVersion` 포함

---

## 9. Backend completion/failure consumer 구현

AI가 끝냈다고 보내면 Backend가 마무리합니다.

먼저 할 것:

* `training.completed`
* `training.failed`
* 필요하면 `training.progress`

그리고:

* `jobId` 기준으로 상태 갱신
* 이미 완료된 job면 중복 이벤트 무시
* `modelVersion` 저장

**산출물**

* MQ consumer
* status update logic
* idempotency 처리

---

## 10. Inference 서버의 모델 로딩을 S3 기반으로 전환

이제 추론 쪽도 운영형으로 맞춥니다.

먼저 할 것:

* `ACTIVE_MODEL_VERSION` 읽기
* startup 시 S3에서 모델 다운로드
* local cache 저장
* cache miss 처리
* 필요하면 reload 전략 정의

이건 retraining 후 바로 붙는 게 맞습니다.
왜냐하면 모델이 S3에 저장되더라도 inference가 못 읽으면 운영 전환이 반쪽이기 때문입니다.

**산출물**

* model loader
* local cache manager
* startup preload

---

## 11. 마지막으로 안정화

맨 마지막은 운영 품질 작업입니다.

포함할 것:

* retry 정책
* DLQ
* structured logging
* queue depth 모니터링
* job status 조회 API
* timeout 정책
* 토큰 사용량 추적

---

# 아주 현실적인 작업 우선순위

진짜 실무적으로 잘게 쪼개면 이 순서가 제일 좋습니다.

## 1주차

1. `Settings` 정리
2. `.env.example` 작성
3. 학교 LLM 연동 교체
4. mock/Claude 제거

## 2주차

5. dataset schema + manifest 확정
6. `retrain_job` 테이블/상태머신 구현
7. Backend export service 구현

## 3주차

8. S3 업로드 붙이기
9. `training.requested` MQ 발행
10. AI consumer 구현

## 4주차

11. training 완료 후 model S3 업로드
12. Backend completion consumer 구현
13. inference S3 model loader 구현

## 5주차

14. retry/DLQ/idempotency
15. 로그/모니터링
16. 관리자 조회 API/운영 점검

---

# 지금 당장 “첫 작업” 하나만 고르라면

**1순위는 `Settings + .env.example + 학교 LLM 연동 교체` 입니다.**

이유는 간단합니다.

* 지금 mock를 더 이상 안 쓴다고 했고
* 학교 LLM을 실제로 붙여야 하고
* 이후 S3/MQ/model loader도 전부 설정 구조 위에서 움직이기 때문입니다

---