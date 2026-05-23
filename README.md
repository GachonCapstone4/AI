# 업무 이메일 자동화 AI 서버

업무 이메일을 자동 분류하고, 요약과 일정 후보 추출을 지원하는 **AI 기반 이메일 자동화 서버**입니다.

단순 GPT API 연결이 아니라, `SBERT + Logistic Regression` 분류기로 이메일의 `Domain / Intent`를 예측하고 LLM은 요약과 일정 추출에만 사용합니다. FastAPI inference API, RabbitMQ consumer, SageMaker training container, Kubernetes dataset batch, S3 model artifact, Prometheus metrics까지 포함해 운영 흐름을 고려했습니다.

![README Hero Diagram](docs/README%20Hero%20Diagram.png)

## Live Service

- Production URL: https://capstone.studylink.click/

실제 운영 중인 업무 이메일 AI 자동화 서비스입니다.

## 핵심 기능

- 이메일 제목/본문 기반 `Domain / Intent` 자동 분류
- `SBERT -> Domain Logistic Regression -> Domain별 Intent Logistic Regression` 계층형 분류
- LLM 기반 이메일 요약 및 일정 정보 추출
- FastAPI 동기 inference API와 RabbitMQ 비동기 consumer 제공
- S3 model artifact와 `latest.json` 기반 모델 버전 관리
- `preload -> validate -> switch` 기반 모델 교체
- SageMaker training container, Kubernetes dataset batch, Prometheus metrics 구성

## 기술 스택

| 영역 | 기술 |
|---|---|
| API 서버 | FastAPI, Uvicorn, Pydantic |
| 모델 | SentenceTransformers, SBERT, Scikit-learn LogisticRegression |
| LLM 연동 | OpenAI-compatible chat client |
| 비동기 처리 | RabbitMQ, pika |
| MLOps | SageMaker Training Job, S3, Kubernetes Job |
| 모니터링 | Prometheus metrics |
| 테스트 | pytest, FastAPI TestClient |
| 실행 환경 | Docker, Python 3.11 |

## 문제 정의

업무 이메일은 분류, 요약, 일정 확인처럼 반복되는 처리가 많습니다. 모든 판단을 LLM에 맡기면 비용, 응답 일관성, latency, 운영 관리 측면에서 부담이 생깁니다.

이 프로젝트는 분류는 전통 ML 모델이 담당하고, LLM은 후처리에 집중하도록 역할을 분리했습니다. 그 위에 모델 재학습, artifact 관리, 모델 교체, 모니터링 흐름을 붙여 실제 AI 서버 형태로 구성했습니다.

## 모델 추론 파이프라인

![AI 추론 파이프라인](docs/AI%20추론%20파이프라인.png)

LLM은 분류기를 대체하지 않습니다. 분류는 `SBERT -> Domain Logistic Regression -> Domain별 Intent Logistic Regression` 순서로 수행하고, LLM은 요약과 일정 표현 추출에 사용합니다.

## 모델 구조

![계층형 모델 구조](docs/계층형%20모델%20구조.png)

핵심은 `Domain -> Intent` hierarchical classification 구조입니다. 먼저 상위 업무 영역을 좁힌 뒤, 해당 Domain의 Intent classifier로 세부 의도를 예측합니다.

| 구성 요소 | 사용 기술 | 역할 |
|---|---|---|
| Text Embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 이메일 텍스트를 의미 벡터로 변환 |
| SBERT Fine-tuning | `ContrastiveLoss` | 같은 intent는 positive, 같은 domain의 다른 intent는 hard negative로 학습 |
| Domain Classifier | `LogisticRegression` | 상위 업무 영역 분류 |
| Intent Classifier | `dict[str, LogisticRegression]` | Domain별 세부 intent 분류 |
| LLM Processor | OpenAI-compatible chat client | 요약 및 일정 표현 추출 |

## 운영 배포 구조

![AI 모델 무중단 배포 및 검증 흐름도](docs/AI%20모델%20무중단%20배포%20및%20검증%20흐름도.png)

새 모델은 바로 active model로 교체하지 않습니다. 먼저 staging 영역에 로드하고, 샘플 추론과 `label_mapping.json` 검증을 통과한 경우에만 current model로 전환합니다.

| 단계 | Endpoint | 동작 |
|---|---|---|
| preload | `POST /deployment/preload` | 요청한 `modelVersion` 또는 `latest.json` 기준 모델을 S3에서 받아 staging에 로드 |
| validate | `POST /deployment/validate` | staging 모델 샘플 추론 및 label mapping 검증 |
| switch | `POST /deployment/switch` | 검증된 staging 모델을 active model로 전환 |

<details>
<summary>preload -> validate -> switch sequence 보기</summary>

```mermaid
sequenceDiagram
    participant Admin as Admin/Backend
    participant AI as AI FastAPI Server
    participant S3 as S3 Model Artifact
    participant Current as Current Model
    participant Staging as Staging Model

    Admin->>AI: POST /deployment/preload
    AI->>S3: models/{version}/ 다운로드
    AI->>Staging: staging_bundle 로드

    Admin->>AI: POST /deployment/validate
    AI->>Staging: 샘플 추론
    AI->>Staging: label_mapping 검증

    Admin->>AI: POST /deployment/switch
    AI->>Current: 검증된 staging 모델을 current로 전환
```

</details>

## AI 운영 및 MLOps 아키텍처

![AI 운영 및 MLOps 아키텍처](docs/AI%20운영%20및%20MLOps%20아키텍처%20다이어그램.png)

재수집, 재학습, 재배포를 분리해 운영합니다. Dataset batch는 DB에서 학습 데이터를 추출해 S3 dataset을 갱신하고, training container는 SageMaker 환경에서 모델 artifact를 생성한 뒤 S3에 업로드합니다.

## 모델 학습 흐름

학습 입력 dataset은 `email_text`, `domain`, `intent` 컬럼을 필수로 사용합니다. Dataset batch에서는 `subject + body`로 `email_text`를 다시 생성해 학습 입력을 일관되게 맞춥니다.

<details>
<summary>모델 학습 흐름 보기</summary>

```mermaid
flowchart TD
    A["dataset_new.csv<br/>email_text, domain, intent"] --> B["Contrastive pair 생성"]
    B --> C["SBERT fine-tuning"]
    C --> D["email_text embedding 생성"]
    D --> E["Domain Logistic Regression 학습"]
    D --> F["Domain별 Intent Logistic Regression 학습"]
    E --> G["domain_model.pkl"]
    F --> H["intent_model.pkl"]
    G --> I["metrics.json / config.json / label_mapping.json"]
    H --> I
```

</details>

## 시스템 설계 포인트

| 구현한 내용 | 설명 |
|---|---|
| 분류와 LLM 역할 분리 | 분류는 SBERT + Logistic Regression이 담당하고, LLM은 요약/일정 추출에 사용합니다. |
| 계층형 분류 구조 | Domain을 먼저 예측한 뒤 해당 Domain의 Intent classifier로 세부 의도를 분류합니다. |
| 모델 교체 안정성 | 새 모델을 바로 덮어쓰지 않고 `preload -> validate -> switch` 단계를 둡니다. |
| 비동기 메시지 처리 | classify/deployment consumer로 API 요청 흐름과 긴 작업을 분리합니다. |
| 학습 산출물 표준화 | SageMaker 학습 결과를 `sbert`, `domain_model.pkl`, `intent_model.pkl`, `label_mapping.json`, `metrics.json`, `config.json` 단위로 관리합니다. |
| 운영 지표 노출 | `/metrics`에서 request count, latency, confidence, error, active model 정보를 Prometheus 형식으로 제공합니다. |

## 성능 및 규모

| 항목 | 값 | 기준 |
|---|---:|---|
| 학습 데이터 샘플 수 | 1,510 | `data/dataset_new.csv` |
| Domain 수 | 7 | `data/dataset_new.csv` |
| Intent 수 | 30 | `data/dataset_new.csv` |
| SBERT embedding dimension | 384 | `paraphrase-multilingual-MiniLM-L12-v2`, `src/train_sbert.py` 주석 |
| confidence threshold | 0.4 | `src/config.py` |

## 담당 역할

- FastAPI 기반 AI inference server 구현
- SBERT 기반 이메일 분류 pipeline 구성
- Domain / Intent 2단계 Logistic Regression 학습 및 추론 구조 구현
- LLM 기반 요약 및 일정 추출 흐름 연동
- RabbitMQ classify/deployment consumer와 publish payload 계약 구현
- SageMaker training container entrypoint 구성
- S3 model artifact 업로드, `latest.json` 갱신, model cache 로딩 구조 구현
- `preload -> validate -> switch` 기반 모델 교체 구조 구현
- Prometheus inference metric 및 `/metrics` endpoint 구현
- dataset batch의 DB 추출, CSV 병합/dedup, S3 업로드, 상태 이벤트 발행 흐름 구현
- API, 메시지 계약, 모델 로더, 배포, 학습 컨테이너, dataset batch 테스트 작성

<details>
<summary>주요 API 보기</summary>

| Method | Endpoint | 역할 |
|---|---|---|
| `GET` | `/health` | 서버 상태 확인 |
| `POST` | `/classify` | 이메일 분류, 요약, 일정 추출, embedding 반환 |
| `POST` | `/summarize` | 요약 및 일정 추출 보조 API |
| `POST` | `/deployment/preload` | 새 모델 staging 로드 |
| `POST` | `/deployment/validate` | staging 모델 검증 |
| `POST` | `/deployment/switch` | active model 전환 |
| `GET` | `/metrics` | Prometheus metric 노출 |

### `/classify` 요청

```json
{
  "outbox_id": 1,
  "email_id": 1,
  "sender_email": "sender@example.com",
  "sender_name": "홍길동",
  "subject": "세금계산서 발행 요청",
  "body_clean": "지난달 납품 건에 대한 세금계산서 발행 부탁드립니다.",
  "received_at": "2026-04-06T10:00:00"
}
```

### `/classify` 응답 예시

```json
{
  "outbox_id": 1,
  "email_id": 1,
  "classification": {
    "domain": "Finance",
    "intent": "Invoice Request"
  },
  "confidence_score": 0.91,
  "summary": "세금계산서 발행 요청 건입니다.",
  "schedule_info": {
    "date": "2026-04-14",
    "time": "14:00",
    "location": "회의실 A"
  },
  "email_embedding": [0.0123, -0.0456],
  "meta": null
}
```

`email_embedding`은 실제 응답에서 SBERT embedding 전체 길이의 float list로 반환됩니다. HTTP `/classify` 응답의 `meta`는 기본적으로 `null`입니다.

</details>

## 실제 서비스 예시

![실행 서비스 화면](docs/실행%20서비스%20화면.png)

실제 서비스 화면에서는 이메일별 분류 결과를 확인할 수 있습니다.

- Domain / Intent 예측 결과
- 이메일 요약
- 일정 추출 결과
- 분류 confidence score

## 모니터링

`GET /metrics`는 운영 중인 inference 상태를 Prometheus 형식으로 노출합니다. 요청 수, inference latency, error count, active model version을 확인할 수 있어 모델 교체 이후에도 서버 상태를 추적할 수 있습니다.

<나중에 추가할 Grafana 대시보드 이미지>

| Metric | 설명 |
|---|---|
| `ai_classify_requests_total` | inference request count |
| `ai_classify_latency_seconds` | inference latency |
| `ai_classify_confidence_score` | 최종 confidence score 분포 |
| `ai_schedule_detected_total` | 일정 정보 감지 횟수 |
| `ai_classify_errors_total` | error monitoring |
| `ai_active_model_info` | active model version 표시 |

## 테스트 전략

모델 추론뿐 아니라 배포, 메시지 계약, MLOps batch까지 테스트 대상으로 포함했습니다.

| 범위 | 테스트 파일 |
|---|---|
| API / schema | `tests/test_classify.py`, `tests/test_deployment_router.py` |
| 메시지 계약 | `tests/test_message_contracts.py`, `tests/test_retry_policy.py` |
| 모델 로딩/교체 | `tests/test_model_loader.py`, `tests/test_model_manager.py` |
| MLOps | `tests/test_training_container_entrypoint.py`, `tests/test_training_events.py`, `tests/test_dataset_batch.py` |
| 모델 학습 보조 | `tests/test_train_sbert_artifact.py`, `tests/test_training_cv_guards.py` |
| 운영 지표/일정 파싱 | `tests/test_metrics_endpoint.py`, `tests/test_schedule_parser.py` |

## 디렉터리 구조

```text
api/          FastAPI router & schema
src/          inference / training / metrics
messaging/    RabbitMQ consumer & publisher
batch/        dataset batch
tests/        API / MLOps / model tests
```

## 주요 설계 결정

| 고민한 지점 | 선택한 방식 | 이유 |
|---|---|---|
| 텍스트 표현 방식 | SBERT 기반 이메일 텍스트 임베딩 | 제목/본문의 의미적 유사성을 반영하기 위해 |
| 분류 모델 | SBERT embedding + Logistic Regression | 데이터셋 규모가 크지 않은 상황에서 학습/추론이 빠르고 baseline으로 안정적이기 때문 |
| 세부 의도 분류 | Domain -> Intent 계층형 구조 | 업무 영역을 먼저 좁혀 세부 의도 오분류를 줄이기 위해 |
| LLM 사용 범위 | 분류는 ML 모델, LLM은 요약/일정 추출 | 비용, 일관성, latency를 관리하기 위해 |
| 모델 교체 | `preload -> validate -> switch` | 검증 실패 시 기존 모델을 유지하기 위해 |
| 긴 작업 처리 | RabbitMQ 메시지 기반 처리 | 학습/배포 작업을 API 요청 흐름과 분리하기 위해 |
| artifact 관리 | S3에 version 단위 저장 | 모델 파일, label mapping, metrics, config를 배포 단위로 관리하기 위해 |

## 트러블슈팅 / 개선 경험

| 항목 | 처리 방식 |
|---|---|
| 모델 artifact 파일명 혼재 | 로컬 legacy artifact와 SageMaker 표준 artifact 로딩 경로 분리 |
| `latest.json` 버전 충돌 | `model_version`, `modelVersion` 값 충돌 시 오류 처리 |
| 불완전한 artifact | SBERT core file, classifier, label mapping, config, metrics 필수 파일 검증 |
| 모델 교체 중 검증 실패 | validate 실패 시 switch가 실행되지 않도록 staging validation flag 사용 |
| LLM 영구 오류 | 분류 결과는 유지하고 summary fallback 적용 |
| dataset overwrite 위험 | 기존 dataset을 S3에서 받아 병합/dedup 후 업로드 |

## 향후 개선 사항

- held-out validation set 기반 `domain_accuracy`, `intent_f1` 평가로 전환
- confidence threshold 기반 human-in-the-loop 검수 흐름 추가
- active learning 기반 dataset 개선
- LLM 요약/일정 추출 품질 평가 자동화
- Grafana dashboard와 README 시각 자료 추가
- 모델 A/B deployment 또는 canary 전환 구조 검토

## 실행 방법

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
pytest
```

Docker 실행:

```bash
docker build -t business-email-ai-server .
docker run --env-file .env -p 8080:8080 business-email-ai-server
```

학습 컨테이너 entrypoint:

```bash
python -m src.mlops.training_container_entrypoint --dry-run
```

로컬 실행에는 `.env`, RabbitMQ 접속 정보, LLM provider 설정, 모델 artifact가 필요합니다. 환경변수 예시는 `.env.example`을 참고합니다.
