# AI 학습 서버 아키텍처 및 Job 처리 설계 (최종 기준)

---

## 1. 목적

본 문서는 AI 학습 서버의 재학습(MLOps) 구조를 정의한다.
특히 관리자 웹에서 발생하는 재수집/재학습/재배포 요청을 안정적으로 처리하기 위해
**Admin 서버 직접 Job 생성 아키텍처**를 기준으로 설계한다.

본 설계는 다음을 목표로 한다:

* 학습 작업을 비동기 Job 형태로 처리
* Backend와 AI 서버 간 책임 분리
* 운영 안정성을 고려한 모델 관리 구조 확보

---

## 2. 설계 원칙

### 2.1 역할 분리

* Admin REST API

  * Job 생성
  * 상태 조회
  * 모델 관리

* Kubernetes API

  * 데이터 재수집 Job 생성
  * dataset-batch ECR 이미지 실행

* SageMaker API

  * 재학습 Training Job 생성
  * training ECR 이미지 실행

* RabbitMQ/SSE

  * 실행 컨테이너의 상태 이벤트와 진행 로그 전달

---

### 2.2 비동기 처리 원칙

* 모든 학습 작업은 Job 단위로 처리한다
* REST 요청은 즉시 응답해야 한다
* 실제 학습은 SageMaker Training 컨테이너에서 수행한다

---

### 2.3 Job 생성 원칙

* Job 생성은 Admin 서버가 직접 수행한다.
* AI repo에는 Kubernetes/SageMaker Job 생성 코드를 두지 않는다.
* ECR에는 실행용 컨테이너 이미지만 배포한다.

| 작업 | Job 생성 주체 | 실행 이미지 |
| -- | -- | -- |
| 데이터 재수집 | Admin 서버 → Kubernetes API | dataset-batch |
| 재학습 | Admin 서버 → SageMaker `create_training_job` | training |
| 재배포 | Admin 서버 → AI FastAPI deployment API | AI FastAPI |

---

## 3. 전체 시스템 흐름

```text
관리자 웹
→ Backend
→ POST /api/admin/ai-training/*
→ Job 생성 및 DB 저장 (QUEUED)
→ 데이터 재수집: Kubernetes API로 dataset-batch Job 생성
→ 재학습: boto3로 SageMaker Training Job 생성
→ 재배포: AI FastAPI /deployment/preload → /validate → /switch
→ 실행 컨테이너/AI 서버가 상태 이벤트 및 SSE 로그 발행
→ Backend 상태 업데이트
→ 관리자 웹 조회/SSE 표시
```

---

## 4. REST API 정의

### 4.1 데이터셋 관리

* GET /api/admin/ai-training/datasets
  데이터셋 목록 조회

* POST /api/admin/ai-training/dataset-collections
  데이터 재수집 Job 생성
  Admin 서버가 Kubernetes API로 dataset-batch Job을 생성한다.

---

### 4.2 학습 관련 Job 생성

* POST /api/admin/ai-training/training-jobs

역할:

* Job 생성
* DB 저장 (status = queued)
* boto3 SageMaker `create_training_job` 호출
* training ECR 이미지 실행

---

### 4.3 Job 상태 조회

* GET /api/admin/ai-training/jobs/{job_id}

역할:

* Job 상태 확인
* 결과 조회
* UI 표시 데이터 제공

---

### 4.4 모델 관리

* GET /api/admin/ai-training/models
* GET /api/admin/ai-training/models/{model_id}
* PATCH /api/admin/ai-training/models/{model_id}

역할:

* 모델 목록 조회
* 모델 상세 조회
* 운영 모델 전환

---

## 5. 실행 이미지와 상태 이벤트

### 5.1 실행 이미지

| 이미지 | 실행 코드 | 역할 |
| ------ | -------- | ---- |
| dataset-batch | `batch/dataset_batch.py` | DB에서 학습 데이터 추출 후 S3 업로드 |
| training | `src.mlops.training_container_entrypoint` | dataset 다운로드, 학습, artifact 업로드, latest.json 갱신 |

---

### 5.2 상태 이벤트

실행 컨테이너는 RabbitMQ/SSE로 진행 상태를 발행할 수 있다.
Admin 서버는 이벤트를 수신해 DB Job 상태를 갱신한다.

---

### 5.3 완료 이벤트 메시지 스펙

```json
{
  "job_id": "job_001",
  "status": "completed",
  "model_version": "v2026_04_12_01",
  "finished_at": "timestamp",
  "metrics": {
    "intent_f1": 0.89,
    "domain_accuracy": 0.92
  },
  "error_message": null
}
```

---

## 6. Training 컨테이너 설계

Training 컨테이너는 다음을 수행한다:

1. SageMaker Training Job 환경변수 수신
2. S3 dataset 다운로드
3. 학습 코드 실행

   * train_sbert.py
   * train_domain.py
   * train_intent.py
4. 표준 모델 artifact 저장
5. S3 업로드 및 latest.json 갱신
6. 상태 이벤트/SSE 로그 발행

---

## 7. Backend 역할

* Job 생성 시 DB 저장 (queued)
* Kubernetes API로 데이터 재수집 Job 생성
* boto3로 SageMaker Training Job 생성
* AI FastAPI deployment API 호출
* 완료 이벤트 수신
* Job 상태 업데이트
* REST API로 상태 제공

---

## 8. 상태 관리

```text
queued → running → completed / failed
```

---

## 9. 모델 관리 전략

### 9.1 버전 관리

* 모델은 덮어쓰기 금지
* 버전 단위로 저장

---

### 9.2 운영 모델 전환

* 학습 완료 후 자동 적용 금지
* PATCH API를 통해 활성 모델 전환

---

### 9.3 롤백

* 이전 모델 유지
* 문제 발생 시 즉시 복구 가능

---

## 10. 임베딩 처리 정책

* SBERT는 AI 서버 내부에서 직접 사용
* 별도 REST API 제공하지 않음
* classify 및 training에서 공통 사용

---

## 11. classify 구조 재사용

classify 구조는 추론 경로에만 사용한다.
재학습 Job 생성은 RabbitMQ 요청 큐가 아니라 Admin 서버의 SageMaker API 호출로 수행한다.
학습 컨테이너는 상태 관리, 완료 이벤트, 모델 버전 정보를 Backend에 전달한다.

---

## 12. 설계 요약

* Admin REST는 Job을 생성하고 상태를 조회한다
* Admin 서버가 Kubernetes/SageMaker API를 직접 호출한다
* RabbitMQ/SSE는 실행 상태 이벤트와 로그 전달에 사용한다
* 학습은 SageMaker Training Job에서 비동기로 수행된다
* 모델은 버전 기반으로 관리된다

---

## 13. 향후 확장

* training worker 분리 (독립 서비스)
* 모델 자동 평가 및 승인 프로세스
* 스케줄 기반 재학습
* 실험/운영 모델 분리
* 모니터링 및 알림 시스템 추가
