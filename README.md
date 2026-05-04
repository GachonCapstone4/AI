# 업무 이메일 자동화 AI 시스템

> 캡스톤 디자인 프로젝트 — AI 기반 업무 이메일 자동 분류 및 응답 자동화 서비스

---

## 프로젝트 개요

Gmail API로 수신된 업무 이메일을 AI가 자동으로 분류하고,
LLM 기반 답장 초안 생성 및 일정 등록까지 자동화하는 AI Agent 서비스입니다.

---

## 전체 시스템 흐름
```
Gmail API
→ 백엔드 1차 필터링
→ AI 서버 (이 레포)
  → 전처리 (subject + body → email_text)
  → SBERT 임베딩 생성
  → 1차 분류: Domain (Logistic Regression)
  → 2차 분류: Intent (Logistic Regression)
  → LLM 후처리
    - GPT   : 이메일 요약 / 일정 추출
    - Claude : 답장 템플릿 초안 생성
  → Google Calendar 등록 후보 생성
```

---

## AI 모델 구조

| 단계 | 모델 | 역할 |
|------|------|------|
| 임베딩 | SBERT (paraphrase-multilingual-MiniLM-L12-v2) | 이메일 텍스트 → 벡터 변환 |
| 1차 분류 | Logistic Regression | 7개 Domain 분류 |
| 2차 분류 | Logistic Regression (Domain별) | 30개 Intent 분류 |
| 요약/추출 | GPT-4o-mini | 이메일 요약 / 일정 추출 |
| 답장 생성 | Claude 3.5 Sonnet | 답장 템플릿 초안 생성 |

---

### Domain / Intent 구조

| Domain | Intent 예시 |
|--------|------------|
| Sales | 견적 요청, 계약 문의, 가격 협상, 제안 요청, 미팅 일정 조율 |
| Marketing & PR | 협찬 제안, 광고 문의, 보도자료 요청, 인터뷰 요청 |
| HR | 채용 문의, 면접 일정 조율, 휴가 신청, 증명서 발급 |
| Finance | 세금계산서 요청, 비용 처리 문의, 입금 확인, 정산 문의 |
| Customer Support | 불만 접수, 기술 지원 요청, 환불 요청, 사용법 문의 |
| IT/Ops | 시스템 오류 보고, 계정 생성 요청, 접근 권한 변경 |
| Admin | 공지 전달, 내부 보고, 자료 요청, 협조 요청 |



---

## 기술 스택

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-2.7-green)
![Scikit--learn](https://img.shields.io/badge/ScikitLearn-1.4-yellow)

---

## MLOps 운영 구조

이 AI repo는 실행 컨테이너와 AI FastAPI 서버를 제공한다. Job 생성은 Admin 서버 책임이다.

| 작업 | Job 생성 주체 | 실행 이미지 | 실행 코드 |
|------|---------------|-------------|-----------|
| 데이터 재수집 | Admin 서버가 Kubernetes API 직접 호출 | dataset-batch ECR 이미지 | `batch/dataset_batch.py` |
| 재학습 | Admin 서버가 boto3 SageMaker `create_training_job` 직접 호출 | training ECR 이미지 | `src.mlops.training_container_entrypoint` |
| 재배포 | Admin 서버가 AI FastAPI API 순차 호출 | AI FastAPI 서버 | `/deployment/preload` → `/deployment/validate` → `/deployment/switch` |

### 데이터 재수집 이미지

`batch/Dockerfile.dataset`은 `dataset_batch.py`를 직접 실행한다.

```dockerfile
CMD ["python", "dataset_batch.py"]
```

Admin 서버는 Kubernetes Job body를 생성하고 dataset-batch ECR 이미지를 지정한다. 컨테이너에는 `JOB_ID`, `ADMIN_USER_ID`, DB 접속 정보, AWS/S3 정보, RabbitMQ 정보를 환경변수 또는 Secret으로 주입해야 한다.

### 재학습 이미지

`Dockerfile.training`은 SageMaker Training 컨테이너 entrypoint로 `training_container_entrypoint.py`를 직접 실행한다.

```dockerfile
ENTRYPOINT ["python", "-m", "src.mlops.training_container_entrypoint"]
```

Admin 서버는 SageMaker Training Job 생성 시 training ECR 이미지, `JOB_ID`, `DATASET_S3_URI`, `MODEL_VERSION`, `S3_BUCKET`, `S3_MODEL_PREFIX` 등을 넘긴다. 컨테이너 내부 학습 순서는 SBERT → Domain Logistic Regression → Intent Logistic Regression이다.

## Prometheus Metrics

AI FastAPI 서버는 로그 파일이나 별도 exporter 없이 프로세스 메모리에 추론 성능 metric을 누적하고 `GET /metrics`로 노출한다. `/metrics`는 classify 요청 수, latency, confidence score 분포, 오류 수, 일정 감지 수, active model version 같은 추론 지표 전용이다.

재학습 성능 결과인 `intent_f1`, `domain_accuracy`는 Prometheus metric으로 노출하지 않는다. training은 epoch별 metric 갱신 구조가 아니라 학습 완료 후 최종 1회 `metrics.json`을 생성하는 구조이며, 같은 최종 결과가 `q.2app.training`의 `COMPLETED` payload에 포함된다. Admin training 로그/결과 화면은 이 `COMPLETED` payload의 `metrics`를 기준으로 표시한다.

Prometheus scrape 예시:

```yaml
scrape_configs:
  - job_name: "ai-server"
    static_configs:
      - targets: ["AI_SERVER_IP:8080"]
```

Grafana PromQL 예시:

```promql
rate(ai_classify_requests_total[1m])
histogram_quantile(0.95, rate(ai_classify_latency_seconds_bucket[5m]))
ai_active_model_info
```

---

## 라이선스

본 프로젝트는 캡스톤 디자인 학술 목적으로 제작되었습니다.
