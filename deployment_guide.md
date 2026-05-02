# AI 서버 재배포 구현 가이드라인

## 1. 목적

현재 데이터 재수집과 재학습 파이프라인은 동작 확인이 완료되었다.

- 데이터 재수집: Admin 서버 → Kubernetes Job → dataset 생성 → S3 업로드
- 재학습: Admin 서버 → SageMaker Training Job → 모델 학습 → S3 모델 산출물 저장
- 재배포: 아직 AI 서버 쪽 코드 정리가 필요함

이 문서는 재배포 기능을 구현하기 위한 기준 문서다.

재배포의 목적은 SageMaker Training Job이 S3에 저장한 새 모델을 AI 추론 서버에 무중단으로 반영하는 것이다.

---

## 2. 현재 성공한 모델 산출물 구조

SageMaker Training Job 성공 후 S3에는 아래 구조로 모델이 저장된다.

```text
s3://capstone-gachon/models/{modelVersion}/
 ├── config.json
 ├── domain_model.pkl
 ├── intent_model.pkl
 ├── label_mapping.json
 ├── metrics.json
 └── sbert/
     ├── config.json
     ├── model.safetensors
     ├── tokenizer.json
     ├── tokenizer_config.json
     ├── modules.json
     └── ...
```

예시:

```text
s3://capstone-gachon/models/training-final-004/
```

latest 포인터:

```text
s3://capstone-gachon/models/latest.json
```

latest.json 예시:

```json
{
  "model_version": "training-final-004",
  "updated_at": "2026-04-30T13:35:13.946648Z",
  "job_id": "training-final-004",
  "artifact_s3_uri": "s3://capstone-gachon/models/training-final-004/",
  "metrics": {
    "domain_accuracy": 0.9013,
    "intent_f1": 0.9907
  }
}
```

---

## 3. 핵심 오개념 방지

### 3.1 재학습과 재배포는 다르다

재학습은 SageMaker Job이 수행한다.

```text
Admin 서버
→ SageMaker Training Job 생성
→ S3에 모델 저장
→ latest.json 갱신
```

재배포는 AI 서버가 수행한다.

```text
Admin 서버
→ AI 서버 preload
→ AI 서버 validate
→ AI 서버 switch
```

SageMaker Job은 AI 서버 메모리 모델을 자동으로 교체하지 않는다.

---

### 3.2 latest.json의 역할

latest.json은 현재 최신 모델 위치를 가리키는 포인터다.

AI 서버는 서버 시작 시 또는 재배포 시 latest.json을 읽어서 최신 modelVersion을 알 수 있어야 한다.

단, 재배포 API가 특정 modelVersion을 받는 경우에는 해당 version을 직접 preload할 수 있다.

---

## 4. 목표 구조

### 4.1 Frontend

프론트는 Admin 서버만 호출한다.

```text
POST /api/admin/ai-training/deployment-jobs
```

프론트는 AI 서버의 `/deployment/preload`, `/deployment/validate`, `/deployment/switch`를 직접 호출하지 않는다.

---

### 4.2 Admin 서버

Admin 서버는 재배포 Job을 생성하고 AI 서버에 배포 요청을 순서대로 보낸다.

```text
1. DB에 Deployment Job 생성: QUEUED
2. 상태 RUNNING 변경
3. AI 서버 /deployment/preload 호출
4. AI 서버 /deployment/validate 호출
5. AI 서버 /deployment/switch 호출
6. 상태 COMPLETED 변경
7. 실패 시 FAILED 저장
```

---

### 4.3 AI 서버

AI 서버는 실제 모델 메모리를 관리한다.

```text
current_bundle = 현재 서비스 중인 모델
staging_bundle = 새로 준비 중인 모델
```

재배포는 아래 순서로 수행한다.

```text
preload → validate → switch
```

---

## 5. AI 서버에 필요한 API

## 5.1 preload

### Endpoint

```text
POST /deployment/preload
```

### Request

```json
{
  "modelVersion": "training-final-004"
}
```

또는 modelVersion이 없으면 latest.json을 읽어 최신 버전을 사용한다.

```json
{}
```

### 역할

```text
1. modelVersion 결정
2. S3에서 models/{modelVersion}/ 다운로드
3. 표준 산출물 구조 검증
4. SBERT, domain_model.pkl, intent_model.pkl, label_mapping.json 로드
5. staging_bundle에 저장
6. current_bundle은 변경하지 않음
```

### Response

```json
{
  "status": "preloaded",
  "modelVersion": "training-final-004",
  "artifactS3Uri": "s3://capstone-gachon/models/training-final-004/"
}
```

---

## 5.2 validate

### Endpoint

```text
POST /deployment/validate
```

### 역할

staging_bundle이 정상적으로 추론 가능한지 확인한다.

검증 기준:

```text
1. staging_bundle이 존재해야 함
2. 샘플 이메일 1~2개로 predict 실행
3. domain이 비어 있으면 실패
4. intent가 비어 있으면 실패
5. 예측된 domain이 label_mapping에 존재해야 함
6. 예측된 intent가 해당 domain의 intent 목록에 존재해야 함
```

### Response

```json
{
  "status": "validated",
  "modelVersion": "training-final-004",
  "samples": [
    {
      "domain": "Finance",
      "intent": "세금계산서 요청"
    }
  ]
}
```

---

## 5.3 switch

### Endpoint

```text
POST /deployment/switch
```

### 역할

검증 완료된 staging_bundle을 current_bundle로 교체한다.

```text
current_bundle = staging_bundle
current_model_version = staging_model_version
staging_bundle = None
staging_model_version = None
```

교체는 lock 안에서 참조만 변경해야 한다.

### Response

```json
{
  "status": "switched",
  "activeModelVersion": "training-final-004"
}
```

---

## 6. ModelManager 구현 기준

ModelManager는 AI 서버의 모델 상태를 관리한다.

```python
class ModelManager:
    current_bundle = None
    current_model_version = None

    staging_bundle = None
    staging_model_version = None

    _lock = RLock()
```

### 필수 메서드

```python
preload(model_version: str | None) -> dict
validate() -> dict
switch() -> dict
predict(subject: str, body: str) -> dict
```

### 중요한 규칙

```text
1. /classify는 current_bundle만 사용한다.
2. preload는 staging_bundle만 변경한다.
3. validate는 staging_bundle만 검증한다.
4. switch 순간에만 current_bundle을 교체한다.
5. validate 실패 시 current_bundle은 그대로 유지한다.
6. 매 요청마다 모델을 다시 로딩하면 안 된다.
```

---

## 7. 모델 로딩 구현 기준

현재 학습 산출물은 표준 포맷이다.

반드시 아래 파일명을 기준으로 로딩해야 한다.

```text
sbert/
domain_model.pkl
intent_model.pkl
label_mapping.json
metrics.json
config.json
```

사용하면 안 되는 예전 파일명:

```text
domain_classifier.pkl
domain_label_encoder.pkl
intent_classifiers.pkl
intent_label_encoders.pkl
```

### 로딩 함수 기준

```python
load_standard_model_bundle(model_version: str)
```

이 함수는 다음을 수행해야 한다.

```text
1. S3 prefix 결정: models/{modelVersion}/
2. 로컬 캐시 경로 결정
3. 필요한 파일 다운로드
4. 필수 파일 존재 여부 검증
5. SentenceTransformer 로드
6. domain_model.pkl 로드
7. intent_model.pkl 로드
8. label_mapping.json 로드
9. bundle 객체 반환
```

---

## 8. latest.json 기반 startup 로딩

서버 시작 시 기본 동작은 latest.json 기반 로딩이어야 한다.

```text
1. S3에서 models/latest.json 읽기
2. latest.json에서 model_version 또는 artifact_s3_uri 확인
3. 해당 modelVersion으로 load_standard_model_bundle 실행
4. current_bundle에 저장
```

### ACTIVE_MODEL_VERSION 정책

ACTIVE_MODEL_VERSION은 optional override로만 사용한다.

```text
ACTIVE_MODEL_VERSION이 있으면 해당 버전 로드
ACTIVE_MODEL_VERSION이 없으면 latest.json 사용
```

하드코딩된 modelVersion에 의존하면 안 된다.

---

## 9. classify API 기준

### Endpoint

```text
POST /classify
```

### Request

```json
{
  "subject": "Invoice request",
  "body": "Please review the attached invoice."
}
```

또는 기존 필드명을 유지한다면:

```json
{
  "subject": "Invoice request",
  "body_clean": "Please review the attached invoice."
}
```

### 입력 검증

subject와 body/body_clean이 모두 비어 있으면 400 에러를 반환한다.

```text
subject empty + body empty = invalid request
```

### 추론 순서

반드시 아래 구조를 유지한다.

```text
subject + body
→ email_text 생성
→ SBERT embedding
→ Domain Logistic Regression
→ Intent Logistic Regression
→ domain, intent 반환
```

LLM을 분류기로 사용하면 안 된다.

---

## 10. 캐시 다운로드 기준

S3 모델 다운로드는 atomic하게 처리해야 한다.

### 문제 상황

여러 worker가 동시에 모델을 다운로드하면 partial file이 남을 수 있다.

### 권장 방식

```text
1. models/{version}/ 를 temp directory에 다운로드
2. 필수 파일 검증
3. 검증 성공 시 최종 cache directory로 rename
4. 완료 marker 파일 생성
```

예시:

```text
.cache/model-cache/
 ├── training-final-004.tmp/
 ├── training-final-004/
 │   ├── .complete
 │   ├── sbert/
 │   ├── domain_model.pkl
 │   ├── intent_model.pkl
 │   └── label_mapping.json
```

---

## 11. 에러 처리 기준

### preload 실패

```text
원인:
- S3 권한 없음
- 모델 파일 누락
- pickle 로딩 실패
- SBERT 로딩 실패

처리:
- staging_bundle 변경하지 않음
- current_bundle 유지
- Admin 서버에 실패 응답
```

### validate 실패

```text
처리:
- current_bundle 유지
- staging_bundle은 폐기 가능
- switch 수행 금지
```

### switch 실패

```text
처리:
- current_bundle이 손상되지 않도록 lock 안에서 최소 작업만 수행
- 실패 시 기존 current_bundle 유지
```

---

## 12. Admin 서버 재배포 흐름

Admin 서버의 deployment job 흐름은 아래와 같다.

```text
1. POST /api/admin/ai-training/deployment-jobs 수신
2. DeploymentJob DB 저장: QUEUED
3. 상태 RUNNING 변경
4. AI_SERVER_URL/deployment/preload 호출
5. AI_SERVER_URL/deployment/validate 호출
6. AI_SERVER_URL/deployment/switch 호출
7. 상태 COMPLETED 변경
8. activeModelVersion 저장
9. 실패 시 FAILED + errorMessage 저장
```

### Admin 요청 예시

```json
{
  "modelVersion": "training-final-004"
}
```

modelVersion을 생략하면 latest.json 기준 배포로 처리할 수도 있다.

```json
{}
```

단, 운영에서는 명시적으로 배포할 modelVersion을 넘기는 방식을 권장한다.

---

## 13. 상태 흐름

```text
QUEUED
→ RUNNING
→ COMPLETED
```

실패 시:

```text
QUEUED
→ RUNNING
→ FAILED
```

실패해도 AI 서버의 current_bundle은 유지되어야 한다.

---

## 14. SSE / 로그 기준

재배포 중 Admin 서버는 프론트에 진행 상태를 전달한다.

예시 로그:

```text
[INFO] Deployment job queued
[INFO] Preload started: training-final-004
[INFO] Model downloaded
[INFO] Model loaded into staging
[INFO] Validation started
[INFO] Validation succeeded
[INFO] Switching active model
[INFO] Deployment completed
```

실패 로그:

```text
[ERROR] Preload failed: missing domain_model.pkl
[ERROR] Validation failed: invalid intent label
[ERROR] Deployment failed
```

---

## 15. 테스트 기준

### 15.1 단위 테스트

필수 테스트:

```text
1. latest.json 읽기 테스트
2. 표준 산출물 로딩 테스트
3. missing file 시 실패 테스트
4. preload 성공 시 staging_bundle 생성 테스트
5. validate 성공/실패 테스트
6. switch 후 current_bundle 변경 테스트
7. switch 후 staging_bundle 초기화 테스트
8. classify가 current_bundle을 사용하는지 테스트
9. 빈 입력 subject/body 검증 테스트
```

---

### 15.2 통합 테스트

실제 S3 모델 버전으로 테스트한다.

```bash
curl -X POST http://AI_SERVER/deployment/preload \
  -H "Content-Type: application/json" \
  -d '{"modelVersion":"training-final-004"}'

curl -X POST http://AI_SERVER/deployment/validate

curl -X POST http://AI_SERVER/deployment/switch

curl -X POST http://AI_SERVER/classify \
  -H "Content-Type: application/json" \
  -d '{"subject":"Invoice request","body":"Please review the attached invoice."}'
```

기대 결과:

```text
preload 성공
validate 성공
switch 성공
classify 정상 응답
```

---

## 16. 현재 코드에서 우선 수정해야 할 항목

현재 검토 결과 기준 우선순위는 아래와 같다.

### High Priority

```text
1. startup 경로를 latest.json → load_standard_model_bundle 구조로 변경
2. startup에서 예전 파일명 resolver를 사용하지 않도록 수정
3. load_standard_model_bundle이 표준 산출물 구조를 기준으로 동작하는지 확인
4. /classify가 ModelManager.current_bundle을 단일 source of truth로 사용하도록 정리
```

### Medium Priority

```text
1. subject/body 둘 다 빈 경우 400 처리
2. S3 cache atomic download 적용
3. label_mapping.json 기반 검증 강화
4. startup 실패 시 정책 결정
   - 운영에서는 이전 캐시 fallback 권장
   - 개발에서는 명확히 crash해도 됨
```

---

## 17. 배포 가능 기준

아래 조건을 모두 만족해야 배포 가능하다.

```text
1. AI 서버 시작 시 latest.json 기반 모델 로딩 성공
2. /classify 정상 응답
3. /deployment/preload 성공
4. /deployment/validate 성공
5. /deployment/switch 성공
6. switch 후 /classify가 새 모델 버전으로 응답
7. validate 실패 시 기존 current 모델 유지
8. 모델을 매 요청마다 reload하지 않음
```

---

## 18. 최종 목표

최종적으로 아래 구조가 되어야 한다.

```text
Frontend
→ Admin Server
→ AI Server /deployment/preload
→ AI Server /deployment/validate
→ AI Server /deployment/switch
→ AI Server current model 교체
→ Frontend에 COMPLETED 표시
```

재배포는 학습이 아니다.

재배포는 S3에 이미 저장된 모델을 AI 서버 메모리에 안전하게 반영하는 작업이다.