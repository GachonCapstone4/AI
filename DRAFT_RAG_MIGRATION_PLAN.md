# DRAFT_RAG_MIGRATION_PLAN

## 문서 목적

이 문서는 `draft` 책임을 장기적으로 AI 서버에서 RAG 서버로 이전하기 위한 아키텍처 설계 초안이다. 현재 단계에서는 구현이나 코드 변경이 아니라, 병행 유지 전략 하에서 경계와 인터페이스를 명확히 정의하는 것이 목적이다.

## 범위

- `classify`는 AI 서버의 공식 코어 기능으로 유지한다.
- `draft`는 당분간 AI 서버 내부에 남겨 두되, 내부용/실험용/fallback/deprecated candidate 경로로 간주한다.
- 장기적으로는 `draft`의 주 책임을 RAG 서버로 옮기는 방향을 검토한다.
- 본 문서는 현재 코드베이스와 `RABBITMQ_SPEC.md`만을 기준으로 작성한다.

## 상태 라벨

- `Confirmed`: 현재 코드 또는 `RABBITMQ_SPEC.md`로 직접 확인된 내용
- `Inferred`: 현재 코드 구조를 기반으로 한 합리적 설계 제안
- `Spec unclear`: `RABBITMQ_SPEC.md`에 명시되지 않아 계약으로 단정할 수 없는 내용

---

## 1. 현재 `draft` 입력/출력 구조

### 1-1. 현재 `/draft` 요청 구조

`Confirmed`

현재 `api/schemas/draft.py` 기준 `DraftRequest`는 아래 필드를 사용한다.

```json
{
  "request_id": "req-draft-001",
  "mode": "generate",
  "emailId": "email-123",
  "subject": "납품 일정 문의",
  "body": "정제되지 않은 원문 또는 본문",
  "domain": "업무",
  "intent": "문의",
  "summary": "납품 일정 확인 요청",
  "previous_draft": null
}
```

필드 요약:

- `request_id`: 요청 추적용 식별자
- `mode`: `"generate"` 또는 `"regenerate"`
- `emailId`: 이메일 식별자
- `subject`: 원본 제목
- `body`: 본문
- `domain`: 분류 결과 도메인
- `intent`: 분류 결과 인텐트
- `summary`: 요약 결과
- `previous_draft`: 재생성 시 사용, 스키마상 optional

`Confirmed`

- `/draft` 라우터는 `request.app.state.draft_pipeline`을 사용한다.
- `mode == "regenerate"` 이고 `previous_draft`가 비어 있으면 `draft_service.run_draft`에서 `ValueError`를 발생시킨다.
- HTTP 경로에서는 이 경우 `422`를 반환한다.

### 1-2. 현재 `/draft` 응답 구조

`Confirmed`

현재 `DraftResponse`는 아래 구조다.

```json
{
  "request_id": "req-draft-001",
  "emailId": "email-123",
  "draft_reply": "안녕하세요. 문의 주신 내용에 대해 답변드립니다.",
  "reply_embedding": [0.123, 0.456, 0.789]
}
```

추가로 MQ consumer 경로에서는 `meta`가 포함될 수 있다.

```json
{
  "request_id": "req-draft-001",
  "emailId": "email-123",
  "draft_reply": "안녕하세요. 문의 주신 내용에 대해 답변드립니다.",
  "reply_embedding": [0.123, 0.456, 0.789],
  "meta": {
    "elapsed_ms": 210.5,
    "source": "consumer.draft"
  }
}
```

### 1-3. 현재 `consumer_draft` 입력/출력 구조

`Confirmed`

코드 기준으로 `messaging/consumer_draft.py`는 JSON body를 받아 `DraftRequest`로 파싱하고, 성공 시 `DraftResponse + meta`를 publish한다.

성공 출력 예시:

```json
{
  "request_id": "req-draft-001",
  "emailId": "email-123",
  "draft_reply": "안녕하세요. 문의 주신 내용에 대해 답변드립니다.",
  "reply_embedding": [0.123, 0.456, 0.789],
  "meta": {
    "elapsed_ms": 123.4,
    "source": "consumer.draft"
  }
}
```

검증 오류 출력 예시:

```json
{
  "request_id": "req-draft-001",
  "emailId": "email-123",
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "error_message": "mode=regenerate 일 때 previous_draft 는 필수입니다.",
  "meta": {
    "elapsed_ms": 12.3,
    "source": "consumer.draft"
  }
}
```

`Spec unclear`

- `RABBITMQ_SPEC.md`에는 `draft`용 queue, routing key, payload 계약이 정의되어 있지 않다.
- 오히려 문서에는 `draft는 온프레미스 RAG 서버에서 처리 — AI 서버 담당 아님`이라고 적혀 있다.
- 따라서 현재 `consumer_draft.py`의 MQ 입출력은 코드상 존재하지만, 스펙 기준 공식 계약으로는 확정할 수 없다.

### 1-4. 현재 `draft` 경로의 의존성

`Confirmed`

현재 `draft` 경로는 다음에 의존한다.

- `DraftRequest`, `DraftResponse`, `ErrorResponse`, `ResponseMeta`
- `draft_service.run_draft`
- `claude_service.generate_draft`
- `draft_pipeline["model"]["sbert"]`
- `Anthropic` API key 및 외부 Claude 호출

`Confirmed`

현재 로직 순서:

1. 입력 검증
2. `generate_draft`로 답장 생성
3. 생성된 답장에 대해 SBERT 임베딩 계산
4. HTTP 응답 또는 MQ publish

### 1-5. 현재 `draft`가 의존하는 classify 유래 정보

`Confirmed`

현재 `draft` 입력에는 아래 classify 유래 정보가 이미 포함되어 있다.

- `domain`
- `intent`
- `summary`

`Inferred`

즉, 현재 `draft`는 원칙적으로 classify 결과를 소비하는 후행 단계이며, 스스로 분류를 수행하지 않는다.

---

## 2. 책임 식별

### 2-1. AI 서버에 남아야 하는 책임

`Confirmed`

현재 코드와 스펙 기준 AI 서버의 공식 코어 책임은 `classify`다.

AI 서버 잔류 권장 책임:

- 이메일 분류
- 요약 생성
- 일정 추출
- 분류 결과 임베딩 생성
- `classify` 관련 HTTP/MQ 계약 유지

`Inferred`

아래 책임도 AI 서버에 남기는 편이 일관적이다.

- `domain`, `intent`, `summary`, `schedule_info`, `email_embedding` 생성
- 분류 결과를 다른 서버가 사용할 수 있도록 안정된 계약으로 제공

### 2-2. RAG 서버로 이동해야 하는 책임

`Confirmed`

현재 코드상 `draft`는 내부/실험/fallback/deprecated candidate로 취급되고 있으며, `claude_service`는 미래 이전 시 제거 후보로 표시되어 있다.

RAG 서버 이전 권장 책임:

- 답장 초안 생성
- 재생성 로직
- 템플릿 선택
- 검색 기반 문맥 결합
- grounded generation
- 문서/정책/FAQ 기반 응답 구성
- `previous_draft`를 활용한 개선 초안 생성
- Claude 또는 다른 생성기 연동 책임

### 2-3. embedding 책임 분리 판단

`Inferred`

RAG 서버 담당 논의 기준으로, `/draft`용 임베딩 생성은 AI 서버가 별도 제공하지 않는 방향이 더 타당하다.

판단 근거:

- RAG 서버가 `draft_reply` 생성 직후 자체적으로 GPT 임베딩 API를 호출하는 방안을 검토 중이다.
- 이 경우 `reply_embedding`은 `draft` 생성 결과물의 후처리이므로, 생성 책임을 가진 RAG 서버가 함께 소유하는 편이 경계가 명확하다.
- AI 서버가 `/draft` 전용 임베딩 생성 기능을 별도로 제공하면, 실제로는 `draft` 책임이 다시 AI 서버 쪽으로 일부 회귀하게 된다.
- 추가적인 AI 서버 호출이 생기면 서비스 hop, 장애 포인트, 계약 관리 비용이 모두 증가한다.

`Recommended`

- `/draft` 응답의 `reply_embedding`이 필요하다면 RAG 서버가 직접 생성해서 반환한다.
- AI 서버는 `/draft`를 위한 별도 임베딩 생성 API 또는 내부 보조 경로를 제공하지 않는다.
- 따라서 `/draft` 관점의 embedding 책임은 AI 서버가 아니라 RAG 서버에 둔다.
- 단, 응답 계약 호환성을 위해 `reply_embedding` 필드는 유지하되 생성 주체만 RAG 서버로 명시하는 것이 적절하다.

---

## 3. Backend ↔ RAG 계약 초안

이 절은 미래 HTTP 계약 초안이다. 구현이 아니라 논의용 인터페이스 정의다.

### 3-1. 제안 경로

`Inferred`

- `POST /draft`

설명:

- Backend가 장기적으로 직접 RAG 서버를 호출하는 경우의 후보 경로
- 현재 AI 서버 경로와 이름을 맞추면 Backend 수정 폭을 줄일 수 있다

### 3-2. 요청 JSON 초안

`Recommended`

```json
{
  "request_id": "req-draft-001",
  "emailId": "email-123",
  "mode": "generate",
  "subject": "납품 일정 문의",
  "body": "안녕하세요. 납품 일정 확인 부탁드립니다.",
  "classification": {
    "domain": "업무",
    "intent": "문의"
  },
  "summary": "납품 일정 확인 요청",
  "previous_draft": null,
  "context": {
    "sender_name": "홍길동",
    "sender_email": "sender@gmail.com",
    "received_at": "2026-04-06T10:00:00"
  }
}
```

### 3-3. 요청 필드 정의

`Required`

- `request_id`
- `emailId`
- `mode`
- `subject`
- `body`
- `classification.domain`
- `classification.intent`
- `summary`

`Optional`

- `previous_draft`
- `context.sender_name`
- `context.sender_email`
- `context.received_at`

`Inferred`

- `classification`을 object로 묶으면 현재 AI 서버 `ClassifyResponse`와 연결하기가 쉬워진다.
- 현재 AI 서버 내부 `DraftRequest`는 `domain`, `intent` 평면 필드이지만, 미래 RAG 계약은 더 구조적인 형태가 유지보수에 유리하다.

### 3-4. 응답 JSON 초안

`Recommended`

```json
{
  "request_id": "req-draft-001",
  "emailId": "email-123",
  "draft_reply": "안녕하세요. 문의 주신 납품 일정에 대해 답변드립니다.",
  "reply_embedding": [0.123, 0.456, 0.789],
  "generation_meta": {
    "generator": "rag-server",
    "strategy": "retrieval-grounded",
    "model": "tbd"
  }
}
```

### 3-5. 응답 필드 정의

`Required`

- `request_id`
- `emailId`
- `draft_reply`

`Optional`

- `reply_embedding`
- `generation_meta.generator`
- `generation_meta.strategy`
- `generation_meta.model`

`Recommended`

- 병행 유지 단계에서는 `reply_embedding`을 optional로 두되, 실제 운영에서는 가능한 한 포함시키는 방향이 좋다.

### 3-6. 에러 처리 초안

`Recommended`

검증 오류:

```json
{
  "request_id": "req-draft-001",
  "emailId": "email-123",
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "error_message": "previous_draft is required when mode=regenerate"
}
```

처리 오류:

```json
{
  "request_id": "req-draft-001",
  "emailId": "email-123",
  "status": "error",
  "error_code": "PROCESSING_ERROR",
  "error_message": "temporary generation failure"
}
```

에러 처리 고려사항:

- `request_id`, `emailId`는 항상 보존
- 검증 오류와 일시적 처리 오류를 구분
- Backend가 재시도 정책을 다르게 적용할 수 있어야 함

### 3-7. 하위 호환성 메모

`Recommended`

- 현재 AI 서버 `DraftResponse`의 핵심 필드인 `request_id`, `emailId`, `draft_reply`, `reply_embedding`는 최대한 유지한다.
- 새 필드는 optional로 추가한다.
- 가능하면 기존 Backend DTO 수정 없이 수용 가능한 형태를 우선 목표로 한다.

---

## 4. 병행 유지 전략

### 4-1. 목표 상태

`Recommended`

- AI 서버 `draft`: 내부용/실험용/fallback 경로
- RAG 서버 `draft`: 장래의 주 경로 후보
- `classify`: 계속 AI 서버의 공식 코어

### 4-2. 병행 유지 동작 방식

`Inferred`

가능한 운영 모드:

1. `AI-primary`
   - 현행 유지
   - RAG는 비교 실험만 수행

2. `RAG-primary`
   - RAG 결과를 우선 사용
   - AI 서버 `draft`는 실패 시 fallback

3. `Shadow`
   - 사용자 응답은 한쪽만 사용
   - 다른 쪽은 비공개 비교용으로만 실행

### 4-3. 비교/테스트 방법

`Recommended`

- 동일 입력에 대해 AI 서버 초안과 RAG 서버 초안을 모두 생성
- 비교 항목:
  - 응답 생성 성공률
  - 평균 지연 시간
  - 재생성 성공률
  - 운영자 품질 평가
  - hallucination 또는 근거 불일치 사례

`Inferred`

초기에는 자동 승패 판정보다 샘플 리뷰 기반 비교가 더 현실적이다.

### 4-4. 병행 유지의 장단점

장점:

- 위험 분산
- 품질 비교 가능
- 급격한 전환 회피
- fallback 확보

단점:

- 운영 경로가 이중화됨
- 장애 판단이 복잡해짐
- 로깅/모니터링 기준이 늘어남
- 계약 동기화 비용이 생김

---

## 5. Proxy 전환 옵션

### 5-1. 개념

`Inferred`

장래에 AI 서버 `/draft`를 유지하되, 내부 구현이 RAG 서버 호출로 바뀌는 방식이다.

### 5-2. 어떤 계층이 RAG를 호출하는가

`Recommended`

- 호출 위치는 `draft_service` 경계가 가장 적절하다.

이유:

- 라우터는 HTTP 계약만 유지
- 서비스는 입력 검증과 전송 책임을 담당
- 실제 생성 책임은 외부 RAG로 이전

### 5-3. Proxy 모드에서 AI 서버가 계속 하는 일

`Inferred`

- 기존 `/draft` HTTP 경로 유지
- 입력 검증
- `mode`/`previous_draft` 검증
- 요청/응답 로깅
- 필요 시 응답 변환
- 필요 시 fallback 결정

### 5-4. 위험 요소

위험:

- AI 서버와 RAG 서버의 이중 장애 가능성
- timeout 관리 복잡도 증가
- 에러 코드 매핑 필요
- 응답 구조 차이로 인한 변환 비용
- fallback 기준이 모호하면 운영 혼란 발생

### 5-5. 장단점 평가

장점:

- Backend 변경이 가장 적음
- 점진적 전환이 쉬움
- AI 서버 fallback 유지가 용이함

단점:

- 중간 hop이 늘어남
- 장기적으로는 AI 서버가 불필요한 중계층이 될 수 있음
- `draft` 책임이 완전히 분리되지 않음

---

## 6. Full migration 옵션

### 6-1. 개념

`Inferred`

`draft`를 AI 서버에서 완전히 제거하고, Backend가 직접 RAG 서버를 사용하거나 별도 RAG 계약으로 완전 이전하는 방식이다.

### 6-2. 제거 가능한 구성요소

`Confirmed`

현재 코드 기준 제거 후보:

- `api/routers/draft.py`
- `api/services/draft_service.py`
- `api/services/claude_service.py`
- `messaging/consumer_draft.py`
- `api.schemas.draft` 중 AI 서버 내 draft 전용 모델
- `app.state.draft_pipeline`
- `load_draft_pipeline` 관련 AI 서버 초기화 경로

### 6-3. AI 서버에 남는 것

`Confirmed`

- `classify` 경로
- `summarize` 경로
- classify 관련 schema, service, consumer
- classify pipeline 및 분류 모델 로딩

### 6-4. 운영 영향

장점:

- AI 서버 책임이 단순해짐
- classify 코어 정체성이 더 선명해짐
- draft 장애가 AI 서버 운영에 덜 영향을 줌

단점:

- Backend 또는 운영 인프라에서 RAG 서버를 별도 관리해야 함
- 전환 시점에 계약 불일치가 있으면 장애 가능성 존재
- fallback 경로를 잃을 수 있음

---

## 7. 권장 마이그레이션 로드맵

### 단계 1. 현재 상태 유지

`Confirmed`

- AI 서버 `classify`는 공식 코어로 유지
- AI 서버 `draft`는 내부/실험/fallback/deprecated candidate로 유지

### 단계 2. 인터페이스 정리

`Recommended`

- Backend ↔ RAG 초안 계약 확정
- 필수/선택 필드 확정
- 에러 코드 체계 정렬
- `reply_embedding` 포함 여부 확정

### 단계 3. 병행 유지 시작

`Recommended`

- RAG 서버 `draft`를 별도 실험 경로로 운영
- 같은 입력으로 결과 비교
- 품질/지연/장애율 측정

### 단계 4. 인터페이스 동결

`Recommended`

- 충분한 비교 후 request/response 구조 동결
- Backend 연동 정책 확정
- fallback 기준 확정

### 단계 5. 경로 선택

`Decision point`

선택지 A: Proxy 전환

- Backend 변경 최소화가 중요할 때 적합

선택지 B: Full migration

- 장기 단순성과 책임 분리가 더 중요할 때 적합

### 단계 6. 최종 정리

`Recommended`

Proxy 선택 시:

- AI 서버 `draft`를 RAG 호출 래퍼로 단순화
- 로컬 생성기와 직접 생성 의존성 제거

Full migration 선택 시:

- AI 서버 draft 관련 모듈 제거
- 운영 문서와 장애 대응 체계 정리

---

## 설계 결론

`Recommended`

가장 현실적인 다음 단계는 아래와 같다.

1. `classify`는 계속 AI 서버 공식 코어로 고정한다.
2. `draft`는 당분간 AI 서버 내부 fallback 경로로 유지한다.
3. RAG 서버용 `POST /draft` 계약을 먼저 동결한다.
4. 초기에는 병행 유지와 shadow 비교를 수행한다.
5. 품질과 운영 복잡도를 평가한 뒤 `proxy` 또는 `full migration` 중 하나를 선택한다.

## 확인된 사실과 미확정 사항 요약

### Confirmed

- AI 서버의 공식 코어는 `classify`다.
- 현재 AI 서버 내부에는 `/draft` HTTP 경로와 `consumer_draft.py`가 존재한다.
- 현재 `draft`는 `domain`, `intent`, `summary`를 입력으로 사용한다.
- 현재 `draft`는 로컬에서 초안 생성 후 `reply_embedding`을 계산한다.

### Spec unclear

- `draft` MQ 계약은 `RABBITMQ_SPEC.md`에 정의되어 있지 않다.
- 따라서 `consumer_draft.py`의 MQ 입출력을 미래 공식 계약으로 그대로 간주할 수는 없다.

### Inferred

- 장기적으로는 RAG 서버가 `draft` 생성 책임의 자연스러운 주체다.
- 병행 유지 후 `proxy` 또는 `full migration`을 결정하는 전략이 가장 안전하다.
