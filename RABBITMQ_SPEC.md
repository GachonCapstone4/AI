# RabbitMQ 연동 스펙

> 최종 수정: 2026-05-07
> 대상: 백엔드(Java) ↔ AI 서버 연동

---

## 1. 연결 정보

| 항목 | 값 |
|---|---|
| AMQP URL | `amqp://admin:admin1234!@192.168.2.20:30672/` |
| 관리 UI | `http://192.168.2.20:31672/#/q` |

---

## 2. 토폴로지

| Exchange | Type | 방향 |
|---|---|---|
| `x.app2ai.direct` | direct | 백엔드 → AI |
| `x.ai2app.direct` | direct | AI → 백엔드 |
| `x.sse.fanout` | fanout | AI/training → SSE 구독자 |

| Queue | Exchange | Binding Key | 방향 |
|---|---|---|---|
| `q.2ai.classify` | x.app2ai.direct | 2ai.classify | 백엔드 → AI |
| `q.2app.classify` | x.ai2app.direct | 2app.classify | AI → 백엔드 |
| `q.ai.deployment` | x.app2ai.direct | deployment | 백엔드/Admin → AI |
| `q.2app.deployment` | `<default>` | q.2app.deployment | AI → 백엔드/Admin |
| `q.2app.training` | x.ai2app.direct | q.2app.training | AI training 컨테이너 → Admin |

- 모든 Exchange / Queue: `durable=true`, `delivery_mode=2` (persistent)
- `content_type`: `application/json`, 인코딩: UTF-8
- `q.2ai.classify` / `q.2app.classify`는 이메일 분류 전용이다. 재배포 요청/상태 전달에 사용하지 않는다.
- `q.2app.training`은 재학습 상태 요약 이벤트 전용이다. 긴 학습 로그, stdout/stderr 성격 로그를 보내지 않는다.
- 긴 학습 로그와 실시간 진행 로그는 `x.sse.fanout`으로 publish한다. fanout exchange이므로 queue routing key에 의존하지 않는다.

---

## 3. 메시지 흐름

```text
백엔드(Java)
  ├─ publish exchange=x.app2ai.direct routing_key=2ai.classify
  ├─ routed to queue=q.2ai.classify
  ├─ AI consumer consumes queue=q.2ai.classify
  ├─ classify processing
  ├─ AI publish exchange=x.ai2app.direct routing_key=2app.classify
  ├─ routed to queue=q.2app.classify
  └─ 백엔드 consumer consumes queue=q.2app.classify
```

> draft는 온프레미스 RAG 서버에서 처리 — AI 서버 담당 아님

---

## 4. classify

### 4-1. 요청 — 백엔드가 exchange `x.app2ai.direct` 로 publish

```text
publish to exchange x.app2ai.direct with routing key 2ai.classify
message is routed to queue q.2ai.classify
AI consumer consumes from queue q.2ai.classify
```

```json
{
  "outbox_id":    1,
  "email_id":     123,
  "sender_email": "sender@gmail.com",
  "sender_name":  "홍길동",
  "subject":      "회의 일정 안내",
  "body_clean":   "정제된 본문...",
  "received_at":  "2026-04-06T10:00:00"
}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| outbox_id | int | ✅ | 발신함 식별자. 응답에 그대로 보존 |
| email_id | int | ✅ | 이메일 식별자. 응답에 그대로 보존 |
| sender_email | string | ✅ | 발신자 이메일 |
| sender_name | string | ✅ | 발신자 이름 |
| subject | string | ✅ | 이메일 제목 |
| body_clean | string | ✅ | 정제된 이메일 본문 |
| received_at | string 또는 배열 | ✅ | 수신 시각 (ISO 문자열 또는 `[year,month,day,hour,min]` 배열) |

### 4-2. 응답 — AI가 exchange `x.ai2app.direct` 로 publish

```text
publish to exchange x.ai2app.direct with routing key 2app.classify
message is routed to queue q.2app.classify
Backend consumer consumes from queue q.2app.classify
```

```json
{
  "outbox_id": 1,
  "email_id":  123,
  "classification": {
    "domain": "업무",
    "intent": "문의"
  },
  "summary":         "납품 일정 확인 요청 이메일입니다.",
  "schedule_info":   null,
  "meta": {
    "elapsed_ms": 41.39,
    "source":     "consumer.classify"
  }
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| outbox_id | int | 요청의 outbox_id 그대로 |
| email_id | int | 요청의 email_id 그대로 |
| classification.domain | string | 분류된 도메인 |
| classification.intent | string | 분류된 인텐트 |
| summary | string | GPT 요약 |
| schedule_info | object \| null | 일정 정보. 없으면 null. 포함 키는 `date`, `time`, `location` |
| meta.elapsed_ms | float | AI 서버 처리 시간 (ms) |
| meta.source | string | 항상 `"consumer.classify"` |

---

## 5. ack / nack 정책

| 상황 | 처리 |
|---|---|
| 정상 처리 완료 | `ack` |
| JSON 파싱 실패 | `nack(requeue=False)` → DLQ |
| 스키마 검증 실패 | `nack(requeue=False)` → DLQ |
| 일시적 오류 (API 다운 등) | `nack(requeue=True)` → 재시도 |
| 결과 publish 실패 / unroutable | `ack` 금지, `nack(requeue=True)` |

---

## 6. deployment

### 6-1. 요청 — 백엔드/Admin이 exchange `x.app2ai.direct` 로 publish

```text
publish to exchange x.app2ai.direct with routing key deployment
message is routed to queue q.ai.deployment
AI deployment consumer consumes from queue q.ai.deployment
```

실제 RabbitMQ binding 기준:

```text
queue=q.ai.deployment
exchange=x.app2ai.direct
binding_key=deployment
```

```json
{
  "job_id": "deploy-2026-05-04-001",
  "model_version": "training-final-004",
  "artifact_s3_uri": "s3://capstone-gachon/models/training-final-004/",
  "requested_by": "admin",
  "requested_at": "2026-05-04T10:30:00Z"
}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| job_id | string | ✅ | 재배포 Job 식별자. 모든 상태 이벤트에 그대로 보존 |
| model_version | string | ✅ | preload 대상 모델 버전 |
| artifact_s3_uri | string \| null |  | 모델 artifact S3 경로. 현재 AI 서버는 `model_version` 기반 ModelManager preload 로직을 재사용한다 |
| requested_by | string \| null |  | 요청자 |
| requested_at | string \| null |  | 요청 시각 |

처리 순서:

```text
ModelManager.preload(model_version)
→ ModelManager.validate()
→ ModelManager.switch()
```

- MQ consumer는 HTTP `/deployment/preload`, `/deployment/validate`, `/deployment/switch`와 같은 `ModelManager` 인스턴스 및 로직을 재사용한다.
- preload 실패 시 `current_bundle`은 유지된다.
- validate 실패 시 `switch`를 수행하지 않는다.
- switch는 `ModelManager` lock 안에서 `current_bundle = staging_bundle` 참조 교체만 수행한다.

### 6-2. 응답 — AI가 default exchange 로 publish

```text
publish to default exchange with routing key q.2app.deployment
message is routed to queue q.2app.deployment
Backend/Admin consumer consumes from queue q.2app.deployment
```

실제 RabbitMQ binding 기준:

```text
queue=q.2app.deployment
exchange=<default>
routing_key=q.2app.deployment
```

#### RUNNING

```json
{
  "job_id": "deploy-2026-05-04-001",
  "status": "RUNNING",
  "model_version": "training-final-004",
  "stage": "PRELOAD",
  "message": "Preloading deployment model",
  "timestamp": "2026-05-04T10:30:01Z"
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| job_id | string | 요청의 job_id 그대로 |
| status | string | 항상 `RUNNING` |
| model_version | string | 요청의 model_version 그대로 |
| stage | string | `PRELOAD`, `VALIDATE`, `SWITCH` 중 하나 |
| message | string | 진행 메시지 |
| timestamp | string | 이벤트 발행 시각 |

#### COMPLETED

```json
{
  "job_id": "deploy-2026-05-04-001",
  "status": "COMPLETED",
  "model_version": "training-final-004",
  "active_model_version": "training-final-004",
  "finished_at": "2026-05-04T10:30:05Z",
  "message": "Deployment completed"
}
```

#### FAILED

```json
{
  "job_id": "deploy-2026-05-04-001",
  "status": "FAILED",
  "model_version": "training-final-004",
  "stage": "VALIDATE",
  "error_message": "validation failed",
  "finished_at": "2026-05-04T10:30:05Z"
}
```

- 실패 시 `FAILED` 이벤트를 반드시 publish한다.
- 성공 시 `COMPLETED` 이벤트를 반드시 publish한다.

---

## 7. training

### 7-1. 상태 이벤트 — AI training 컨테이너가 `q.2app.training` 으로 publish

`q.2app.training`에는 Admin 서버가 저장/처리할 상태 요약 이벤트만 보낸다.

```json
{
  "job_id": "train-2026-05-04-001",
  "status": "COMPLETED",
  "model_version": "training-final-004",
  "finished_at": "2026-05-04T10:30:05Z",
  "metrics": {
    "intent_f1": 0.91,
    "domain_accuracy": 0.88
  },
  "error_message": null
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| job_id | string | 재학습 Job 식별자 |
| status | string | `RUNNING`, `COMPLETED`, `FAILED` 중 하나 |
| model_version | string \| null | 학습 대상/결과 모델 버전 |
| finished_at | string \| null | `COMPLETED` 또는 `FAILED` 완료 시각. `RUNNING`은 null 가능 |
| metrics.intent_f1 | float \| null | intent 분류 F1 |
| metrics.domain_accuracy | float \| null | domain 분류 accuracy |
| error_message | string \| null | 실패 사유. 성공/진행 중에는 null 가능 |

- `RUNNING`: Job 시작 상태만 전달한다.
- `COMPLETED`: `model_version`, `finished_at`, `metrics`를 포함한다.
- `FAILED`: `finished_at`, `error_message`를 포함하며 반드시 publish한다.
- 긴 학습 로그, 실시간 진행 로그, stdout/stderr 성격 데이터는 이 큐로 보내지 않는다.

### 7-2. SSE 로그 — training 컨테이너가 `x.sse.fanout` 으로 publish

```json
{
  "user_id": "admin",
  "sse_type": "ai-training-updated",
  "data": "[INFO] SBERT 학습 시작"
}
```

- Exchange: `x.sse.fanout`
- Type: `fanout`
- Routing key: 사용하지 않음
- 용도: SSE 구독자/관리자 화면용 실시간 로그 스트림

---

## 8. 구분 규칙

- Queue name: `q.2ai.classify`, `q.2app.classify`
- Queue name: `q.ai.deployment`, `q.2app.deployment`
- Queue name: `q.2app.training`
- Exchange name: `x.sse.fanout`
- Routing key / Binding key: `2ai.classify`, `2app.classify`, `deployment`, `q.2app.deployment`, `q.2app.training`
- Consumer 는 queue 이름으로 consume 한다.
- Publisher 는 exchange + routing key 로 publish 한다.
- `/classify` 경로는 default exchange `""` 를 사용하지 않는다.
- 분류는 SBERT → Domain Logistic Regression → Intent Logistic Regression 구조를 사용하며, LLM을 분류기로 사용하지 않는다.
