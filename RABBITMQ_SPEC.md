# RabbitMQ 연동 스펙

> 최종 수정: 2026-04-11
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

| Queue | Exchange | Binding Key | 방향 |
|---|---|---|---|
| `q.2ai.classify` | x.app2ai.direct | 2ai.classify | 백엔드 → AI |
| `q.2app.classify` | x.ai2app.direct | 2app.classify | AI → 백엔드 |

- 모든 Exchange / Queue: `durable=true`, `delivery_mode=2` (persistent)
- `content_type`: `application/json`, 인코딩: UTF-8

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
  "email_embedding": [0.123, 0.456, ...],
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
| email_embedding | float[] | SBERT 임베딩 벡터 |
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

## 6. 구분 규칙

- Queue name: `q.2ai.classify`, `q.2app.classify`
- Routing key / Binding key: `2ai.classify`, `2app.classify`
- Consumer 는 queue 이름으로 consume 한다.
- Publisher 는 exchange + routing key 로 publish 한다.
- `/classify` 경로는 default exchange `""` 를 사용하지 않는다.
