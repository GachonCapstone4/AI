# ============================================================
# 메시지 계약 테스트 (Message Contract Tests)
#
# RabbitMQ / FastAPI 없이 순수 스키마 레벨 검증
#
# 커버 범위
# ----------
# q.2ai.classify  입력 파싱  (ClassifyRequest)
# q.2app.classify 출력 검증 (ClassifyResponse + ResponseMeta)
# ============================================================

import json
import pytest
from pydantic import ValidationError

from api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    Classification,
    ResponseMeta,
)
from messaging.consumer_classify import _build_backend_classify_payload


# ── 픽스처: 표준 입력 메시지 ─────────────────────────────────

@pytest.fixture
def classify_input():
    return {
        "outbox_id":    1,
        "email_id":     1,
        "sender_email": "sender@example.com",
        "sender_name":  "홍길동",
        "subject":      "납품 일정 문의",
        "body_clean":   "이번 달 납품 일정을 알려주시겠어요?",
        "received_at":  "2026-04-06T10:00:00",
    }


@pytest.fixture
def classify_output():
    return {
        "outbox_id":       1,
        "email_id":        1,
        "classification":  {"domain": "업무", "intent": "문의"},
        "summary":         "납품 일정 확인 요청 이메일입니다.",
        "schedule_info":   None,
        "email_embedding": [0.1, 0.2, 0.3],
    }

# ── q.2ai.classify 입력 파싱 ─────────────────────────────────

class TestClassifyInput:
    def test_valid_message_parses(self, classify_input):
        req = ClassifyRequest(**classify_input)
        assert req.outbox_id    == 1
        assert req.email_id     == 1
        assert req.subject      == "납품 일정 문의"
        assert req.body_clean   == "이번 달 납품 일정을 알려주시겠어요?"

    def test_missing_outbox_id_raises(self, classify_input):
        classify_input.pop("outbox_id")
        with pytest.raises(ValidationError):
            ClassifyRequest(**classify_input)

    def test_missing_subject_raises(self, classify_input):
        classify_input.pop("subject")
        with pytest.raises(ValidationError):
            ClassifyRequest(**classify_input)

    def test_missing_body_clean_raises(self, classify_input):
        classify_input.pop("body_clean")
        with pytest.raises(ValidationError):
            ClassifyRequest(**classify_input)

    def test_json_roundtrip(self, classify_input):
        """JSON 직렬화 → 역직렬화 무결성"""
        raw  = json.dumps(classify_input)
        data = json.loads(raw)
        req  = ClassifyRequest(**data)
        assert req.email_id == classify_input["email_id"]

    def test_camel_case_aliases_parse(self):
        req = ClassifyRequest(
            outboxId=7,
            emailId=11,
            senderEmail="sender@example.com",
            senderName="홍길동",
            subject="회의 일정 안내",
            bodyClean="정제된 본문...",
            receivedAt="2026-04-06T10:00:00",
        )
        assert req.outbox_id == 7
        assert req.email_id == 11


# ── q.2app.classify 출력 검증 ────────────────────────────────

class TestClassifyOutput:
    def test_valid_response_parses(self, classify_output):
        resp = ClassifyResponse(**classify_output)
        assert resp.outbox_id == 1
        assert resp.classification.domain == "업무"
        assert resp.classification.intent == "문의"
        assert isinstance(resp.email_embedding, list)
        assert all(isinstance(v, float) for v in resp.email_embedding)

    def test_schedule_info_optional(self, classify_output):
        classify_output["schedule_info"] = None
        resp = ClassifyResponse(**classify_output)
        assert resp.schedule_info is None

    def test_schedule_info_with_dict(self, classify_output):
        classify_output["schedule_info"] = {
            "date": "2026-04-10", "time": "14:00",
            "location": "회의실 A", "attendees": ["홍길동"],
        }
        resp = ClassifyResponse(**classify_output)
        assert resp.schedule_info["date"] == "2026-04-10"

    def test_embedding_must_be_float_list(self, classify_output):
        classify_output["email_embedding"] = [0.1, 0.2, 0.3]
        resp = ClassifyResponse(**classify_output)
        assert len(resp.email_embedding) == 3

    def test_outbox_id_preserved(self, classify_output):
        """outbox_id 가 입력과 동일하게 출력에 포함되어야 함"""
        resp = ClassifyResponse(**classify_output)
        assert resp.outbox_id == classify_output["outbox_id"]

    def test_json_serializable(self, classify_output):
        resp = ClassifyResponse(**classify_output)
        dumped = resp.model_dump()
        raw = json.dumps(dumped)            # JSON 직렬화 가능해야 함
        assert "outbox_id" in json.loads(raw)

# ── ResponseMeta 검증 ────────────────────────────────────────

class TestResponseMeta:
    def test_valid_meta_parses(self):
        meta = ResponseMeta(elapsed_ms=123.4, source="consumer.classify")
        assert meta.elapsed_ms == 123.4
        assert meta.source     == "consumer.classify"

    def test_source_defaults_to_ai_server(self):
        meta = ResponseMeta(elapsed_ms=50.0)
        assert meta.source == "ai-server"

    def test_meta_embedded_in_classify_response(self, classify_output):
        classify_output["meta"] = {"elapsed_ms": 99.9, "source": "consumer.classify"}
        resp = ClassifyResponse(**classify_output)
        assert resp.meta is not None
        assert resp.meta.elapsed_ms == 99.9
        assert resp.meta.source     == "consumer.classify"

    def test_meta_absent_is_none(self, classify_output):
        resp = ClassifyResponse(**classify_output)
        assert resp.meta is None

    def test_meta_json_serializable(self, classify_output):
        classify_output["meta"] = {"elapsed_ms": 77.0, "source": "consumer.classify"}
        resp = ClassifyResponse(**classify_output)
        dumped = json.dumps(resp.model_dump())
        parsed = json.loads(dumped)
        assert parsed["meta"]["elapsed_ms"] == 77.0


class TestBackendClassifyPublishPayload:
    def test_builds_flat_backend_contract_with_defaults(self):
        result = ClassifyResponse(
            outbox_id=1,
            email_id=2,
            classification=Classification(domain="업무", intent="문의"),
            summary="납품 일정 확인 요청 이메일입니다.",
            schedule_info={"date": "2026-04-10", "time": "14:00"},
            email_embedding=[0.1, 0.2, 0.3],
            meta=ResponseMeta(elapsed_ms=12.3, source="consumer.classify"),
        )

        payload = _build_backend_classify_payload(result)

        assert payload == {
            "outbox_id": 1,
            "email_id": 2,
            "domain": "업무",
            "intent": "문의",
            "confidence_score": 0.0,
            "summary_text": "납품 일정 확인 요청 이메일입니다.",
            "schedule_detected": True,
            "email_embedding": [0.1, 0.2, 0.3],
            "entities_json": '{"date": "2026-04-10", "time": "14:00"}',
            "model_version": "unknown",
        }

    def test_builds_empty_schedule_defaults(self):
        result = ClassifyResponse(
            outbox_id=1,
            email_id=2,
            classification=Classification(domain="업무", intent="문의"),
            summary="요약",
            schedule_info=None,
            email_embedding=[0.1],
        )

        payload = _build_backend_classify_payload(result)

        assert payload["schedule_detected"] is False
        assert payload["entities_json"] == "{}"
