# ============================================================
# /classify 엔드포인트 테스트
# ============================================================

import pytest
from unittest.mock import patch

from api.services.llm_client import LLMPermanentError


BASE_PAYLOAD = {
    "outbox_id":    1,
    "email_id":     1,
    "sender_email": "sender@example.com",
    "sender_name":  "홍길동",
    "subject":      "세금계산서 발행 요청",
    "body_clean":   "지난달 납품 건에 대한 세금계산서 발행 부탁드립니다.",
    "received_at":  "2026-04-06T10:00:00",
}


def _mock_summarize(email_text: str, base_datetime=None) -> dict:
    return {"summary": "세금계산서 발행 요청 건입니다.", "schedule": None}


def _mock_summarize_short(email_text: str, base_datetime=None) -> dict:
    return {"summary": "짧음", "schedule": None}  # len < 10 → fallback


def _mock_summarize_empty(email_text: str, base_datetime=None) -> dict:
    return {"summary": "", "schedule": None}  # 빈 summary → fallback


def _mock_summarize_with_schedule(email_text: str, base_datetime=None) -> dict:
    return {
        "summary": "회의 일정 안내 메일입니다.",
        "schedule": {
            "date_text": "다음주 화요일",
            "time_text": "오후 2시",
            "location": "회의실 A",
            "attendees": ["홍길동"],
        },
    }


def _mock_summarize_429(email_text: str, base_datetime=None) -> dict:
    raise LLMPermanentError("LLM request rejected permanently: 429 Daily Quota Exceeded")


class TestClassifySuccess:
    def test_returns_summary_and_schedule_when_llm_succeeds(self, app_client):
        with patch("api.services.classify_service.summarize_email", side_effect=_mock_summarize_with_schedule):
            with patch("api.services.classify_service.parse_datetime_kst", return_value=("2026-04-14", "14:00")):
                resp = app_client.post("/classify", json=BASE_PAYLOAD)

        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] == "회의 일정 안내 메일입니다."
        assert data["schedule_info"] == {
            "date": "2026-04-14",
            "time": "14:00",
            "location": "회의실 A",
            "attendees": ["홍길동"],
        }

    def test_returns_required_fields(self, app_client):
        with patch("api.services.classify_service.summarize_email", side_effect=_mock_summarize):
            resp = app_client.post("/classify", json=BASE_PAYLOAD)

        assert resp.status_code == 200
        data = resp.json()
        assert data["outbox_id"] == 1
        assert data["email_id"] == 1
        assert "classification" in data
        assert "domain" in data["classification"]
        assert "intent" in data["classification"]
        assert isinstance(data["email_embedding"], list)
        assert len(data["email_embedding"]) > 0

    def test_embedding_uses_summary_when_valid(self, app_client):
        with patch("api.services.classify_service.summarize_email", side_effect=_mock_summarize) as mock_sum:
            resp = app_client.post("/classify", json=BASE_PAYLOAD)

        assert resp.status_code == 200

    def test_embedding_fallback_on_empty_summary(self, app_client):
        """summary가 빈 문자열이면 email_text 로 fallback"""
        with patch("api.services.classify_service.summarize_email", side_effect=_mock_summarize_empty):
            resp = app_client.post("/classify", json=BASE_PAYLOAD)

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["email_embedding"], list)

    def test_fallbacks_when_llm_429_occurs(self, app_client):
        with patch("api.services.classify_service.summarize_email", side_effect=_mock_summarize_429):
            resp = app_client.post("/classify", json=BASE_PAYLOAD)

        assert resp.status_code == 200
        data = resp.json()
        assert data["classification"]["domain"] == "업무"
        assert data["classification"]["intent"] == "문의"
        assert data["summary"] == "요약 생성 실패"
        assert data["schedule_info"] is None

    def test_embedding_fallback_on_short_summary(self, app_client):
        """summary 길이가 MIN_SUMMARY_LENGTH(10) 미만이면 email_text 로 fallback"""
        with patch("api.services.classify_service.summarize_email", side_effect=_mock_summarize_short):
            resp = app_client.post("/classify", json=BASE_PAYLOAD)

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["email_embedding"], list)


class TestClassifyValidation:
    def test_missing_subject_returns_422(self, app_client):
        payload = {k: v for k, v in BASE_PAYLOAD.items() if k != "subject"}
        resp = app_client.post("/classify", json=payload)
        assert resp.status_code == 422

    def test_missing_body_returns_422(self, app_client):
        payload = {k: v for k, v in BASE_PAYLOAD.items() if k != "body_clean"}
        resp = app_client.post("/classify", json=payload)
        assert resp.status_code == 422
