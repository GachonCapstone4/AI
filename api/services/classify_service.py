# ============================================================
# classify 비즈니스 로직 — 라우터 / consumer 공용 코어 오케스트레이션
# /classify 와 classify consumer 가 동일한 진입점으로 사용한다.
# ============================================================

from api.schemas import ClassifyRequest, ClassifyResponse, Classification
from api.services.llm_client import LLMPermanentError
from api.services.summarize_service import parse_datetime_kst, summarize_email
from messaging.structured_log import get_logger

MIN_SUMMARY_LENGTH = 10
log = get_logger("service.classify")


def _preprocess(subject: str, body: str) -> str:
    return f"{subject}\n{body}".strip()


def run_classify(payload: ClassifyRequest, pipeline: dict) -> ClassifyResponse:
    """
    classify 코어 경로 전용 오케스트레이션.
    AI 서버의 안정적이고 공식적인 계약은 이 진입점을 기준으로 유지한다.

    Parameters
    ----------
    payload  : ClassifyRequest (pydantic)
    pipeline : {"model": {...sbert/clf...}, "predict": predict_email}

    Returns
    -------
    ClassifyResponse (pydantic)
    """
    email_text = _preprocess(payload.subject, payload.body_clean)

    # 1. 도메인 / 인텐트 분류
    result = pipeline["predict"](
        email_text=email_text,
        pipeline=pipeline["model"],
    )
    runtime = pipeline["model"].get("runtime") or {}
    log.info(
        "classification_result",
        outbox_id=payload.outbox_id,
        email_id=payload.email_id,
        loaded_sbert_path=runtime.get("loaded_sbert_path"),
        loaded_domain_model_path=runtime.get("loaded_domain_model_path"),
        loaded_intent_model_path=runtime.get("loaded_intent_model_path"),
        model_source=runtime.get("model_source"),
        active_model_version=runtime.get("active_model_version"),
        metadata_model_version=runtime.get("metadata_model_version"),
        domain_pred=result.get("domain"),
        domain_confidence=result.get("domain_confidence"),
        intent_pred=result.get("intent"),
        intent_confidence=result.get("intent_confidence"),
        final_confidence=result.get("confidence_score"),
        final_confidence_formula="min(domain_confidence, intent_confidence)",
        domain_source=result.get("domain_source"),
        low_confidence=result.get("low_confidence"),
    )

    # 2. LLM 요약 + 일정 추출
    try:
        summarize_result = summarize_email(email_text, payload.received_at)
    except LLMPermanentError as e:
        log.warning(
            "llm_summarize_failed_fallback",
            outbox_id=payload.outbox_id,
            email_id=payload.email_id,
            error=str(e),
        )
        summarize_result = {
            "summary": "요약 생성 실패",
            "schedule": None,
        }

    summary = summarize_result["summary"]
    raw_schedule = summarize_result["schedule"]
    schedule_info = None

    if raw_schedule is not None:
        if raw_schedule:
            raw_schedule.pop("attendees", None)
        date_text = raw_schedule.get("date_text")
        time_text = raw_schedule.get("time_text")
        date, time = parse_datetime_kst(date_text, time_text, payload.received_at)
        schedule_info = {
            "date": date,
            "time": time,
            "location": raw_schedule.get("location"),
        }

    # 3. SBERT 임베딩 — summary 비거나 너무 짧으면 email_text fallback
    embed_text = summary if summary and len(summary) >= MIN_SUMMARY_LENGTH else email_text
    embedding = pipeline["model"]["sbert"].encode(
        [embed_text], normalize_embeddings=True
    )[0].tolist()

    return ClassifyResponse(
        outbox_id=payload.outbox_id,
        email_id=payload.email_id,
        classification=Classification(
            domain=result["domain"],
            intent=result["intent"],
        ),
        confidence_score=result["confidence_score"],
        summary=summary,
        schedule_info=schedule_info,
        email_embedding=embedding,
    )
