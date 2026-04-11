# ============================================================
# classify 비즈니스 로직 — 라우터 / consumer 공용 코어 오케스트레이션
# /classify 와 classify consumer 가 동일한 진입점으로 사용한다.
# ============================================================

from api.schemas import ClassifyRequest, ClassifyResponse, Classification
from api.services.gpt_service import summarize_email

MIN_SUMMARY_LENGTH = 10


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

    # 2. GPT 요약 + 일정 추출
    gpt_result = summarize_email(email_text)
    summary = gpt_result["summary"]
    schedule_info = gpt_result["schedule"]

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
        summary=summary,
        schedule_info=schedule_info,
        email_embedding=embedding,
    )
