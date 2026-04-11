# ============================================================
# draft 비즈니스 로직 — 라우터 / consumer 공용 보조 경로
# 내부 실험 / fallback 용 deprecated candidate 경로.
# classify 오케스트레이션과 분리된 내부 지원 책임만 가진다.
# ============================================================

from api.schemas import DraftRequest, DraftResponse
from api.services.claude_service import generate_draft


def run_draft(payload: DraftRequest, pipeline: dict) -> DraftResponse:
    """
    draft 내부/보조 경로 전용 처리.
    공식 코어 계약이 아니므로 미래 이전 시 이 함수 경계를 우선 분리 대상으로 본다.

    Parameters
    ----------
    payload  : DraftRequest (pydantic)
    pipeline : {"model": {"sbert": SentenceTransformer}}

    Returns
    -------
    DraftResponse (pydantic)

    Raises
    ------
    ValueError
        mode=regenerate 인데 previous_draft 가 없는 경우
        → consumer: publish error + ack  /  router: HTTP 422
    """
    if payload.mode == "regenerate" and not payload.previous_draft:
        raise ValueError("mode=regenerate 일 때 previous_draft 는 필수입니다.")

    # TODO: draft 가 RAG 서버로 이전되면 이 함수는 로컬 초안 생성 대신
    # 전송/검증 경계로 축소하거나 제거하는 우선 후보이다.
    draft_reply = generate_draft(
        subject=payload.subject,
        body=payload.body,
        domain=payload.domain,
        intent=payload.intent,
        summary=payload.summary,
        mode=payload.mode,
        previous_draft=payload.previous_draft or "",
    )

    reply_embedding = pipeline["model"]["sbert"].encode(
        [draft_reply], normalize_embeddings=True
    )[0].tolist()

    return DraftResponse(
        request_id=payload.request_id,
        emailId=payload.emailId,
        draft_reply=draft_reply,
        reply_embedding=reply_embedding,
    )
