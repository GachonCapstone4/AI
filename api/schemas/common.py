from pydantic import BaseModel
from typing import Optional


class ResponseMeta(BaseModel):
    """
    선택적 메타 블록 — consumer 가 populate, HTTP 라우터는 null.
    백엔드 SLA 모니터링 / 디버깅 용도.
    """

    elapsed_ms: float
    source: str = "ai-server"


class ErrorResponse(BaseModel):
    """
    비즈니스 로직 오류 시 q.2app.* 로 publish 되는 에러 응답.
    request_id / emailId 는 원본 메시지 그대로 보존.

    error_code 목록
    ---------------
    VALIDATION_ERROR   : regenerate 시 previous_draft 누락 등 입력 오류
    PROCESSING_ERROR   : Claude/GPT API 실패 등 일시적 처리 오류
    """

    request_id: str
    emailId: str
    status: str = "error"
    error_code: str
    error_message: str
    meta: Optional[ResponseMeta] = None


class SummarizeResponse(BaseModel):
    emailId: str
    summary: str
    schedule: Optional[dict] = None
