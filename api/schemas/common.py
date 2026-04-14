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
    비즈니스 로직 오류 시 q.2app.* 로 publish 가능한 공통 에러 응답.
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
