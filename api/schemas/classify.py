from pydantic import BaseModel
from typing import Any, List, Optional

from api.schemas.common import ResponseMeta


class ClassifyRequest(BaseModel):
    outbox_id: int
    email_id: int
    sender_email: str
    sender_name: str
    subject: str
    body_clean: str
    received_at: Any


class Classification(BaseModel):
    domain: str
    intent: str


class ClassifyResponse(BaseModel):
    outbox_id: int
    email_id: int
    classification: Classification
    summary: str
    schedule_info: Optional[dict] = None
    email_embedding: List[float]
    meta: Optional[ResponseMeta] = None
