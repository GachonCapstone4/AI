from pydantic import BaseModel
from typing import List, Optional

from api.schemas.common import ResponseMeta


class DraftRequest(BaseModel):
    request_id: str
    mode: str
    emailId: str
    subject: str
    body: str
    domain: str
    intent: str
    summary: str
    previous_draft: Optional[str] = None


class DraftResponse(BaseModel):
    request_id: str
    emailId: str
    draft_reply: str
    reply_embedding: List[float]
    meta: Optional[ResponseMeta] = None
