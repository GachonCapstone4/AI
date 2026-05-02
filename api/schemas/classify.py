from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing import Any, List, Optional

from api.schemas.common import ResponseMeta


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    outbox_id: int = Field(validation_alias=AliasChoices("outbox_id", "outboxId"))
    email_id: int = Field(validation_alias=AliasChoices("email_id", "emailId"))
    sender_email: str = Field(validation_alias=AliasChoices("sender_email", "senderEmail"))
    sender_name: str = Field(validation_alias=AliasChoices("sender_name", "senderName"))
    subject: str
    body_clean: str = Field(validation_alias=AliasChoices("body_clean", "bodyClean", "body"))
    received_at: Any = Field(validation_alias=AliasChoices("received_at", "receivedAt"))


class Classification(BaseModel):
    domain: str
    intent: str


class ClassifyResponse(BaseModel):
    outbox_id: int
    email_id: int
    classification: Classification
    confidence_score: float
    summary: str
    schedule_info: Optional[dict] = None
    email_embedding: List[float]
    meta: Optional[ResponseMeta] = None
