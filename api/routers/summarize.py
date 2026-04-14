# ============================================================
# /summarize 엔드포인트 - LLM 요약 + 일정 추출 (보조 엔드포인트)
# ============================================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.schemas import SummarizeResponse
from api.services.summarize_service import summarize_email

router = APIRouter()


class SummarizeRequest(BaseModel):
    emailId: str
    subject: str
    body: str


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(payload: SummarizeRequest):
    try:
        email_text = f"{payload.subject}\n{payload.body}".strip()
        result = summarize_email(email_text)

        return SummarizeResponse(
            emailId=payload.emailId,
            summary=result["summary"],
            schedule=result["schedule"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
