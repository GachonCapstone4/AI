# ============================================================
# /summarize 엔드포인트 - LLM 요약 + 일정 추출 (보조 엔드포인트)
# ============================================================

import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.schemas import SummarizeResponse
from api.services.summarize_service import parse_datetime_kst, summarize_email

router = APIRouter()


class SummarizeRequest(BaseModel):
    emailId: str
    subject: str
    body: str


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(payload: SummarizeRequest):
    try:
        email_text = f"{payload.subject}\n{payload.body}".strip()
        base_datetime = datetime.now(ZoneInfo("Asia/Seoul"))
        result = summarize_email(email_text, base_datetime)
        raw_schedule = result.get("schedule")
        final_schedule = None

        if raw_schedule is not None:
            if raw_schedule:
                raw_schedule.pop("attendees", None)
            date_text = raw_schedule.get("date_text")
            time_text = raw_schedule.get("time_text")
            date, time = parse_datetime_kst(date_text, time_text, base_datetime)
            final_schedule = {
                "date": date,
                "time": time,
                "location": raw_schedule.get("location"),
            }

        return SummarizeResponse(
            emailId=payload.emailId,
            summary=result["summary"],
            schedule=final_schedule,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
