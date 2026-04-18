import json
from datetime import datetime
from datetime import timedelta
import re
from zoneinfo import ZoneInfo

from api.services.llm_client import get_llm_client
from messaging.structured_log import get_logger

SUMMARIZE_SYSTEM_PROMPT = (
    "당신은 비즈니스 이메일 분석 전문가입니다. "
    "요청된 JSON 형식으로만 응답합니다."
)
log = get_logger("api.services.summarize_service")


def _extract_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.startswith("```")]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def _build_prompt(email_text: str, base_datetime) -> str:
    base_text = str(base_datetime) if base_datetime is not None else "unknown"
    return f"""
다음 비즈니스 이메일을 분석하여 아래 JSON 형식으로만 응답하세요.
다른 설명이나 마크다운 없이 JSON만 출력하세요.

[이메일 수신 시각]
{base_text}

[추출 규칙]
- 날짜와 시간은 절대 계산하지 말고 이메일 원문에 나온 표현 그대로 추출하세요.
- "다음주 화요일", "내일 오후 2시", "3월 28일", "14:00" 같은 표현은 정규화하지 말고 그대로 반환하세요.
- location은 Zoom, Teams, 회의실, 본사, 카페, 주소, 온라인 회의 링크 등 명시된 장소만 추출하세요.
- attendees는 이메일 본문에 명시된 이름만 추출하세요.
- 값이 없으면 location은 null, attendees는 []로 설정하세요.

[이메일 내용]
{email_text}

[출력 형식]
{{
  "summary": "이메일 핵심 내용을 1~2문장으로 요약",
  "schedule": {{
    "date_text": "다음주 화요일 또는 3월 28일 또는 null",
    "time_text": "오후 2시 또는 14:00 또는 null",
    "location": "장소 (없으면 null)",
    "attendees": ["참석자 목록"]
  }}
}}

일정 정보가 전혀 없으면 schedule 전체를 null로 설정하세요.
""".strip()


def parse_datetime_kst(date_text, time_text, base_datetime):
    try:
        import dateparser
    except ImportError:
        text = " ".join(part for part in (date_text, time_text) if part)
        log.warning(
            "schedule_datetime_parse_skipped_missing_dependency",
            date_text=date_text,
            time_text=time_text,
            base_datetime=str(base_datetime) if base_datetime is not None else None,
            joined_text=text,
            timezone="Asia/Seoul",
        )
        return None, None

    if not date_text and not time_text:
        return None, None

    base_dt = _coerce_base_datetime(base_datetime)
    text = " ".join(part for part in (date_text, time_text) if part)
    fallback_dt = _parse_korean_datetime_fallback(date_text, time_text, base_dt)

    dt = dateparser.parse(
        text,
        languages=["ko"],
        settings={
            "RELATIVE_BASE": base_dt,
            "TIMEZONE": "Asia/Seoul",
            "RETURN_AS_TIMEZONE_AWARE": True,
        },
    )

    if fallback_dt is not None:
        return fallback_dt.strftime("%Y-%m-%d"), fallback_dt.strftime("%H:%M")

    if not dt:
        log.warning(
            "schedule_datetime_parse_failed",
            date_text=date_text,
            time_text=time_text,
            base_datetime=base_dt.isoformat(),
            joined_text=text,
            timezone="Asia/Seoul",
        )
        return None, None

    dt_kst = dt.astimezone(ZoneInfo("Asia/Seoul"))
    return dt_kst.strftime("%Y-%m-%d"), dt_kst.strftime("%H:%M")


def _parse_korean_datetime_fallback(date_text, time_text, base_dt: datetime) -> datetime | None:
    date_part = _parse_korean_date_text(date_text, base_dt)
    time_part = _parse_korean_time_text(time_text)

    if date_part is None and time_part is None:
        return None

    target = date_part or base_dt
    hour = time_part.hour if time_part is not None else 0
    minute = time_part.minute if time_part is not None else 0
    return target.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _parse_korean_date_text(date_text, base_dt: datetime) -> datetime | None:
    if not date_text:
        return None

    text = str(date_text).strip()
    if not text:
        return None

    weekday_match = re.fullmatch(r"다음\s*주\s*([월화수목금토일])요일?", text)
    if weekday_match:
        weekday_map = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
        monday = base_dt - timedelta(days=base_dt.weekday())
        target = monday + timedelta(days=7 + weekday_map[weekday_match.group(1)])
        return target

    month_day_match = re.fullmatch(r"(\d{1,2})월\s*(\d{1,2})일", text)
    if month_day_match:
        month = int(month_day_match.group(1))
        day = int(month_day_match.group(2))
        try:
            return base_dt.replace(month=month, day=day)
        except ValueError:
            return None

    return None


def _parse_korean_time_text(time_text) -> datetime | None:
    if not time_text:
        return None

    text = str(time_text).strip()
    if not text:
        return None

    hm_match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if hm_match:
        hour = int(hm_match.group(1))
        minute = int(hm_match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return datetime(2000, 1, 1, hour, minute)
        return None

    meridiem_match = re.fullmatch(r"(오전|오후)\s*(\d{1,2})시(?:\s*(\d{1,2})분)?", text)
    if meridiem_match:
        meridiem = meridiem_match.group(1)
        hour = int(meridiem_match.group(2))
        minute = int(meridiem_match.group(3) or 0)
        if not 1 <= hour <= 12 or not 0 <= minute <= 59:
            return None
        if meridiem == "오전":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
        return datetime(2000, 1, 1, hour, minute)

    return None


def _coerce_base_datetime(base_datetime) -> datetime:
    kst = ZoneInfo("Asia/Seoul")

    if isinstance(base_datetime, datetime):
        dt = base_datetime
    else:
        value = str(base_datetime).strip() if base_datetime is not None else ""
        if not value:
            return datetime.now(kst)

        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(kst)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=kst)
    return dt.astimezone(kst)


def summarize_email(email_text: str, base_datetime=None) -> dict:
    client = get_llm_client()
    raw = client.chat(
        system_prompt=SUMMARIZE_SYSTEM_PROMPT,
        user_prompt=_build_prompt(email_text, base_datetime),
        max_output_tokens=400,
        temperature=0.1,
    )

    try:
        result = _extract_json(raw)
        schedule = result.get("schedule", None)
        if isinstance(schedule, dict):
            schedule = {
                "date_text": schedule.get("date_text"),
                "time_text": schedule.get("time_text"),
                "location": schedule.get("location"),
                "attendees": schedule.get("attendees") or [],
            }
        return {
            "summary": result.get("summary", ""),
            "schedule": schedule,
        }
    except json.JSONDecodeError:
        return {"summary": raw, "schedule": None}
