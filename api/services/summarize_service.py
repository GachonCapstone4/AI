import json

from api.services.llm_client import get_llm_client

SUMMARIZE_SYSTEM_PROMPT = (
    "당신은 비즈니스 이메일 분석 전문가입니다. "
    "요청된 JSON 형식으로만 응답합니다."
)


def _extract_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.startswith("```")]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def _build_prompt(email_text: str) -> str:
    return f"""
다음 비즈니스 이메일을 분석하여 아래 JSON 형식으로만 응답하세요.
다른 설명이나 마크다운 없이 JSON만 출력하세요.

[이메일 내용]
{email_text}

[출력 형식]
{{
  "summary": "이메일 핵심 내용을 1~2문장으로 요약",
  "schedule": {{
    "date": "YYYY-MM-DD 형식 (없으면 null)",
    "time": "HH:MM 형식 (없으면 null)",
    "location": "장소 (없으면 null)",
    "attendees": ["참석자 목록 (없으면 빈 배열)"]
  }}
}}

일정 정보가 전혀 없으면 schedule 전체를 null로 설정하세요.
""".strip()


def summarize_email(email_text: str) -> dict:
    client = get_llm_client()
    raw = client.chat(
        system_prompt=SUMMARIZE_SYSTEM_PROMPT,
        user_prompt=_build_prompt(email_text),
        max_output_tokens=400,
        temperature=0.1,
    )

    try:
        result = _extract_json(raw)
        return {
            "summary": result.get("summary", ""),
            "schedule": result.get("schedule", None),
        }
    except json.JSONDecodeError:
        return {"summary": raw, "schedule": None}
