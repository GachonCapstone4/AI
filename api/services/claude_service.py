# ============================================================
# Claude 연동 서비스 - 답장 초안 생성
# /draft 엔드포인트에서 호출
# 내부 실험 / fallback 용 deprecated candidate 경로
# TODO: draft 가 RAG 서버로 이전되면 이 로컬 생성기는 대체 또는 제거 대상이다.
# MOCK_MODE=True → 실제 API 호출 없이 테스트 가능
# ============================================================

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"


def _get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
    return anthropic.Anthropic(api_key=api_key)


def _mock_response(domain: str, intent: str, mode: str) -> str:
    return f"""[MOCK] 답장 초안

안녕하세요.

문의하신 {intent} 관련하여 답변 드립니다.
도메인: {domain} | 모드: {mode}

확인 후 빠른 시일 내에 처리해 드리겠습니다.

감사합니다.
[MOCK 응답 - 실제 Claude API 호출 아님]"""


def _build_prompt(
    subject        : str,
    body           : str,
    domain         : str,
    intent         : str,
    summary        : str,
    mode           : str,
    previous_draft : str = "",
) -> str:
    mode_section = ""
    if mode == "regenerate" and previous_draft:
        mode_section = f"\n[이전 초안]\n{previous_draft}\n\n이전 초안을 개선하여 다시 작성해주세요."

    return f"""당신은 한국 비즈니스 이메일 답장 작성 전문가입니다.
아래 정보를 바탕으로 적절한 답장 초안을 작성해주세요.
{mode_section}

[수신 이메일]
제목: {subject}
본문: {body}

[요약]
{summary}

[분류 결과]
도메인: {domain}
인텐트: {intent}

[작성 조건]
- 인사말과 맺음말 포함
- 비즈니스 맥락에 맞는 전문적인 표현 사용
- 불필요한 내용 없이 간결하게 작성
- 답장 본문만 출력 (설명이나 부가 텍스트 없이)
"""


def generate_draft(
    subject        : str,
    body           : str,
    domain         : str,
    intent         : str,
    summary        : str = "",
    mode           : str = "generate",
    previous_draft : str = "",
) -> str:
    if MOCK_MODE:
        print("[Claude Service] MOCK 모드 - 실제 API 호출 없음")
        return _mock_response(domain, intent, mode)

    try:
        client = _get_client()
        prompt = _build_prompt(subject, body, domain, intent, summary, mode, previous_draft)

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text.strip()

    except Exception as e:
        print(f"[Claude Service] API 호출 오류: {e}")
        raise
