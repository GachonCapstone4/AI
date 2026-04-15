from __future__ import annotations

from functools import lru_cache
import json

import requests
from openai import OpenAI

from messaging.structured_log import get_logger
from src.settings import get_settings


class LLMTransientError(RuntimeError):
    pass


class LLMPermanentError(RuntimeError):
    pass


log = get_logger("api.services.llm_client")


def _mask_secret(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return f"{value[:2]}***"
    return f"{value[:4]}...{value[-4:]}"


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        model: str,
        base_url: str | None,
        max_input_chars: int,
        max_output_tokens: int,
    ) -> None:
        normalized_base_url = (base_url or "").rstrip("/")
        if normalized_base_url and not normalized_base_url.endswith("/v1"):
            normalized_base_url = f"{normalized_base_url}/v1"

        self._api_key = api_key.strip().strip("'\"")
        self._base_url = normalized_base_url
        client_kwargs = {"api_key": self._api_key}
        if normalized_base_url:
            client_kwargs["base_url"] = normalized_base_url

        self._client = OpenAI(**client_kwargs)
        self._provider = provider
        self._model = model
        self._max_input_chars = max_input_chars
        self._max_output_tokens = max_output_tokens

    @property
    def provider(self) -> str:
        return self._provider

    def truncate_input(self, text: str, *, max_chars: int | None = None) -> str:
        limit = max_chars or self._max_input_chars
        if len(text) <= limit:
            return text

        truncated = text[:limit]
        return (
            f"{truncated}\n\n"
            "[TRUNCATED]\n"
            "Input was truncated to respect configured LLM input limits."
        )

    def _request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _masked_request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {_mask_secret(self._api_key)}",
            "Content-Type": "application/json",
        }

    def _request_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None,
        temperature: float,
    ) -> dict:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.truncate_input(user_prompt)},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens or self._max_output_tokens,
            "stream": False,
        }

    def _chat_via_requests(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None,
        temperature: float,
    ) -> str:
        if not self._base_url:
            raise LLMPermanentError("LLM base_url is required for provider requests flow.")

        url = f"{self._base_url}/chat/completions"
        payload = self._request_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        headers = self._request_headers()
        prepared_request = requests.Request(
            method="POST",
            url=url,
            headers=headers,
            json=payload,
        ).prepare()
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        log.info(
            "llm_outgoing_request",
            provider=self._provider,
            method=prepared_request.method,
            url=prepared_request.url,
            model=self._model,
            api_key_present=bool(self._api_key),
            auth_header_has_bearer_prefix=headers["Authorization"].startswith("Bearer "),
            auth_header_masked=self._masked_request_headers()["Authorization"],
            headers=self._masked_request_headers(),
            payload=payload,
            payload_json=payload_json,
            curl_repro=(
                f"curl {prepared_request.url} "
                '-H "Authorization: Bearer <MASKED_API_KEY>" '
                '-H "Content-Type: application/json" '
                f"-d '{payload_json}'"
            ),
        )

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(5, 60),
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise LLMTransientError(f"LLM request failed transiently: {exc}") from exc
        except requests.RequestException as exc:
            raise LLMTransientError(f"LLM request failed: {exc}") from exc

        if response.status_code in {400, 401, 403}:
            raise LLMPermanentError(
                "LLM request rejected permanently: "
                f"{response.status_code} {response.text} "
                f"(provider={self._provider}, base_url={self._base_url}, "
                f"model={self._model}, api_key={_mask_secret(self._api_key)})"
            )
        if response.status_code >= 500:
            raise LLMTransientError(
                f"LLM upstream server error: {response.status_code} {response.text}"
            )
        if response.status_code >= 400:
            raise LLMPermanentError(
                f"LLM client error: {response.status_code} {response.text}"
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMPermanentError(f"LLM response format invalid: {exc}") from exc

        if not content:
            raise LLMPermanentError("LLM returned an empty response.")
        return content.strip()

    def _chat_via_openai(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None,
        temperature: float,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.truncate_input(user_prompt)},
                ],
                temperature=temperature,
                max_tokens=max_output_tokens or self._max_output_tokens,
                extra_headers=self._request_headers(),
            )
        except Exception as exc:
            message = str(exc)
            if any(code in message for code in ("400", "401", "403", "Invalid API Key")):
                raise LLMPermanentError(f"LLM request rejected permanently: {message}") from exc
            if any(token in message.lower() for token in ("timeout", "timed out", "connection")):
                raise LLMTransientError(f"LLM request failed transiently: {message}") from exc
            raise

        content = response.choices[0].message.content
        if not content:
            raise LLMPermanentError("LLM returned an empty response.")
        return content.strip()

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
        temperature: float = 0.1,
    ) -> str:
        return self._chat_via_requests(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )


@lru_cache
def get_llm_client() -> OpenAICompatibleLLMClient:
    settings = get_settings()
    llm_config = settings.resolve_llm_config()
    return OpenAICompatibleLLMClient(
        provider=llm_config.provider,
        api_key=llm_config.api_key,
        model=llm_config.model,
        base_url=llm_config.base_url,
        max_input_chars=settings.LLM_MAX_INPUT_CHARS,
        max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
    )
