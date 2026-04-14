from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from src.settings import get_settings


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
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

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

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
        temperature: float = 0.1,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.truncate_input(user_prompt)},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens or self._max_output_tokens,
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty response.")
        return content.strip()


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
