from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


def _read_env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None

    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


class LLMResolvedConfig(BaseModel):
    provider: Literal["school", "openai"]
    model: str
    api_key: str
    base_url: str | None = None


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Application/runtime
    APP_ENV: Literal["local", "dev", "test", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = Field(default=8000, ge=1, le=65535)

    # RabbitMQ
    RABBITMQ_URL: str

    # AWS / S3
    AWS_REGION: str = "ap-northeast-2"
    S3_DATASET_BUCKET: str | None = None
    S3_DATASET_PREFIX: str = "datasets"
    S3_MODEL_BUCKET: str | None = None
    S3_MODEL_PREFIX: str = "models"

    # Model runtime
    MODEL_SOURCE: Literal["local", "s3"] = "local"
    ACTIVE_MODEL_VERSION: str | None = None
    MODEL_LOCAL_CACHE_DIR: str = ".cache/model-cache"

    # LLM common
    LLM_PROVIDER: Literal["school", "openai"] = "school"
    LLM_MAX_INPUT_CHARS: int = Field(default=12000, ge=100)
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=800, ge=1)

    # School LLM
    SCHOOL_LLM_BASE_URL: str = "http://cellm.gachon.ac.kr:8000/v1"
    SCHOOL_LLM_API_KEY: str | None = None
    SCHOOL_LLM_MODEL: str | None = None

    # OpenAI
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str | None = None
    OPENAI_BASE_URL: str | None = None

    # Training
    TRAINING_SAFE_MODE: bool = True

    @model_validator(mode="after")
    def validate_runtime_constraints(self) -> "Settings":
        if self.LLM_PROVIDER == "school":
            if not self.SCHOOL_LLM_BASE_URL.startswith(("http://", "https://")):
                raise ValueError("SCHOOL_LLM_BASE_URL must start with http:// or https://")
            if not self.SCHOOL_LLM_API_KEY:
                raise ValueError("SCHOOL_LLM_API_KEY is required when LLM_PROVIDER=school")
            if not self.SCHOOL_LLM_MODEL:
                raise ValueError("SCHOOL_LLM_MODEL is required when LLM_PROVIDER=school")

        if self.LLM_PROVIDER == "openai":
            if not self.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            if not self.OPENAI_MODEL:
                raise ValueError("OPENAI_MODEL is required when LLM_PROVIDER=openai")
            if self.OPENAI_BASE_URL and not self.OPENAI_BASE_URL.startswith(("http://", "https://")):
                raise ValueError("OPENAI_BASE_URL must start with http:// or https://")

        if self.MODEL_SOURCE == "s3":
            if not self.S3_MODEL_BUCKET:
                raise ValueError("S3_MODEL_BUCKET is required when MODEL_SOURCE=s3")
            if not self.ACTIVE_MODEL_VERSION:
                raise ValueError("ACTIVE_MODEL_VERSION is required when MODEL_SOURCE=s3")

        return self

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        values: dict[str, Any] = {
            "APP_ENV": _read_env_str("APP_ENV", "dev"),
            "LOG_LEVEL": _read_env_str("LOG_LEVEL", "INFO"),
            "API_HOST": _read_env_str("API_HOST", "0.0.0.0"),
            "API_PORT": _read_env_str("API_PORT", "8000"),
            "RABBITMQ_URL": _read_env_str("RABBITMQ_URL"),
            "AWS_REGION": _read_env_str("AWS_REGION", "ap-northeast-2"),
            "S3_DATASET_BUCKET": _read_env_str("S3_DATASET_BUCKET"),
            "S3_DATASET_PREFIX": _read_env_str("S3_DATASET_PREFIX", "datasets"),
            "S3_MODEL_BUCKET": _read_env_str("S3_MODEL_BUCKET"),
            "S3_MODEL_PREFIX": _read_env_str("S3_MODEL_PREFIX", "models"),
            "MODEL_SOURCE": _read_env_str("MODEL_SOURCE", "local"),
            "ACTIVE_MODEL_VERSION": _read_env_str("ACTIVE_MODEL_VERSION"),
            "MODEL_LOCAL_CACHE_DIR": _read_env_str("MODEL_LOCAL_CACHE_DIR", ".cache/model-cache"),
            "LLM_PROVIDER": _read_env_str("LLM_PROVIDER", "school"),
            "LLM_MAX_INPUT_CHARS": _read_env_str("LLM_MAX_INPUT_CHARS", "12000"),
            "LLM_MAX_OUTPUT_TOKENS": _read_env_str("LLM_MAX_OUTPUT_TOKENS", "800"),
            "SCHOOL_LLM_BASE_URL": _read_env_str("SCHOOL_LLM_BASE_URL", "http://cellm.gachon.ac.kr:8000/v1"),
            "SCHOOL_LLM_API_KEY": _read_env_str("SCHOOL_LLM_API_KEY"),
            "SCHOOL_LLM_MODEL": _read_env_str("SCHOOL_LLM_MODEL"),
            "OPENAI_API_KEY": _read_env_str("OPENAI_API_KEY"),
            "OPENAI_MODEL": _read_env_str("OPENAI_MODEL"),
            "OPENAI_BASE_URL": _read_env_str("OPENAI_BASE_URL"),
            "TRAINING_SAFE_MODE": _read_env_str("TRAINING_SAFE_MODE", "true"),
        }
        return cls(**values)

    def resolve_llm_config(self) -> LLMResolvedConfig:
        if self.LLM_PROVIDER == "school":
            return LLMResolvedConfig(
                provider="school",
                base_url=self.SCHOOL_LLM_BASE_URL,
                api_key=self.SCHOOL_LLM_API_KEY or "",
                model=self.SCHOOL_LLM_MODEL or "",
            )

        return LLMResolvedConfig(
            provider="openai",
            base_url=self.OPENAI_BASE_URL,
            api_key=self.OPENAI_API_KEY or "",
            model=self.OPENAI_MODEL or "",
        )


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["Invalid application configuration:"]
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        lines.append(f"- {loc}: {error['msg']}")
    return "\n".join(lines)


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings.from_env()
    except ValidationError as exc:
        raise RuntimeError(_format_validation_error(exc)) from exc


def validate_startup_settings() -> Settings:
    return get_settings()
