import sys
import types

from api.services.llm_client import LLMPermanentError

sys.modules.setdefault(
    "inference",
    types.SimpleNamespace(load_classify_pipeline=lambda: {}, predict_email=lambda **kwargs: {}),
)

from messaging.consumer_classify import _is_permanent_processing_error


def test_daily_quota_exceeded_is_treated_as_permanent_error():
    exc = LLMPermanentError(
        "LLM request rejected permanently: 429 Daily Quota Exceeded"
    )

    assert _is_permanent_processing_error(exc) is True
