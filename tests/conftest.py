# ============================================================
# pytest 공통 fixture — 경로별 pipeline mock
# lifespan 우회: app.state.classify_pipeline 을 직접 주입
# ============================================================

import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.routers import classify, summarize


def _make_mock_classify_pipeline():
    sbert = MagicMock()
    # encode() → numpy array (shape: (1, 3))  → [0].tolist() 가 동작해야 함
    sbert.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    return {
        "model": {"sbert": sbert},
        "predict": lambda email_text, pipeline: {
            "domain": "업무",
            "intent": "문의",
            "confidence_score": 0.91,
        },
    }

@pytest.fixture
def app_client():
    """lifespan 없이 라우터만 등록한 테스트용 FastAPI 앱"""
    test_app = FastAPI()
    test_app.include_router(classify.router)
    test_app.include_router(summarize.router)
    test_app.state.classify_pipeline = _make_mock_classify_pipeline()

    with TestClient(test_app) as client:
        yield client
