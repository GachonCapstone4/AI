# ============================================================
# pytest 공통 fixture — lifespan 우회: app.state.model_manager 직접 주입
# ============================================================

import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.routers import classify, summarize


def _make_mock_model_manager():
    sbert = MagicMock()
    # encode() → numpy array (shape: (1, 3))  → [0].tolist() 가 동작해야 함
    sbert.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    class MockModelManager:
        current_model_version = "test-model"

        def __init__(self):
            self.current_bundle = {"sbert": sbert, "runtime": {"active_model_version": "test-model"}}

        def predict(self, _email_text):
            return {
                "domain": "업무",
                "intent": "문의",
                "confidence_score": 0.91,
            }

    return MockModelManager()

@pytest.fixture
def app_client():
    """lifespan 없이 라우터만 등록한 테스트용 FastAPI 앱"""
    test_app = FastAPI()
    test_app.include_router(classify.router)
    test_app.include_router(summarize.router)
    test_app.state.model_manager = _make_mock_model_manager()

    with TestClient(test_app) as client:
        yield client
