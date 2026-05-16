from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import deployment


class _FakeModelManager:
    def __init__(self) -> None:
        self.calls = []

    def preload(self, model_version: str | None = None) -> dict:
        self.calls.append(("preload", model_version))
        return {
            "status": "preloaded",
            "model_version": model_version,
            "artifact_s3_uri": f"s3://bucket/models/{model_version}/",
        }

    def validate(self) -> dict:
        self.calls.append(("validate", None))
        return {
            "status": "validated",
            "model_version": "training-final-004",
            "samples": [],
        }

    def switch(self) -> dict:
        self.calls.append(("switch", None))
        return {
            "status": "switched",
            "model_version": "training-final-004",
        }


def _client_with_manager(manager: _FakeModelManager) -> TestClient:
    app = FastAPI()
    app.include_router(deployment.router)
    app.state.model_manager = manager
    return TestClient(app)


def test_http_deployment_preload_validate_switch_still_work(monkeypatch):
    monkeypatch.setattr(deployment, "_safe_publish_sse_log", lambda _message: None)
    manager = _FakeModelManager()
    client = _client_with_manager(manager)

    preload = client.post(
        "/deployment/preload",
        json={"modelVersion": "training-final-004"},
    )
    validate = client.post("/deployment/validate")
    switch = client.post("/deployment/switch")

    assert preload.status_code == 200
    assert preload.json()["modelVersion"] == "training-final-004"
    assert validate.status_code == 200
    assert validate.json()["status"] == "validated"
    assert switch.status_code == 200
    assert switch.json()["activeModelVersion"] == "training-final-004"
    assert manager.calls == [
        ("preload", "training-final-004"),
        ("validate", None),
        ("switch", None),
    ]
