from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.mlops.training_events import publish_sse_log


router = APIRouter()


class PreloadRequest(BaseModel):
    model_version: str


def _get_model_manager(request: Request):
    manager = getattr(request.app.state, "model_manager", None)
    if manager is None:
        raise RuntimeError("ModelManager is not initialized.")
    return manager


def _safe_publish_sse_log(message: str) -> None:
    try:
        publish_sse_log(message)
    except Exception as exc:
        print(f"[deployment] SSE publish failed: {exc}")


@router.post("/preload")
async def preload_model(payload: PreloadRequest, request: Request):
    try:
        _safe_publish_sse_log("[INFO] S3에서 모델 다운로드 시작")
        result = _get_model_manager(request).preload(payload.model_version)
        _safe_publish_sse_log("[INFO] 모델 로딩 완료")
        return {
            "status": result["status"],
            "model_version": result["model_version"],
        }
    except Exception as exc:
        _safe_publish_sse_log(f"[ERROR] {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/validate")
async def validate_model(request: Request):
    try:
        result = _get_model_manager(request).validate()
        _safe_publish_sse_log("[INFO] 검증 성공")
        return {
            "status": result["status"],
            "model_version": result["model_version"],
            "predictions": result["predictions"],
        }
    except Exception as exc:
        _safe_publish_sse_log(f"[ERROR] {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/switch")
async def switch_model(request: Request):
    try:
        manager = _get_model_manager(request)
        _safe_publish_sse_log("[INFO] active model 전환")
        result = manager.switch()
        classify_pipeline = getattr(request.app.state, "classify_pipeline", None)
        if classify_pipeline is not None:
            classify_pipeline["model"] = manager.current_bundle
        _safe_publish_sse_log("[INFO] 배포 완료")
        return {
            "status": result["status"],
            "model_version": result["model_version"],
        }
    except Exception as exc:
        _safe_publish_sse_log(f"[ERROR] {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
