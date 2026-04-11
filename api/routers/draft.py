# ============================================================
# /draft 엔드포인트 — 비즈니스 로직은 draft_service 에 위임
# 내부 실험 / fallback 용 deprecated candidate 경로.
# 외부 계약은 유지하지만 classify 코어 경로는 아님.
# ============================================================

from fastapi import APIRouter, HTTPException, Request

from api.schemas import DraftRequest, DraftResponse
from api.services.draft_service import run_draft

router = APIRouter()


@router.post(
    "/draft",
    response_model=DraftResponse,
    summary="Internal experimental draft fallback",
    description="내부/실험/fallback 용 경로입니다. 외부 동작은 유지하지만 공식 코어 경로가 아니며 deprecated candidate 입니다.",
    deprecated=True,
)
async def draft_email(payload: DraftRequest, request: Request):
    try:
        return run_draft(payload, request.app.state.draft_pipeline)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
