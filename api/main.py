# ============================================================
# FastAPI 앱 진입점
# 서버 시작 시 classify 코어 파이프라인을 한 번만 로드
# /draft 는 유지하되 deprecated 내부 경로로 취급한다.
# ============================================================

import sys
import os

# src/ 디렉토리를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI
from contextlib import asynccontextmanager

from inference import load_classify_pipeline, load_draft_pipeline, predict_email
from api.routers import classify, summarize, draft


# ── 서버 시작/종료 시 실행 ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시: classify 코어 파이프라인 로드 (한 번만 로드, 메모리 유지)
    print("[Startup] classify 파이프라인 로딩 중...")
    classify_model = load_classify_pipeline()
    classify_pipeline = {
        "model"  : classify_model,
        "predict": predict_email,   # 함수도 같이 저장
    }
    app.state.classify_pipeline = classify_pipeline
    print("[Startup] classify 파이프라인 로드 완료")

    # /draft 는 deprecated 내부 경로지만 서버 내에서 별도 상태를 유지한다.
    print("[Startup] draft 파이프라인 로딩 중...")
    draft_pipeline = {"model": load_draft_pipeline()}
    app.state.draft_pipeline = draft_pipeline
    print("[Startup] draft 파이프라인 로드 완료")

    yield  # 서버 실행 중

    # 서버 종료 시: 필요 시 정리 작업
    print("[Shutdown] 서버 종료")


# ── FastAPI 앱 생성 ──────────────────────────────────────────
app = FastAPI(
    title="Business Email AI Server",
    description="이메일 분류 중심 AI API. classify 가 공식 코어 경로이며, /draft 는 내부 실험용 fallback 경로로 유지됩니다.",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Classification",
            "description": "AI 서버의 공식 코어 기능입니다. 분류, 요약, 임베딩 생성을 담당합니다.",
        },
        {
            "name": "Summarization",
            "description": "보조 요약 기능입니다.",
        },
        {
            "name": "Draft",
            "description": "내부/실험/fallback 용 경로입니다. deprecated candidate 이며 향후 RAG 서버 이전 대상입니다.",
        },
    ],
)


# ── 라우터 등록 ──────────────────────────────────────────────
app.include_router(classify.router,  tags=["Classification"])
app.include_router(summarize.router, tags=["Summarization"])
app.include_router(draft.router,     tags=["Draft"])


# ── 헬스체크 ────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ── 로컬 실행 ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
