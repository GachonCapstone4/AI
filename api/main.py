import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

# src/ 디렉토리를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from inference import load_classify_pipeline, predict_email
from api.routers import classify, summarize
from src.settings import validate_startup_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = validate_startup_settings()
    resolved_llm = settings.resolve_llm_config()
    print(
        f"[Startup] configuration validated: env={settings.APP_ENV}, "
        f"llm_provider={resolved_llm.provider}, model_source={settings.MODEL_SOURCE}"
    )

    print("[Startup] classify pipeline loading...")
    classify_model = load_classify_pipeline()
    app.state.classify_pipeline = {
        "model": classify_model,
        "predict": predict_email,
    }
    print("[Startup] classify pipeline ready")

    yield

    print("[Shutdown] server stopped")


app = FastAPI(
    title="Business Email AI Server",
    description="이메일 분류와 요약을 제공하는 AI API 서버입니다.",
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
    ],
)

app.include_router(classify.router, tags=["Classification"])
app.include_router(summarize.router, tags=["Summarization"])


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    settings = validate_startup_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.APP_ENV in {"local", "dev"},
    )
