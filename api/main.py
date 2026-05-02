import os
import sys
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI

# src/ 디렉토리를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model_manager import ModelManager
from api.routers import classify, deployment, summarize
from messaging.consumer_classify import ClassifyConsumerRunner
from messaging.structured_log import get_logger
from src.settings import validate_startup_settings

log = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = validate_startup_settings()
    resolved_llm = settings.resolve_llm_config()
    print(
        f"[Startup] configuration validated: env={settings.APP_ENV}, "
        f"llm_provider={resolved_llm.provider}, model_source={settings.MODEL_SOURCE}"
    )

    print("[Startup] model manager loading...")
    model_manager = ModelManager()
    initial = model_manager.load_initial_model()
    runtime = initial.get("runtime") or {}
    log.info(
        "model_manager_ready",
        model_source=runtime.get("model_source"),
        active_model_version=runtime.get("active_model_version"),
        metadata_model_version=runtime.get("metadata_model_version"),
        loaded_sbert_path=runtime.get("loaded_sbert_path"),
        loaded_domain_model_path=runtime.get("loaded_domain_model_path"),
        loaded_intent_model_path=runtime.get("loaded_intent_model_path"),
    )
    app.state.model_manager = model_manager
    print("[Startup] model manager ready")

    print("[Startup] classify consumer starting...")
    consumer_runner = ClassifyConsumerRunner(app.state.model_manager)
    consumer_runner.start()
    app.state.classify_consumer_runner = consumer_runner
    await asyncio.sleep(0)
    print("[Startup] classify consumer started")

    yield

    print("[Shutdown] classify consumer stopping...")
    consumer_runner.stop()
    print("[Shutdown] classify consumer stopped")
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
        {
            "name": "Deployment",
            "description": "무중단 모델 배포를 위한 preload, validate, switch API입니다.",
        },
    ],
)

app.include_router(classify.router, tags=["Classification"])
app.include_router(summarize.router, tags=["Summarization"])
app.include_router(deployment.router, tags=["Deployment"])


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
