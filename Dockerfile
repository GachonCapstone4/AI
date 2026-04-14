# syntax=docker/dockerfile:1
# 1단계: Builder Stage (의존성 설치 및 컴파일)
FROM --platform=linux/arm64 python:3.11-slim-bookworm AS builder

WORKDIR /app

# APT 캐시 마운트 사용하여 시스템 패키지 설치 속도 개선
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 가상환경 생성 (최종 스테이지로 복사하기 위함)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pip 캐시 마운트 사용하여 라이브러리 설치 속도 개선
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# ------------------------------------------------------------
# 2단계: Final Stage (실행 환경)
FROM --platform=linux/arm64 python:3.11-slim-bookworm AS runner

WORKDIR /app

# 1단계에서 설치된 가상환경(라이브러리 포함)만 복사
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 소스 코드 복사 (변경이 잦은 코드는 마지막에 배치하여 레이어 캐시 활용)
COPY api/ ./api/
COPY src/ ./src/
COPY messaging/ ./messaging/
COPY models/ ./models/

# 포트 노출
EXPOSE 8000

# 헬스 체크 (가상환경 내의 python 사용)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 앱 실행
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
