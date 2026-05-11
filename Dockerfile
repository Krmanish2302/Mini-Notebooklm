# =============================================================================
#  Mini NotebookLM — Production Dockerfile  (multi-stage, non-root)
#  Version: 2.0.0
# 
#  Stages:
#    builder  — installs all Python deps into /install prefix
#    runtime  — lean python:3.12-slim, copies only /install + app source
# 
#  Build:
#    docker build -t mini-notebooklm:latest .
#  Run:
#    docker compose up --build
# =============================================================================

# -- Stage 1: builder ---------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential gcc g++ cmake git \
        ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# -- Stage 2: runtime ---------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="kumar2302github"
LABEL description="Mini NotebookLM — local open-source RAG research assistant"
LABEL version="2.0.0"
LABEL org.opencontainers.image.source="https://github.com/kumar2302github/Mini_NotebooLM"

# Runtime system packages only
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser

WORKDIR /app

# Copy app source (respects .dockerignore)
COPY --chown=appuser:appuser . .

# Runtime data directories
RUN mkdir -p data/vector_store data/graph && \
    chown -R appuser:appuser data

USER appuser

EXPOSE 8000

# Health check — polls /api/health every 30 s, 60 s grace period on start
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

# Environment defaults (overridden via .env / docker-compose env_file)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO \
    FAISS_INDEX_PATH=/app/data/vector_store \
    SQLITE_DB_PATH=/app/data/sources.db \
    GRAPH_STORAGE_PATH=/app/data/graph

# Start FastAPI via uvicorn
CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
