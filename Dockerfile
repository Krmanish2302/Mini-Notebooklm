# ─────────────────────────────────────────────────────────────────────────────
#  Mini NotebookLM — Production Dockerfile
#  Multi-stage build: builder installs deps, runtime is lean.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build-time system deps
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        git \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies into a prefix
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="kumar2302github" \
      description="Mini NotebookLM — local RAG research assistant" \
      version="1.0.0"

# Runtime system packages
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy application source
COPY --chown=appuser:appuser . .

# Create runtime data directories and set permissions
RUN mkdir -p data/vector_store data/graph && \
    chown -R appuser:appuser data

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Health check — polls /api/health every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

# Default environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO

# Start the FastAPI server
CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
