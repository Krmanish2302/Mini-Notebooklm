# ── Stage 1: builder ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed for faiss-cpu, PyMuPDF, tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /app/data/uploads /app/data/chroma && chown -R appuser:appuser /app

COPY --chown=appuser:appuser . .

USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000 8501

# Default: FastAPI; override CMD in compose for Streamlit
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]