#!/usr/bin/env python3
"""
api.py  –  FastAPI backend for Mini NotebookLM
Run with:  uvicorn api:app --reload --port 8000
"""

import os
import uuid
import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Optional

import nest_asyncio
nest_asyncio.apply()  # Fix asyncio.run() inside existing event loop

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.master_pipeline import MasterPipeline

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Mini NotebookLM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── global pipeline singleton ─────────────────────────────────────────────────
pipeline: MasterPipeline = MasterPipeline(mode="chat")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── request models ────────────────────────────────────────────────────────────
class ConfigRequest(BaseModel):
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    api_key: str


class QueryRequest(BaseModel):
    query: str
    mode: str = "chat"
    stream: bool = True


class ModeRequest(BaseModel):
    mode: str


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "pipeline": "ready"}


@app.get("/api/stats")
def get_stats():
    """Return pipeline stats for the right panel."""
    try:
        stats = pipeline.get_stats()
        stats.setdefault("total_chunks", stats.get("chunks", {}).get("total_chunks", 0))
        stats.setdefault("total_sources", stats.get("sources", 0))
        stats.setdefault("graph", {"nodes": 0, "edges": 0})
        stats.setdefault("chunks", {"total_chunks": 0, "dimensions": {}})
        return stats
    except Exception as e:
        return {
            "total_chunks": 0,
            "total_sources": 0,
            "graph": {"nodes": 0, "edges": 0},
            "chunks": {"total_chunks": 0, "dimensions": {}},
            "error": str(e),
        }


@app.post("/api/config")
def set_config(req: ConfigRequest):
    """Set LLM provider, model and API key."""
    try:
        pipeline.set_llm(
            provider=req.provider.lower(),
            model=req.model,
            api_key=req.api_key,
        )
        return {"status": "configured", "provider": req.provider, "model": req.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mode")
def set_mode(req: ModeRequest):
    """
    Switch pipeline mode WITHOUT resetting chat history.
    Directly sets pipeline.mode — does NOT call pipeline.set_mode() which
    would recreate the session UUID and wipe history.
    """
    mode_map = {"chat": "chat", "deep": "deep_research", "study": "study"}
    internal_mode = mode_map.get(req.mode, "chat")
    pipeline.mode = internal_mode
    return {"mode": req.mode, "internal": internal_mode}


@app.post("/api/analyze")
async def analyze_source(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    source_type: str = Form("pdf"),
):
    """
    Pre-ingest analysis: returns chunking recommendation + chunk previews.
    Feeds the EmbedFlow component in the sidebar.
    """
    try:
        from src.ingestion.preprocessing.content_analyzer import ContentAnalyzer
        from src.ingestion.chunking.adaptive_chunker import AdaptiveChunker

        analyzer = ContentAnalyzer()
        STRATEGIES = ["recursive", "paragraph", "page", "semantic", "hierarchical"]

        if file:
            suffix = Path(file.filename).suffix or ".tmp"
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir=str(UPLOAD_DIR)
            )
            tmp.write(await file.read())
            tmp.close()
            sour