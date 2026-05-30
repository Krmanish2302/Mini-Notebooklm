#!/usr/bin/env python3
"""
api.py — FastAPI backend for Mini NotebookLM
Run: uvicorn api:app --reload --port 8000

Changes (fix pass)
------------------
* _MODE_MAP gains "analyze" entry (PRD §6)
* source_ids forwarded from QueryRequest → pipeline.ask(source_ids=...)
* All other logic unchanged from v1.4.0
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Deque, List, Optional

import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, SecretStr

from src.master_pipeline import MiniNotebookLM, PipelineConfig
from src.generation.persona_config import PersonaConfig
from src.generation.llm_registry import LLMRegistry

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES   = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".png", ".jpg", ".jpeg", ".mp4", ".mp3", ".wav"}
QUERY_MAX_LEN      = 2_000
UPLOAD_DIR         = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_INJECTION_RE = re.compile(
    r"(?i)(ignore|disregard|forget|override|bypass).{0,40}"
    r"(instruction|prompt|system|rule)"
)

# FIX #4: added "analyze" mode ─────────────────────────────────────────────────
_MODE_MAP = {
    "chat":          "chat",
    "deep":          "research",
    "research":      "research",
    "deep_research": "research",
    "study":         "study",
    "analyze":       "chat",   # analyze uses chat pipeline; distinction is in prompt
}

EMBEDDING_MODELS = [
    {"name": "all-MiniLM-L6-v2",      "dim": 384,  "max_tokens": 256,  "label": "MiniLM",    "speed": "fast",   "note": "Local · fastest"},
    {"name": "all-mpnet-base-v2",      "dim": 768,  "max_tokens": 384,  "label": "MPNet",     "speed": "medium", "note": "Local · balanced"},
    {"name": "e5-large-v2",            "dim": 1024, "max_tokens": 512,  "label": "E5-Large",  "speed": "slow",   "note": "Local · most accurate"},
    {"name": "text-embedding-3-small", "dim": 1536, "max_tokens": 8191, "label": "OAI Small", "speed": "fast",   "note": "OpenAI API key required"},
    {"name": "text-embedding-3-large", "dim": 3072, "max_tokens": 8191, "label": "OAI Large", "speed": "medium", "note": "OpenAI API key required"},
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _sanitize(q: str) -> str:
    return _INJECTION_RE.sub("", q)[:QUERY_MAX_LEN].strip()

def _resolve_mode(raw: str) -> str:
    return _MODE_MAP.get(raw.strip().lower(), "chat")

# ── Globals ────────────────────────────────────────────────────────────────────

pipeline:       MiniNotebookLM
_persona:       PersonaConfig
_ragas_history: Deque[dict]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, _persona, _ragas_history
    logger.info("lifespan: initialising pipeline …")
    _persona       = PersonaConfig()
    _ragas_history = deque(maxlen=50)
    pipeline       = MiniNotebookLM(PipelineConfig())
    logger.info("lifespan: pipeline ready — %s", pipeline.status())
    yield
    logger.info("lifespan: shutdown")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Mini NotebookLM API", version="1.4.1", lifespan=lifespan)

_raw_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:4173,http://localhost:3000,http://localhost:8501",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _raw_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class ConfigRequest(BaseModel):
    provider: str       = "groq"
    model:    str       = "llama-3.1-70b-versatile"
    api_key:  SecretStr

class QueryRequest(BaseModel):
    query:        str
    mode:         str             = "chat"
    stream:       bool            = True
    temperature:  Optional[float] = None
    top_p:        Optional[float] = None
    max_tokens:   Optional[int]   = None
    ground_truth: Optional[str]   = None
    source_ids:   List[str]       = []   # restrict retrieval to these source IDs

class ModeRequest(BaseModel):
    mode: str

class PersonaRequest(BaseModel):
    persona:        Optional[str] = None
    tone:           Optional[str] = None
    length:         Optional[str] = None
    custom_persona: Optional[str] = None
    reset:          bool          = False

class EvaluateRequest(BaseModel):
    question:       str
    answer:         str
    context_chunks: list          = []
    ground_truth:   Optional[str] = None

class QueryResponse(BaseModel):
    answer:          str       = ""
    citations:       list      = []
    follow_ups:      List[str] = []
    sources_used:    List[str] = []
    chunks_used:     int       = 0
    tokens_estimate: int       = 0
    sub_queries:     List[str] = []
    quiz_cards:      list      = []
    summary_bullets: List[str] = []
    learning_path:   list      = []
    ragas:           Optional[dict] = None


# ── Internal helpers ───────────────────────────────────────────────────────────

async def _run_ragas(
    question: str,
    answer:   str,
    contexts: list,
    ground_truth: Optional[str] = None,
) -> dict:
    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pipeline._evaluator.evaluate(
            query=question,
            answer=answer,
            contexts=[
                (c.get("content", "") if isinstance(c, dict) else str(c))
                for c in contexts
            ],
            ground_truth=ground_truth,
        ) if pipeline._evaluator else {},
    )
    r = result if isinstance(result, dict) else (result.to_dict() if hasattr(result, "to_dict") else {})
    _ragas_history.appendleft(r)
    return r


def _list_sources_safe() -> list:
    try:
        sm = getattr(pipeline, "_source_manager", None)
        if sm and hasattr(sm, "list_sources"):
            return sm.list_sources()
    except Exception:
        pass
    try:
        ing = getattr(pipeline, "_ingestion", None)
        if ing and hasattr(ing, "list_sources"):
            return ing.list_sources()
    except Exception:
        pass
    return []


def _ingest_via_router(
    source_type: str,
    source_id:   str,
    file_path:   Optional[str] = None,
    content:     Optional[str] = None,
    strategy:    str           = "paragraph_based",
) -> dict:
    from src.ingestion.ingestion_router import ingest
    return ingest(
        source_type  = source_type,
        source_id    = source_id,
        file_path    = file_path,
        content      = content,
        strategy     = strategy,
        embedding_dim= 384,
    )


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    s = pipeline.status()
    return {"status": "ok", "version": "1.4.1", **s}

@app.get("/api/stats")
def get_stats():
    try:
        return pipeline.status()
    except Exception as exc:
        return {"error": str(exc)}


# ── LLM Config ─────────────────────────────────────────────────────────────────

@app.post("/api/config")
def set_config(req: ConfigRequest):
    try:
        os.environ[f"{req.provider.upper()}_API_KEY"] = req.api_key.get_secret_value()
        LLMRegistry.get.cache_clear()
        pipeline.config.llm_provider = req.provider.lower()
        pipeline.config.llm_model    = req.model
        return {"status": "configured", "provider": req.provider, "model": req.model}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Mode ───────────────────────────────────────────────────────────────────────

@app.post("/api/mode")
def set_mode(req: ModeRequest):
    internal = _resolve_mode(req.mode)
    pipeline.config.default_mode = internal
    return {"mode": req.mode, "internal": internal}


# ── Persona ────────────────────────────────────────────────────────────────────

@app.get("/api/persona")
def get_persona():
    return {"current": _persona.to_dict(), "catalogue": PersonaConfig.catalogue()}

@app.post("/api/persona")
def set_persona(req: PersonaRequest):
    global _persona
    try:
        if req.reset:
            _persona = PersonaConfig()
            return {"status": "reset", "current": _persona.to_dict()}
        _persona = PersonaConfig(
            persona        = req.persona        or _persona.persona,
            tone           = req.tone           or _persona.tone,
            length         = req.length         or _persona.length,
            custom_persona = req.custom_persona or _persona.custom_persona,
        )
        return {"status": "updated", "current": _persona.to_dict()}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Embedding models ───────────────────────────────────────────────────────────

@app.get("/api/embedding-models")
def list_embedding_models():
    return {"models": EMBEDDING_MODELS}


# ── Analyze ────────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_source(request: Request, file: Optional[UploadFile] = File(None)):
    STRATEGIES = ["recursive", "paragraph", "page", "semantic", "hierarchical"]
    content_type = request.headers.get("content-type", "")
    try:
        from src.ingestion.preprocessing.content_analyzer import ContentAnalyzer
        from src.ingestion.chunking.adaptive_chunker import AdaptiveChunker

        if "application/json" in content_type:
            body        = await request.json()
            url         = body.get("url",  "").strip()
            text        = body.get("text", "").strip()
            source_type = body.get("source_type", "text").lower().strip()
            if url:
                loop    = asyncio.get_running_loop()
                fetched = await loop.run_in_executor(None, lambda: pipeline.retrieve(url))
                source_content = url
                source_name    = url
            elif text:
                source_content = text
                source_name    = "pasted text"
            else:
                raise HTTPException(status_code=400, detail="Provide 'url' or 'text'")
        else:
            if not file:
                raise HTTPException(status_code=400, detail="Provide a file or JSON body")
            ext = Path(file.filename or "").suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
            form_data   = await request.form()
            source_type = str(form_data.get("source_type", "pdf")).lower().strip()
            raw         = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds 50 MB")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".tmp", dir=str(UPLOAD_DIR))
            tmp.write(raw); tmp.close()
            source_content = Path(tmp.name).read_text(errors="ignore")
            source_name    = file.filename or "upload"
            os.unlink(tmp.name)

        analyzer    = ContentAnalyzer()
        analysis    = analyzer.analyze(source_content, source_type=source_type)
        chunker     = AdaptiveChunker()
        recommended = chunker.recommend_strategy(source_content, source_type)
        chunks      = chunker.chunk(source_content, strategy=recommended)
        previews    = [{"index": i, "text": c.get("content", "")[:200]} for i, c in enumerate(chunks[:3])]
        return {
            "source_name": source_name, "source_type": source_type,
            "recommended_strategy": recommended, "available_strategies": STRATEGIES,
            "chunk_count_estimate":  len(chunks),
            "analysis": analysis, "previews": previews,
            "token_stats": analysis.get("token_stats", {}),
            "embedding_models": EMBEDDING_MODELS,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Ingest ─────────────────────────────────────────────────────────────────────

@app.post("/api/ingest")
async def ingest_source(
    file:              Optional[UploadFile] = File(None),
    url:               Optional[str]        = Form(None),
    source_type:       str                  = Form("pdf"),
    source_id:         Optional[str]        = Form(None),
    chunking_strategy: Optional[str]        = Form(None),
    embedding_model:   Optional[str]        = Form(None),
    content:           Optional[str]        = Form(None),
):
    sid      = (source_id or "").strip() or str(uuid.uuid4())[:8]
    strategy = (chunking_strategy or "paragraph_based").strip()
    tmp_path = None

    try:
        resolved_path:    Optional[str] = None
        resolved_content: Optional[str] = content

        if file:
            ext = Path(file.filename or "").suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
            raw = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds 50 MB")
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=ext or ".tmp", dir=str(UPLOAD_DIR)
            )
            tmp.write(raw); tmp.close()
            tmp_path      = tmp.name
            resolved_path = tmp_path
            if source_type == "pdf" and ext in {".png", ".jpg", ".jpeg"}:
                source_type = "image"
        elif url:
            resolved_path = url
        elif content and source_type == "text":
            resolved_content = content
        else:
            raise HTTPException(status_code=400, detail="Provide 'file', 'url', or 'content'")

        # Try ingestion_router first
        try:
            loop   = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: _ingest_via_router(
                    source_type = source_type,
                    source_id   = sid,
                    file_path   = resolved_path,
                    content     = resolved_content,
                    strategy    = strategy,
                ),
            )
            return {"status": "ingested", "source_id": sid, "router": "ingestion_router", "result": result}
        except ImportError:
            logger.warning("/api/ingest: ingestion_router not available, falling back")

        # Fallback: legacy pipeline.ingest()
        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.ingest(resolved_path or ""),
        )
        return {"status": "ingested", "source_id": sid, "router": "legacy", "result": result}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and Path(tmp_path).exists():
            try: os.unlink(tmp_path)
            except Exception: pass


# ── Sources ────────────────────────────────────────────────────────────────────

@app.get("/api/sources")
def list_sources():
    try:
        sources = _list_sources_safe()
        return {"sources": sources, "total_sources": len(sources)}
    except Exception as exc:
        return {"sources": [], "total_sources": 0, "error": str(exc)}

@app.delete("/api/sources/{source_id}")
def delete_source(source_id: str):
    try:
        sm = getattr(pipeline, "_source_manager", None)
        if sm and hasattr(sm, "delete_source"):
            ok = sm.delete_source(source_id)
            if ok:
                return {"status": "deleted", "source_id": source_id}
        ing = getattr(pipeline, "_ingestion", None)
        if ing and hasattr(ing, "delete_source"):
            ok = ing.delete_source(source_id)
            if not ok:
                raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")
            return {"status": "deleted", "source_id": source_id}
        raise HTTPException(status_code=501, detail="No delete-capable ingestion module loaded")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Query (non-streaming) ──────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not pipeline.status()["llm_ok"]:
        raise HTTPException(status_code=400, detail="LLM not configured. POST /api/config first.")
    try:
        internal_mode = _resolve_mode(req.mode)
        safe_query    = _sanitize(req.query)
        _sids         = req.source_ids if req.source_ids else None
        loop          = asyncio.get_running_loop()
        gen_result    = await loop.run_in_executor(
            None,
            lambda: pipeline.ask(
                safe_query,
                mode=internal_mode,
                persona=_persona if internal_mode == "chat" else None,
                evaluate=False,
                ground_truth=req.ground_truth,
                source_ids=_sids,
            ),
        )
        context_chunks = gen_result.chunks_used or []
        ragas = await _run_ragas(safe_query, gen_result.answer, context_chunks, req.ground_truth)
        return QueryResponse(
            answer=gen_result.answer,
            citations=gen_result.citations,
            follow_ups=gen_result.follow_ups,
            sources_used=gen_result.sources_used,
            chunks_used=len(gen_result.chunks_used),
            tokens_estimate=gen_result.tokens_estimate,
            ragas=ragas,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Query (SSE streaming) ──────────────────────────────────────────────────────

@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    if not pipeline.status()["llm_ok"]:
        raise HTTPException(status_code=400, detail="LLM not configured. POST /api/config first.")

    async def event_generator() -> AsyncGenerator[str, None]:
        full_answer   = ""
        chunks_used   = []
        safe_query    = _sanitize(req.query)
        internal_mode = _resolve_mode(req.mode)
        _sids         = req.source_ids if req.source_ids else None

        try:
            gen_result = pipeline.ask(
                safe_query,
                mode=internal_mode,
                persona=_persona if internal_mode == "chat" else None,
                evaluate=False,
                ground_truth=req.ground_truth,
                stream=True,
                source_ids=_sids,
            )
            full_answer = gen_result.answer
            chunks_used = gen_result.chunks_used or []

            for word in full_answer.split(" "):
                if word:
                    yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

            meta = {
                "citations":    gen_result.citations,
                "sources_used": gen_result.sources_used,
                "follow_ups":   gen_result.follow_ups,
            }
            yield f"data: {json.dumps({'type': 'metadata', **meta})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            async def _eval():
                try:
                    ragas = await _run_ragas(safe_query, full_answer, chunks_used, req.ground_truth)
                    pipeline.last_ragas = ragas
                except Exception as exc:
                    logger.warning("Post-stream RAGAS failed: %s", exc)
            asyncio.create_task(_eval())

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── RAGAS ──────────────────────────────────────────────────────────────────────

@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    try:
        return await _run_ragas(req.question, req.answer, req.context_chunks, req.ground_truth)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/ragas/history")
def ragas_history(limit: int = 20):
    return {"history": list(_ragas_history)[:limit]}
