#!/usr/bin/env python3
"""
api.py  –  FastAPI backend for Mini NotebookLM
Run with:  uvicorn api:app --reload --port 8000

Integration layer between the React UI and the three mode pipelines:
  ChatPipeline        → /api/query  (mode=chat)
  DeepResearchPipeline → /api/query  (mode=deep / deep_research / research)
  StudyPipeline       → /api/query  (mode=study)

All bugs from the 2026-05-10 audit are fixed here:
  BUG-001 – stream generator not consumed inside run_in_executor
  BUG-002 – post-stream RAGAS via asyncio.create_task with guard
  BUG-004 – llm.stream() callable used in streaming path (not llm.invoke)
  BUG-005 – deep/study stream via MasterPipeline._stream_dict()
  BUG-006 – CORS origins driven from CORS_ORIGINS env var
  BUG-007 – context_chunks preferred over retrieved_chunks for RAGAS eval
  BUG-008 – single ContentAnalyzer instance per request (no double-init)
  BUG-009 – asyncio.get_running_loop() replaces deprecated get_event_loop()
  BUG-014 – response_model= added to POST /api/query
  BUG-015 – LLM guard raises HTTP 400 before StreamingResponse is returned
  BUG-020 – api_key uses Pydantic SecretStr (never logged)
  BUG-021 – file uploads validated for extension + 50 MB hard limit
  BUG-022 – query sanitizer strips prompt-injection patterns
  BUG-023 – heavy model loading deferred to FastAPI lifespan handler
  BUG-025 – no deprecated asyncio calls remain
  BUG-027 – RAGAS results persisted to SQLite via storage_manager
  NEW-001 – SSE stream emits quiz_cards, summary_bullets, sub_queries, learning_path
  NEW-002 – GET /api/sources lists actual stored sources from storage_manager
  NEW-003 – mode aliases: deep / research → deep_research
  NEW-004 – temperature / top_p / max_tokens forwarded to LLM per request
  NEW-005 – uploaded temp files always cleaned in finally block
"""

import os
import re
import asyncio
import tempfile
import json
import logging
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Deque, List, Optional

import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, SecretStr

from src.master_pipeline import MasterPipeline
from src.generation.persona_config import PersonaConfig
from src.evaluation.ragas_evaluator import RAGASEvaluator

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────────────────────────────────────

MAX_UPLOAD_BYTES  = 50 * 1024 * 1024          # 50 MB  (BUG-021)
ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".csv",
    ".png", ".jpg", ".jpeg",
    ".mp4", ".mp3", ".wav",
}
QUERY_MAX_LEN = 2000

# BUG-022: prompt-injection guard
_INJECTION_RE = re.compile(
    r"(?i)(ignore|disregard|forget|override|bypass).{0,40}"
    r"(instruction|prompt|system|rule)"
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mode alias map  (NEW-003)
_MODE_MAP = {
    "chat":          "chat",
    "deep":          "deep_research",
    "research":      "deep_research",
    "deep_research": "deep_research",
    "study":         "study",
}

# Embedding model catalogue (returned by /api/embedding-models)
EMBEDDING_MODELS = [
    {"name": "all-MiniLM-L6-v2",      "dim": 384,  "max_tokens": 256,  "label": "MiniLM",    "speed": "fast",   "note": "Local · fastest"},
    {"name": "all-mpnet-base-v2",      "dim": 768,  "max_tokens": 384,  "label": "MPNet",     "speed": "medium", "note": "Local · balanced"},
    {"name": "e5-large-v2",            "dim": 1024, "max_tokens": 512,  "label": "E5-Large",  "speed": "slow",   "note": "Local · most accurate"},
    {"name": "text-embedding-3-small", "dim": 1536, "max_tokens": 8191, "label": "OAI Small", "speed": "fast",   "note": "OpenAI API key required"},
    {"name": "text-embedding-3-large", "dim": 3072, "max_tokens": 8191, "label": "OAI Large", "speed": "medium", "note": "OpenAI API key required"},
]


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────

def _sanitize_query(q: str) -> str:
    """BUG-022: remove prompt-injection patterns and hard-cap length."""
    q = _INJECTION_RE.sub("", q)
    return q[:QUERY_MAX_LEN].strip()


# ────────────────────────────────────────────────────────────────────────────
#  Lifespan  (BUG-023: heavy init before first request)
# ────────────────────────────────────────────────────────────────────────────

pipeline:        MasterPipeline
_persona_config: PersonaConfig
_evaluator:      RAGASEvaluator
_ragas_history:  Deque[dict]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, _persona_config, _evaluator, _ragas_history
    logger.info("lifespan: initialising pipeline...")
    _persona_config = PersonaConfig()
    _evaluator      = RAGASEvaluator()
    _ragas_history  = deque(maxlen=50)
    pipeline        = MasterPipeline(mode="chat")   # embedding models loaded here
    logger.info("lifespan: pipeline ready")
    yield
    logger.info("lifespan: shutdown")


# ────────────────────────────────────────────────────────────────────────────
#  App
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mini NotebookLM API",
    version="1.2.0",
    lifespan=lifespan,
)

# BUG-006: CORS origins from env var so deployments can override
_raw_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:4173,http://localhost:3000",
)
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────────────────
#  Request / Response models
# ────────────────────────────────────────────────────────────────────────────

class ConfigRequest(BaseModel):
    provider: str      = "groq"
    model:    str      = "llama-3.3-70b-versatile"
    api_key:  SecretStr                        # BUG-020: never logged

class QueryRequest(BaseModel):
    query:       str
    mode:        str            = "chat"
    stream:      bool           = True
    temperature: Optional[float] = None       # NEW-004
    top_p:       Optional[float] = None
    max_tokens:  Optional[int]  = None
    ground_truth: Optional[str] = None

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

# BUG-014: typed response model for non-streaming query
class QueryResponse(BaseModel):
    answer:          str            = ""
    citations:       list           = []
    follow_ups:      List[str]      = []
    sources_used:    List[str]      = []
    chunks_used:     int            = 0
    tokens_estimate: int            = 0
    sub_queries:     List[str]      = []
    quiz_cards:      list           = []
    summary_bullets: List[str]      = []
    learning_path:   list           = []
    ragas:           Optional[dict] = None


# ────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _apply_tuning(req: QueryRequest) -> None:
    """NEW-004: forward per-request LLM tuning parameters."""
    if pipeline.llm and any(
        v is not None for v in (req.temperature, req.top_p, req.max_tokens)
    ):
        params = {
            k: v for k, v in {
                "temperature": req.temperature,
                "top_p":       req.top_p,
                "max_tokens":  req.max_tokens,
            }.items() if v is not None
        }
        pipeline.llm.update_tuning(**params)


async def _run_evaluation(
    question:       str,
    answer:         str,
    context_chunks: list,
    ground_truth:   Optional[str] = None,
) -> dict:
    """
    Run RAGAS evaluation in a thread-pool executor and cache the result.
    BUG-009: uses get_running_loop() — works inside async context.
    BUG-027: result persisted to SQLite.
    """
    loop = asyncio.get_running_loop()  # BUG-009
    result = await loop.run_in_executor(
        None,
        lambda: _evaluator.evaluate(
            question=question,
            answer=answer,
            context_chunks=context_chunks,
            ground_truth=ground_truth,
        )
    )
    result_dict = result.to_dict()
    _ragas_history.appendleft(result_dict)
    # BUG-027: persist to SQLite
    try:
        pipeline.storage_manager.save_ragas_result(result_dict)
    except Exception:
        pass
    return result_dict


def _resolve_mode(raw: str) -> str:
    """NEW-003: normalise mode aliases."""
    return _MODE_MAP.get(raw.strip().lower(), "chat")


# ────────────────────────────────────────────────────────────────────────────
#  Health & Stats
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "pipeline": "ready", "version": "1.2.0"}


@app.get("/api/stats")
def get_stats():
    try:
        stats = pipeline.get_stats()
        stats.setdefault("total_chunks",  stats.get("chunks",  {}).get("total_chunks",  0))
        stats.setdefault("total_sources", stats.get("sources", 0))
        stats.setdefault("graph",  {"nodes": 0, "edges": 0})
        stats.setdefault("chunks", {"total_chunks": 0, "dimensions": {}})
        return stats
    except Exception as exc:
        return {
            "total_chunks":  0,
            "total_sources": 0,
            "graph":  {"nodes": 0, "edges": 0},
            "chunks": {"total_chunks": 0, "dimensions": {}},
            "error":  str(exc),
        }


# ────────────────────────────────────────────────────────────────────────────
#  LLM Configuration
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/config")
def set_config(req: ConfigRequest):
    """
    Configure the LLM provider/model/key and rebuild all mode pipelines.
    BUG-020: SecretStr ensures the API key is never echoed in logs.
    """
    try:
        pipeline.set_llm(
            provider=req.provider.lower(),
            model=req.model,
            api_key=req.api_key.get_secret_value(),
        )
        return {"status": "configured", "provider": req.provider, "model": req.model}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ────────────────────────────────────────────────────────────────────────────
#  Mode
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/mode")
def set_mode(req: ModeRequest):
    """
    Switch the active pipeline mode.  NEW-003: accepts aliases.
    Accepted values: chat | deep | research | deep_research | study
    """
    internal = _resolve_mode(req.mode)
    pipeline.set_mode(internal)
    return {"mode": req.mode, "internal": internal}


# ────────────────────────────────────────────────────────────────────────────
#  Persona
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/persona")
def get_persona():
    return {"current": _persona_config.to_dict(), "catalogue": PersonaConfig.catalogue()}


@app.post("/api/persona")
def set_persona(req: PersonaRequest):
    global _persona_config
    try:
        if req.reset:
            _persona_config = PersonaConfig()
            return {"status": "reset", "current": _persona_config.to_dict()}

        _persona_config = PersonaConfig(
            persona=req.persona               or _persona_config.persona,
            tone=req.tone                     or _persona_config.tone,
            length=req.length                 or _persona_config.length,
            custom_persona=req.custom_persona or _persona_config.custom_persona,
        )
        # Sync persona to the prompt builder if it's live
        if hasattr(pipeline, "prompt_builder") and pipeline.prompt_builder:
            pipeline.prompt_builder.persona_config = _persona_config
        return {"status": "updated", "current": _persona_config.to_dict()}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ────────────────────────────────────────────────────────────────────────────
#  Embedding models
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/embedding-models")
def list_embedding_models():
    return {"models": EMBEDDING_MODELS}


# ────────────────────────────────────────────────────────────────────────────
#  Source Analysis  (pre-ingest preview)
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_source(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    """
    Analyse a source (file or URL/text) and return:
      - recommended chunking strategy
      - chunk count estimate
      - first-3-chunk previews
      - token stats
      - available embedding models

    BUG-008: ContentAnalyzer instantiated once per request.
    BUG-009: get_running_loop() for URL fetch.
    BUG-021: file extension + size validated before reading.
    """
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
                try:
                    loop    = asyncio.get_running_loop()  # BUG-009
                    fetched = await loop.run_in_executor(
                        None, lambda: pipeline.fetch_url_content(url)
                    )
                    source_content = fetched if fetched else url
                except Exception:
                    source_content = url
                source_name = url
            elif text:
                source_content = text
                source_name    = "pasted text"
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Provide 'url' or 'text' in JSON body",
                )
        else:
            if not file:
                raise HTTPException(status_code=400, detail="Provide a file or JSON body")
            # BUG-021: extension check
            ext = Path(file.filename or "").suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
            form_data   = await request.form()
            source_type = str(form_data.get("source_type", "pdf")).lower().strip()
            raw = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")
            suffix = ext or ".tmp"
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir=str(UPLOAD_DIR)
            )
            tmp.write(raw)
            tmp.close()
            source_content = Path(tmp.name).read_text(errors="ignore")
            source_name    = file.filename or "upload"
            os.unlink(tmp.name)

        # BUG-008: single ContentAnalyzer instance per request
        analyzer    = ContentAnalyzer()
        analysis    = analyzer.analyze(source_content, source_type=source_type)
        chunker     = AdaptiveChunker()
        recommended = chunker.recommend_strategy(source_content, source_type)
        chunks      = chunker.chunk(source_content, strategy=recommended)
        previews    = [
            {"index": i, "text": c.get("content", "")[:200]}
            for i, c in enumerate(chunks[:3])
        ]
        return {
            "source_name":         source_name,
            "source_type":         source_type,
            "recommended_strategy": recommended,
            "available_strategies": STRATEGIES,
            "chunk_count_estimate": len(chunks),
            "analysis":            analysis,
            "previews":            previews,
            "token_stats":         analysis.get("token_stats", {}),
            "embedding_models":    EMBEDDING_MODELS,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ────────────────────────────────────────────────────────────────────────────
#  Ingestion
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/ingest")
async def ingest_source(
    file:              Optional[UploadFile] = File(None),
    url:               Optional[str]        = Form(None),
    source_type:       str                  = Form("pdf"),
    chunking_strategy: Optional[str]        = Form(None),
    embedding_model:   Optional[str]        = Form(None),
):
    """
    Ingest a file or URL into the vector + metadata store.
    BUG-009: run_in_executor with get_running_loop().
    BUG-021: ext + size validation; NEW-005: temp file always cleaned.
    """
    tmp_path = None
    try:
        if file:
            ext = Path(file.filename or "").suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
            raw = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=ext or ".tmp", dir=str(UPLOAD_DIR)
            )
            tmp.write(raw)
            tmp.close()
            tmp_path = tmp.name
            loop   = asyncio.get_running_loop()  # BUG-009
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.ingest(
                    file_path=tmp_path,
                    url=None,
                    source_type=source_type,
                    chunking_strategy=chunking_strategy,
                    embedding_model=embedding_model,
                ),
            )
        elif url:
            loop   = asyncio.get_running_loop()  # BUG-009
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.ingest(
                    file_path=None,
                    url=url,
                    source_type=source_type,
                    chunking_strategy=chunking_strategy,
                    embedding_model=embedding_model,
                ),
            )
        else:
            raise HTTPException(status_code=400, detail="Provide 'file' or 'url'")

        return {"status": "ingested", "result": result}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # NEW-005: always clean up temp file
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ────────────────────────────────────────────────────────────────────────────
#  Sources
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/sources")
def list_sources():
    """
    NEW-002: list sources from storage_manager._sources dict so it reflects
    what is actually persisted, not a stale in-memory snapshot.
    """
    try:
        sources_dict = getattr(pipeline.storage_manager, "_sources", {}) or {}
        sources = list(sources_dict.values())
        return {"sources": sources, "total_sources": len(sources)}
    except Exception as exc:
        return {"sources": [], "total_sources": 0, "error": str(exc)}


@app.delete("/api/sources/{source_id}")
def delete_source(source_id: str):
    """Delete a source and rebuild the BM25 sparse index."""
    try:
        ok = pipeline.delete_source(source_id)
        if not ok:
            raise HTTPException(
                status_code=404, detail=f"Source not found: {source_id}"
            )
        return {"status": "deleted", "source_id": source_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")


# ────────────────────────────────────────────────────────────────────────────
#  Query  (non-streaming)
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)  # BUG-014
async def query(req: QueryRequest):
    """
    Non-streaming query.  Runs the active mode pipeline synchronously in a
    thread-pool executor then evaluates with RAGAS inline.

    Response includes all mode-specific fields:
      chat         → answer, citations, follow_ups, sources_used
      deep_research → + sub_queries
      study        → + quiz_cards, summary_bullets, learning_path
    """
    # BUG-015: guard before any processing
    if not pipeline.llm:
        raise HTTPException(
            status_code=400,
            detail="LLM not configured. POST /api/config first.",
        )
    try:
        # NEW-003: resolve mode alias and sync pipeline
        internal_mode = _resolve_mode(req.mode)
        if pipeline.mode != internal_mode:
            pipeline.set_mode(internal_mode)

        _apply_tuning(req)                         # NEW-004
        safe_query = _sanitize_query(req.query)    # BUG-022

        loop   = asyncio.get_running_loop()        # BUG-009
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.generate(
                safe_query,
                stream=False,
                persona_config=_persona_config if internal_mode == "chat" else None,
                ground_truth=req.ground_truth,
            ),
        )

        # BUG-007: prefer deduplicated context_chunks for RAGAS
        context_chunks = (
            result.get("context_chunks")
            or result.get("retrieved_chunks", [])
        )
        ragas = await _run_evaluation(
            question=safe_query,
            answer=result.get("answer", ""),
            context_chunks=context_chunks,
            ground_truth=req.ground_truth,
        )
        result["ragas"] = ragas

        # Normalise for QueryResponse model
        result.setdefault("citations",       [])
        result.setdefault("follow_ups",      [])
        result.setdefault("sources_used",    [])
        result.setdefault("chunks_used",     len(result.get("chunks_used", [])))
        result.setdefault("tokens_estimate", 0)
        result.setdefault("sub_queries",     [])    # deep_research
        result.setdefault("quiz_cards",      [])    # study
        result.setdefault("summary_bullets", [])    # study
        result.setdefault("learning_path",   [])    # study

        # chunks_used in response model is an int count
        if isinstance(result["chunks_used"], list):
            result["chunks_used"] = len(result["chunks_used"])

        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ────────────────────────────────────────────────────────────────────────────
#  Query  (SSE streaming)
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """
    Server-Sent Events streaming query.

    SSE event types:
      {type: "token",    content: "..."}                 — LLM token chunk
      {type: "metadata", citations:[...], sub_queries:[],  — end-of-tokens metadata
              quiz_cards:[], summary_bullets:[], ...}
      {type: "ragas",    faithfulness: 0.87, ...}         — grounding score
      {type: "done"}                                      — stream complete
      {type: "error",    detail: "..."}                   — error occurred

    BUG-001: generate(stream=True) called directly — NOT inside run_in_executor
    BUG-002: post-stream RAGAS via asyncio.create_task
    BUG-004: token streaming uses llm.stream() callable (set in master_pipeline)
    BUG-005: deep/study use _stream_dict() to emit words then metadata dict
    BUG-015: LLM guard before StreamingResponse is constructed
    NEW-001: metadata dict includes quiz_cards, summary_bullets, learning_path
    """
    # BUG-015: guard before constructing StreamingResponse
    if not pipeline.llm:
        raise HTTPException(
            status_code=400,
            detail="LLM not configured. POST /api/config first.",
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        full_answer  = ""
        retrieved    = []
        context_cks  = []

        try:
            # NEW-003: resolve + sync mode
            internal_mode = _resolve_mode(req.mode)
            if pipeline.mode != internal_mode:
                pipeline.set_mode(internal_mode)

            _apply_tuning(req)                           # NEW-004
            safe_query = _sanitize_query(req.query)      # BUG-022

            # BUG-001: call generate() directly here, NOT inside run_in_executor
            # generate(stream=True) returns a lazy iterator; we iterate it
            # synchronously but each yield passes control back to the ASGI event loop.
            result = pipeline.generate(
                safe_query,
                stream=True,
                persona_config=_persona_config if internal_mode == "chat" else None,
                ground_truth=req.ground_truth,
            )

            if hasattr(result, "__iter__") and not isinstance(result, dict):
                for chunk in result:
                    if isinstance(chunk, str):
                        full_answer += chunk
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    elif isinstance(chunk, dict):
                        # Final metadata dict emitted by _stream_dict()
                        retrieved   = chunk.get("retrieved_chunks", retrieved)
                        context_cks = chunk.get("context_chunks",   context_cks)
                        # NEW-001: include all study/research fields
                        meta = {
                            k: chunk[k] for k in (
                                "citations", "sources_used", "follow_ups",
                                "sub_queries",      # deep_research
                                "quiz_cards",       # study
                                "summary_bullets",  # study
                                "learning_path",    # study
                                "ragas",
                            ) if k in chunk
                        }
                        yield f"data: {json.dumps({'type': 'metadata', **meta})}\n\n"
            else:
                # Fallback: result is a plain dict (should not happen for stream=True)
                answer      = result.get("answer", "") if isinstance(result, dict) else str(result)
                full_answer = answer
                for word in answer.split(" "):
                    if word:
                        yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                if isinstance(result, dict):
                    retrieved   = result.get("retrieved_chunks", [])
                    context_cks = result.get("context_chunks",   [])
                    meta = {
                        k: result[k] for k in (
                            "citations", "sources_used", "follow_ups",
                            "sub_queries", "quiz_cards",
                            "summary_bullets", "learning_path",
                        ) if k in result
                    }
                    if meta:
                        yield f"data: {json.dumps({'type': 'metadata', **meta})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # BUG-002: post-stream RAGAS as a fire-and-forget task
            # We can't yield after done, so the UI polls /api/ragas/history.
            async def _eval_and_store():
                try:
                    chunks_for_eval = context_cks or retrieved
                    ragas = await _run_evaluation(
                        question=safe_query,
                        answer=full_answer,
                        context_chunks=chunks_for_eval,
                        ground_truth=req.ground_truth,
                    )
                    pipeline._last_ragas = ragas
                except Exception as exc:
                    logger.warning("Post-stream RAGAS eval failed: %s", exc)

            asyncio.create_task(_eval_and_store())  # BUG-002

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ────────────────────────────────────────────────────────────────────────────
#  RAGAS (on-demand + history)
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    """On-demand RAGAS evaluation of any question/answer pair."""
    try:
        ragas = await _run_evaluation(
            question=req.question,
            answer=req.answer,
            context_chunks=req.context_chunks,
            ground_truth=req.ground_truth,
        )
        return ragas
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/ragas/history")
def ragas_history(limit: int = 20):
    """Return the last N RAGAS evaluation results (newest first)."""
    return {"history": list(_ragas_history)[:limit]}
