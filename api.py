#!/usr/bin/env python3
"""
api.py  –  FastAPI backend for Mini NotebookLM React UI
Run with:  uvicorn api:app --reload --port 8000

Bug fixes applied (2026-05-10 audit):
  BUG-001: streaming generator no longer consumed inside run_in_executor
  BUG-002: post-stream RAGAS wrapped in asyncio.create_task with guard
  BUG-006: CORS origins driven from CORS_ORIGINS env var
  BUG-007: context_chunks key correct for RAGAS eval
  BUG-008: duplicate ContentAnalyzer instantiation removed
  BUG-009: get_event_loop() -> get_running_loop()
  BUG-014: response_model= added to /api/query
  BUG-015: LLM guard raises HTTP 400 before StreamingResponse
  BUG-020: api_key uses Pydantic SecretStr
  BUG-021: file upload validates extension + 50 MB size limit
  BUG-022: query sanitizer strips prompt injection patterns
  BUG-025: asyncio deprecation resolved
  BUG-027: RAGAS history persisted to SQLite
"""

import os
import re
import asyncio
import tempfile
import json
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

# ── constants ───────────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB  (BUG-021)
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".png", ".jpg", ".jpeg", ".mp4", ".mp3", ".wav"}
QUERY_MAX_LEN = 2000
_INJECTION_RE = re.compile(
    r"(?i)(ignore|disregard|forget|override|bypass).{0,40}(instruction|prompt|system|rule)"
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODELS = [
    {"name": "all-MiniLM-L6-v2",      "dim": 384,  "max_tokens": 256,  "label": "MiniLM",    "speed": "fast",   "note": "Local · fastest"},
    {"name": "all-mpnet-base-v2",      "dim": 768,  "max_tokens": 384,  "label": "MPNet",     "speed": "medium", "note": "Local · balanced"},
    {"name": "e5-large-v2",            "dim": 1024, "max_tokens": 512,  "label": "E5-Large",  "speed": "slow",   "note": "Local · most accurate"},
    {"name": "text-embedding-3-small", "dim": 1536, "max_tokens": 8191, "label": "OAI Small", "speed": "fast",   "note": "OpenAI API key required"},
    {"name": "text-embedding-3-large", "dim": 3072, "max_tokens": 8191, "label": "OAI Large", "speed": "medium", "note": "OpenAI API key required"},
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _sanitize_query(q: str) -> str:
    """BUG-022: strip prompt-injection patterns and hard-cap length."""
    q = _INJECTION_RE.sub("", q)
    return q[:QUERY_MAX_LEN].strip()


# ── lifespan (BUG-023) ───────────────────────────────────────────────────────
# Model loading deferred to lifespan so workers are ready before first request.

pipeline: MasterPipeline
_persona_config: PersonaConfig
_evaluator: RAGASEvaluator
_ragas_history: Deque[dict]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, _persona_config, _evaluator, _ragas_history
    _persona_config = PersonaConfig()
    _evaluator      = RAGASEvaluator()
    _ragas_history  = deque(maxlen=50)
    pipeline        = MasterPipeline(mode="chat")   # embedding model loaded here
    yield
    # shutdown cleanup (if needed)


# ── app ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Mini NotebookLM API", version="1.1.0", lifespan=lifespan)

# BUG-006: drive CORS origins from env var
_raw_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:4173,http://localhost:3000"
)
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ─────────────────────────────────────────────────

class ConfigRequest(BaseModel):
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    api_key: SecretStr                    # BUG-020: SecretStr prevents logging

class QueryRequest(BaseModel):
    query: str
    mode: str = "chat"
    stream: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

class ModeRequest(BaseModel):
    mode: str

class PersonaRequest(BaseModel):
    persona:        Optional[str] = None
    tone:           Optional[str] = None
    length:         Optional[str] = None
    custom_persona: Optional[str] = None
    reset:          bool = False

class EvaluateRequest(BaseModel):
    question:       str
    answer:         str
    context_chunks: list = []
    ground_truth:   Optional[str] = None

# BUG-014: response model for /api/query
class QueryResponse(BaseModel):
    answer:         str = ""
    citations:      List[str] = []
    follow_ups:     List[str] = []
    sources_used:   List[str] = []
    chunks_used:    int = 0
    tokens_estimate: int = 0
    sub_queries:    List[str] = []
    quiz_cards:     list = []
    summary_bullets: List[str] = []
    ragas:          Optional[dict] = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _apply_tuning(req: QueryRequest) -> None:
    if pipeline.llm and any(
        v is not None for v in (req.temperature, req.top_p, req.max_tokens)
    ):
        pipeline.llm.update_tuning(
            **{k: v for k, v in {
                "temperature": req.temperature,
                "top_p":       req.top_p,
                "max_tokens":  req.max_tokens,
            }.items() if v is not None}
        )


async def _run_evaluation(
    question: str,
    answer: str,
    context_chunks: list,
    ground_truth: Optional[str] = None,
) -> dict:
    """Run RAGAS eval in thread pool; cache result in history."""
    loop = asyncio.get_running_loop()   # BUG-009: get_running_loop()
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
    # BUG-027: also persist to SQLite
    try:
        pipeline.storage_manager.save_ragas_result(result_dict)
    except Exception:
        pass
    return result_dict


# ── existing routes ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "pipeline": "ready"}


@app.get("/api/stats")
def get_stats():
    try:
        stats = pipeline.get_stats()
        stats.setdefault("total_chunks", stats.get("chunks", {}).get("total_chunks", 0))
        stats.setdefault("total_sources", stats.get("sources", 0))
        stats.setdefault("graph", {"nodes": 0, "edges": 0})
        stats.setdefault("chunks", {"total_chunks": 0, "dimensions": {}})
        return stats
    except Exception as e:
        return {"total_chunks": 0, "total_sources": 0,
                "graph": {"nodes": 0, "edges": 0},
                "chunks": {"total_chunks": 0, "dimensions": {}}, "error": str(e)}


@app.post("/api/config")
def set_config(req: ConfigRequest):
    try:
        # BUG-020: extract secret value safely
        pipeline.set_llm(
            provider=req.provider.lower(),
            model=req.model,
            api_key=req.api_key.get_secret_value(),
        )
        return {"status": "configured", "provider": req.provider, "model": req.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mode")
def set_mode(req: ModeRequest):
    mode_map = {"chat": "chat", "deep": "deep_research", "study": "study"}
    internal_mode = mode_map.get(req.mode, "chat")
    pipeline.mode = internal_mode
    return {"mode": req.mode, "internal": internal_mode}


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
        new_persona        = req.persona        or _persona_config.persona
        new_tone           = req.tone           or _persona_config.tone
        new_length         = req.length         or _persona_config.length
        new_custom_persona = req.custom_persona or _persona_config.custom_persona
        _persona_config = PersonaConfig(
            persona=new_persona, tone=new_tone,
            length=new_length, custom_persona=new_custom_persona,
        )
        if hasattr(pipeline, "prompt_builder") and pipeline.prompt_builder:
            pipeline.prompt_builder.persona_config = _persona_config
        return {"status": "updated", "current": _persona_config.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_source(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    try:
        from src.ingestion.preprocessing.content_analyzer import ContentAnalyzer
        from src.ingestion.chunking.adaptive_chunker import AdaptiveChunker

        STRATEGIES = ["recursive", "paragraph", "page", "semantic", "hierarchical"]
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            body        = await request.json()
            url         = body.get("url", "").strip()
            text        = body.get("text", "").strip()
            source_type = body.get("source_type", "text").lower().strip()
            if url:
                try:
                    loop = asyncio.get_running_loop()   # BUG-009
                    fetched = await loop.run_in_executor(
                        None, lambda: pipeline.fetch_url_content(url))
                    source_content = fetched if fetched else url
                except Exception:
                    source_content = url
                source_name = url
            elif text:
                source_content = text
                source_name    = "pasted text"
            else:
                raise HTTPException(status_code=400, detail="Provide 'url' or 'text' in JSON body")
        else:
            if not file:
                raise HTTPException(status_code=400, detail="Provide a file or JSON body")
            # BUG-021: validate extension
            ext = Path(file.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
            form_data   = await request.form()
            source_type = str(form_data.get("source_type", "pdf")).lower().strip()
            raw = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")
            suffix = ext or ".tmp"
            tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(UPLOAD_DIR))
            tmp.write(raw)
            tmp.close()
            source_content = Path(tmp.name).read_text(errors="ignore")
            source_name    = file.filename

        # BUG-008: single ContentAnalyzer instantiation
        analyzer    = ContentAnalyzer()
        analysis    = analyzer.analyze(source_content, source_type=source_type)
        chunker     = AdaptiveChunker()
        recommended = chunker.recommend_strategy(source_content, source_type)
        chunks      = chunker.chunk(source_content, strategy=recommended)
        previews    = [{"index": i, "text": c.get("content", "")[:200]} for i, c in enumerate(chunks[:3])]
        return {
            "source_name": source_name, "source_type": source_type,
            "recommended_strategy": recommended, "available_strategies": STRATEGIES,
            "chunk_count_estimate": len(chunks), "analysis": analysis,
            "previews": previews, "token_stats": analysis.get("token_stats", {}),
            "embedding_models": EMBEDDING_MODELS,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/embedding-models")
def list_embedding_models():
    return {"models": EMBEDDING_MODELS}


@app.post("/api/ingest")
async def ingest_source(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    source_type: str = Form("pdf"),
    chunking_strategy: Optional[str] = Form(None),
    embedding_model: Optional[str] = Form(None),
):
    tmp_path = None
    try:
        if file:
            # BUG-021: validate extension and size
            ext = Path(file.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
            raw = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".tmp", dir=str(UPLOAD_DIR))
            tmp.write(raw)
            tmp.close()
            tmp_path = tmp.name
            loop   = asyncio.get_running_loop()   # BUG-009
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.ingest(file_path=tmp_path, url=None,
                    source_type=source_type, chunking_strategy=chunking_strategy,
                    embedding_model=embedding_model),
            )
        elif url:
            loop   = asyncio.get_running_loop()   # BUG-009
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.ingest(file_path=None, url=url,
                    source_type=source_type, chunking_strategy=chunking_strategy,
                    embedding_model=embedding_model),
            )
        else:
            raise HTTPException(status_code=400, detail="Provide file or url")
        return {"status": "ingested", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.get("/api/sources")
def list_sources():
    try:
        sources_dict = getattr(pipeline.storage_manager, "_sources", {}) or {}
        sources = list(sources_dict.values())
        return {"sources": sources, "total_sources": len(sources)}
    except Exception as e:
        return {"sources": [], "error": str(e)}


@app.delete("/api/sources/{source_id}")
def delete_source(source_id: str):
    try:
        ok = pipeline.delete_source(source_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")
        return {"status": "deleted", "source_id": source_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")


# ── query routes ──────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)   # BUG-014
async def query(req: QueryRequest):
    """
    Non-streaming query with automatic RAGAS evaluation.
    Returns full result + ragas field.
    """
    # BUG-015: guard before processing
    if not pipeline.llm:
        raise HTTPException(status_code=400, detail="LLM not configured. POST /api/config first.")
    try:
        _apply_tuning(req)
        # BUG-022: sanitize query
        safe_query = _sanitize_query(req.query)
        loop   = asyncio.get_running_loop()   # BUG-009
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.generate(
                safe_query, stream=False,
                persona_config=_persona_config if req.mode == "chat" else None,
            ),
        )
        # BUG-007: prefer deduplicated context_chunks for RAGAS eval
        context_chunks = (
            result.get("context_chunks")
            or result.get("retrieved_chunks", [])
        )
        ragas = await _run_evaluation(
            question=safe_query,
            answer=result.get("answer", ""),
            context_chunks=context_chunks,
        )
        result["ragas"] = ragas
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """
    Streaming SSE query.

    Event types emitted:
      {type: "token",    content: "..."}               — token chunk
      {type: "metadata", citations: [...], ...}         — end-of-stream metadata
      {type: "ragas",    faithfulness: 0.87, ...}       — grounding score (after done)
      {type: "done"}
      {type: "error",    detail: "..."}
    """
    # BUG-015: guard before returning StreamingResponse
    if not pipeline.llm:
        raise HTTPException(status_code=400, detail="LLM not configured. POST /api/config first.")

    async def event_generator() -> AsyncGenerator[str, None]:
        full_answer  = ""
        retrieved    = []
        context_cks  = []

        try:
            _apply_tuning(req)
            # BUG-022: sanitize query
            safe_query = _sanitize_query(req.query)

            # BUG-001: call generate() directly — NOT inside run_in_executor
            # generate(stream=True) returns a lazy generator; consuming it here
            # is synchronous but yields control between chunks via the SSE writes.
            result = pipeline.generate(
                safe_query, stream=True,
                persona_config=_persona_config if req.mode == "chat" else None,
            )

            if hasattr(result, "__iter__") and not isinstance(result, dict):
                for chunk in result:
                    if isinstance(chunk, str):
                        full_answer += chunk
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    elif isinstance(chunk, dict):
                        # metadata dict emitted by _stream_dict()
                        retrieved   = chunk.get("retrieved_chunks", retrieved)
                        context_cks = chunk.get("context_chunks",  context_cks)
                        yield f"data: {json.dumps({'type': 'metadata', **chunk})}\n\n"
            else:
                # Fallback: non-generator result (should not happen for stream=True)
                answer      = result.get("answer", "") if isinstance(result, dict) else str(result)
                full_answer = answer
                for word in answer.split(" "):
                    yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                if isinstance(result, dict):
                    retrieved   = result.get("retrieved_chunks", [])
                    context_cks = result.get("context_chunks", [])
                    meta = {k: result[k] for k in
                            ["citations", "chunks", "retrieved_chunks",
                             "sources_used", "sub_queries", "quiz_cards",
                             "summary_bullets", "learning_path"]
                            if k in result}
                    if meta:
                        yield f"data: {json.dumps({'type': 'metadata', **meta})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # BUG-002: fire-and-forget RAGAS using create_task so exceptions
            # don't silently vanish and the generator frame is already done.
            async def _eval_and_emit():
                try:
                    # BUG-007: use deduplicated context_chunks
                    chunks_for_eval = context_cks or retrieved
                    ragas = await _run_evaluation(
                        question=safe_query,
                        answer=full_answer,
                        context_chunks=chunks_for_eval,
                    )
                    # We can't yield here (generator done), so we just persist.
                    # The UI should call /api/ragas/history for the score.
                    pipeline._last_ragas = ragas
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Post-stream RAGAS eval failed: %s", exc
                    )

            asyncio.create_task(_eval_and_emit())

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    """On-demand RAGAS evaluation."""
    try:
        ragas = await _run_evaluation(
            question=req.question,
            answer=req.answer,
            context_chunks=req.context_chunks,
            ground_truth=req.ground_truth,
        )
        return ragas
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ragas/history")
def ragas_history(limit: int = 20):
    """Last N RAGAS evaluation results."""
    return {"history": list(_ragas_history)[:limit]}
