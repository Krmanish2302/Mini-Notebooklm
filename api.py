#!/usr/bin/env python3
"""
api.py  –  FastAPI backend for Mini NotebookLM React UI
Run with:  uvicorn api:app --reload --port 8000
"""

import os
import asyncio
import tempfile
import json
from pathlib import Path
from typing import AsyncGenerator, Optional

import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.master_pipeline import MasterPipeline
from src.generation.persona_config import PersonaConfig

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

# ── global persona config (Chat mode only — Study/Research use fixed personas) ─
_persona_config: PersonaConfig = PersonaConfig()   # default: sagan / neutral / medium

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── embedding model catalogue (single source of truth) ────────────────────────
EMBEDDING_MODELS = [
    {"name": "all-MiniLM-L6-v2",      "dim": 384,  "max_tokens": 256,  "label": "MiniLM",   "speed": "fast",   "note": "Local · fastest"},
    {"name": "all-mpnet-base-v2",      "dim": 768,  "max_tokens": 384,  "label": "MPNet",    "speed": "medium", "note": "Local · balanced"},
    {"name": "e5-large-v2",            "dim": 1024, "max_tokens": 512,  "label": "E5-Large", "speed": "slow",   "note": "Local · most accurate"},
    {"name": "text-embedding-3-small", "dim": 1536, "max_tokens": 8191, "label": "OAI Small","speed": "fast",   "note": "OpenAI API key required"},
    {"name": "text-embedding-3-large", "dim": 3072, "max_tokens": 8191, "label": "OAI Large","speed": "medium", "note": "OpenAI API key required"},
]


# ── request / response models ─────────────────────────────────────────────────
class ConfigRequest(BaseModel):
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    api_key: str


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
    """
    All fields optional — only the provided fields are updated.
    Send {} to reset to defaults.
    """
    persona:        Optional[str] = None
    tone:           Optional[str] = None
    length:         Optional[str] = None
    custom_persona: Optional[str] = None
    reset:          bool = False   # set True to restore sagan/neutral/medium


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
    Sets pipeline.mode directly — does NOT call pipeline.set_mode() which
    would recreate the session UUID and wipe history.
    """
    mode_map = {"chat": "chat", "deep": "deep_research", "study": "study"}
    internal_mode = mode_map.get(req.mode, "chat")
    pipeline.mode = internal_mode
    return {"mode": req.mode, "internal": internal_mode}


# ── Persona endpoints ─────────────────────────────────────────────────────────

@app.get("/api/persona")
def get_persona():
    """
    Return the current persona config + the full catalogue of available
    options (for building the UI dropdowns).
    """
    return {
        "current": _persona_config.to_dict(),
        "catalogue": PersonaConfig.catalogue(),
    }


@app.post("/api/persona")
def set_persona(req: PersonaRequest):
    """
    Update persona settings for Chat mode.
    Partial updates are supported — only sent fields are changed.
    Set `reset: true` to restore defaults.
    """
    global _persona_config
    try:
        if req.reset:
            _persona_config = PersonaConfig()
            return {"status": "reset", "current": _persona_config.to_dict()}

        # Merge: start from current values, apply only what was sent
        new_persona        = req.persona        or _persona_config.persona
        new_tone           = req.tone           or _persona_config.tone
        new_length         = req.length         or _persona_config.length
        new_custom_persona = req.custom_persona or _persona_config.custom_persona

        _persona_config = PersonaConfig(
            persona=new_persona,
            tone=new_tone,
            length=new_length,
            custom_persona=new_custom_persona,
        )
        # Inject updated config into pipeline prompt builder
        if hasattr(pipeline, "prompt_builder") and pipeline.prompt_builder:
            pipeline.prompt_builder.persona_config = _persona_config

        return {"status": "updated", "current": _persona_config.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Analyze / Ingest ──────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_source(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    source_type: str = Form("pdf"),
):
    """
    Pre-ingest analysis: returns chunking recommendation + chunk previews
    + per-strategy token stats + embedding model catalogue.
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
            source_content = Path(tmp.name).read_text(errors="ignore")
            source_name = file.filename
        elif url:
            source_content = url
            source_name = url
        else:
            raise HTTPException(status_code=400, detail="Provide file or url")

        analysis = analyzer.analyze(source_content, source_type=source_type)
        chunker = AdaptiveChunker()
        recommended = chunker.recommend_strategy(source_content, source_type)
        chunks = chunker.chunk(source_content, strategy=recommended)
        previews = [
            {"index": i, "text": c.get("content", "")[:200]}
            for i, c in enumerate(chunks[:3])
        ]

        return {
            "source_name":          source_name,
            "source_type":          source_type,
            "recommended_strategy": recommended,
            "available_strategies": STRATEGIES,
            "chunk_count_estimate": len(chunks),
            "analysis":             analysis,
            "previews":             previews,
            "token_stats":          analysis.get("token_stats", {}),
            "embedding_models":     EMBEDDING_MODELS,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/embedding-models")
def list_embedding_models():
    """Return available embedding models with token limits and dims for the UI model picker."""
    return {"models": EMBEDDING_MODELS}


@app.post("/api/ingest")
async def ingest_source(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    source_type: str = Form("pdf"),
    chunking_strategy: Optional[str] = Form(None),
    embedding_model: Optional[str] = Form(None),
):
    """
    Full ingest: file or URL → pipeline → FAISS + KG + SQLite.
    chunking_strategy and embedding_model are forwarded to the pipeline.
    Returns source metadata when done.
    """
    tmp_path = None
    try:
        if file:
            suffix = Path(file.filename).suffix or ".tmp"
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir=str(UPLOAD_DIR)
            )
            tmp.write(await file.read())
            tmp.close()
            tmp_path = tmp.name
            result = await asyncio.get_event_loop().run_in_executor(
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
            result = await asyncio.get_event_loop().run_in_executor(
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
    """Return all ingested sources for the left sidebar."""
    try:
        sources_dict = getattr(pipeline.storage_manager, "_sources", {}) or {}
        sources = list(sources_dict.values())
        return {"sources": sources, "total_sources": len(sources)}
    except Exception as e:
        return {"sources": [], "error": str(e)}


@app.delete("/api/sources/{source_id}")
def delete_source(source_id: str):
    """Remove a source and all its chunks from the knowledge base."""
    try:
        ok = pipeline.delete_source(source_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")
        return {"status": "deleted", "source_id": source_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")


@app.post("/api/query")
async def query(req: QueryRequest):
    """
    Non-streaming query. Returns full response + citations + retrieved chunks.
    Passes the current persona config to the pipeline for Chat mode.
    """
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipeline.generate(
                req.query,
                stream=False,
                persona_config=_persona_config if req.mode == "chat" else None,
            ),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """
    Streaming query via SSE.
    Event format:
      data: {"type": "token",    "content": "..."}
      data: {"type": "metadata", "citations": [...], "chunks": [...]}
      data: {"type": "done"}
      data: {"type": "error",    "detail": "..."}
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            loop = asyncio.get_event_loop()
            # Apply per-request generation tuning if the UI sent any sliders
            if any(v is not None for v in (req.temperature, req.top_p, req.max_tokens)):
                if pipeline.llm:
                    pipeline.llm.update_tuning(
                        **{k: v for k, v in {
                            "temperature": req.temperature,
                            "top_p":       req.top_p,
                            "max_tokens":  req.max_tokens,
                        }.items() if v is not None}
                    )
            
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.generate(
                    req.query,
                    stream=True,
                    persona_config=_persona_config if req.mode == "chat" else None,
                ),
            )
            
            if hasattr(result, "__iter__") and not isinstance(result, dict):
                for chunk in result:
                    if isinstance(chunk, str):
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    elif isinstance(chunk, dict):
                        yield f"data: {json.dumps({'type': 'metadata', **chunk})}\n\n"
            else:
                answer = result.get("answer", "") if isinstance(result, dict) else str(result)
                for word in answer.split(" "):
                    yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                meta_keys = ["citations", "chunks", "retrieved_chunks", "sources_used"]
                meta = {k: result[k] for k in meta_keys if isinstance(result, dict) and k in result}
                if meta:
                    yield f"data: {json.dumps({'type': 'metadata', **meta})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/new-chat")
def new_chat():
    """Start a fresh chat session (keeps sources intact)."""
    try:
        pipeline.chat_history.clear()
        return {"status": "reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
def get_history():
    """Return current chat history for the sidebar."""
    try:
        history = pipeline.chat_history.get_history_context("", max_messages=100)
        return {"history": history}
    except Exception as e:
        return {"history": [], "error": str(e)}


@app.get("/api/graph")
def get_graph():
    """
    Return knowledge graph nodes + edges for the MiniGraph component.
    Capped at 120 nodes / 300 edges to keep the UI responsive.
    """
    try:
        kg = pipeline.storage_manager.knowledge_graph
        nodes = list(kg.graph.nodes(data=True))[:120]
        edges = list(kg.graph.edges(data=True))[:300]
        return {
            "nodes": [{"id": n, **d} for n, d in nodes],
            "edges": [{"from": u, "to": v, **d} for u, v, d in edges],
        }
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}
