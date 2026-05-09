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
    allow_origins=["http://localhost:5173", "http://localhost:4173", "http://localhost:3000"],
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
        # Normalise keys so the frontend never sees KeyError
        stats.setdefault("total_chunks", stats.get("chunks", {}).get("total_chunks", 0))
        stats.setdefault("total_sources", stats.get("sources", 0))
        stats.setdefault("graph", {"nodes": 0, "edges": 0})
        stats.setdefault("chunks", {"total_chunks": 0, "dimensions": {}})
        return stats
    except Exception as e:
        return {
            "total_chunks": 0, "total_sources": 0,
            "graph": {"nodes": 0, "edges": 0},
            "chunks": {"total_chunks": 0, "dimensions": {}},
            "error": str(e)
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
    (Fixes the set_mode() bug in master_pipeline.py where history was wiped.)
    """
    mode_map = {"chat": "chat", "deep": "deep_research", "study": "study"}
    internal_mode = mode_map.get(req.mode, "chat")
    # Directly set mode attribute — do NOT call pipeline.set_mode() which recreates history
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
    Feeds the DataAnalysis component in the sidebar.
    """
    try:
        from src.ingestion.preprocessing.content_analyzer import ContentAnalyzer
        from src.ingestion.chunking.adaptive_chunker import AdaptiveChunker

        analyzer = ContentAnalyzer()
        STRATEGIES = ["recursive", "paragraph", "page", "semantic", "hierarchical"]

        if file:
            suffix = Path(file.filename).suffix or ".tmp"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(UPLOAD_DIR))
            tmp.write(await file.read())
            tmp.close()
            source_path = tmp.name
        elif url:
            source_path = url
        else:
            raise HTTPException(400, "Provide a file or URL")

        analysis = analyzer.analyze(source_path)
        recommendation = analysis.get("recommendation", {})

        # Best-effort chunk previews for each strategy
        previews: dict = {}
        try:
            adaptive = AdaptiveChunker()
            for strategy in STRATEGIES:
                chunks = adaptive.chunk(source_path, strategy=strategy)
                previews[strategy] = [
                    {"content": c.content[:300], "token_count": getattr(c, "token_count", 0)}
                    for c in chunks[:3]
                ]
        except Exception:
            pass

        return {
            "analysis": analysis,
            "previews": previews,
            "recommendation": recommendation,
            "estimated_tokens": analysis.get("estimated_tokens", 0),
            "avg_tokens_per_paragraph": analysis.get("avg_tokens_per_paragraph", 0),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")


@app.post("/api/ingest")
async def ingest_source(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    source_type: str = Form("pdf"),
    strategy: str = Form("recursive"),
    embedding_model: str = Form("all-MiniLM-L6-v2"),
):
    """Ingest a file or URL into the knowledge base."""
    try:
        if file:
            suffix = Path(file.filename).suffix or ".tmp"
            dest = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
            dest.write_bytes(await file.read())
            # master_pipeline.ingest() only accepts file_path and url as named args
            source_id = pipeline.ingest(
                file_path=str(dest),
                source_type=source_type,
            )
            return {
                "source_id": source_id,
                "name": file.filename,
                "type": source_type,
                "status": "ready",
            }

        elif url:
            source_id = pipeline.ingest(
                url=url,
                source_type=source_type,
            )
            return {
                "source_id": source_id,
                "name": url,
                "type": source_type,
                "status": "ready",
            }

        else:
            raise HTTPException(400, "Provide a file or URL")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")


@app.get("/api/sources")
def list_sources():
    """Return all ingested sources for the left sidebar."""
    try:
        sources = pipeline.source_manager.get_all_sources()
        return {"sources": sources}
    except Exception as e:
        return {"sources": [], "error": str(e)}


@app.delete("/api/sources/{source_id}")
def delete_source(source_id: str):
    """Remove a source from the knowledge base."""
    try:
        pipeline.source_manager.remove_source(source_id)
        return {"status": "deleted", "source_id": source_id}
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {e}")


@app.post("/api/query")
async def query(req: QueryRequest):
    """
    Query the pipeline with SSE streaming support.
    Frontend reads: `data: <token>\\n\\n` until `data: [DONE]\\n\\n`
    """
    mode_map = {"chat": "chat", "deep": "deep_research", "study": "study"}
    pipeline.mode = mode_map.get(req.mode, "chat")

    if not pipeline.llm:
        raise HTTPException(400, "LLM not configured — POST /api/config first.")

    if req.stream:
        async def event_stream() -> AsyncGenerator[str, None]:
            try:
                # Check if LLM supports native streaming
                if hasattr(pipeline.llm, "stream"):
                    # Use real token streaming via _stream_response
                    query_embedding = pipeline.embedder.embed_query(req.query)
                    if pipeline.mode == "chat":
                        retrieved = pipeline.hybrid_retriever.retrieve(req.query, query_embedding)
                    elif pipeline.mode == "deep_research":
                        retrieved = pipeline.advanced_retriever.retrieve(req.query, query_embedding)
                    else:
                        study_result = pipeline.study_retriever.retrieve(req.query, query_embedding)
                        retrieved = study_result["chunks"]

                    history_context = pipeline.chat_history.get_history_context(req.query)
                    context = pipeline.prompt_builder.format_context(retrieved)

                    if pipeline.mode == "chat":
                        prompt = pipeline.prompt_builder.build_chat_prompt(req.query, context, history_context)
                    elif pipeline.mode == "deep_research":
                        prompt = pipeline.prompt_builder.build_deep_research_prompt(req.query, context, history_context)
                    else:
                        prompt = pipeline.prompt_builder.build_study_mode_prompt(req.query, context, [], history_context)

                    full = []
                    for token in pipeline.llm.stream(prompt):
                        full.append(token)
                        yield f"data: {token}\n\n"
                    pipeline.chat_history.add_message("user", req.query)
                    pipeline.chat_history.add_message("assistant", "".join(full))
                else:
                    # Fallback: get full response then word-stream it
                    response = pipeline.generate(req.query)
                    if asyncio.iscoroutine(response):
                        response = await response
                    for word in response.split(" "):
                        yield f"data: {word} \n\n"
                        await asyncio.sleep(0.012)

                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: [ERROR] {e}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        try:
            response = pipeline.generate(req.query)
            if asyncio.iscoroutine(response):
                response = await response
            return {"response": response}
        except Exception as e:
            raise HTTPException(500, str(e))


@app.post("/api/web-search")
async def web_search(query: str = Form(...)):
    """Web search — returns results for the sidebar search panel."""
    try:
        results = pipeline.web_search.search_and_format(query)
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.post("/api/web-ingest")
async def web_ingest(url: str = Form(...)):
    """Directly ingest a URL from web search results."""
    try:
        source_id = pipeline.ingest(url=url, source_type="website")
        return {"source_id": source_id, "status": "ready"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/new-chat")
def new_chat():
    """Start a fresh chat session (keeps sources intact)."""
    try:
        pipeline.chat_history.new_session()
        pipeline.clear_cache()
        return {"status": "reset"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/llm-models")
def list_models(provider: str = "groq"):
    """Available models per provider — used by the config dropdown."""
    models = {
        "groq":   ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
        "ollama": ["llama3", "mistral", "phi3", "gemma2"],
    }
    return {"models": models.get(provider.lower(), [])}
