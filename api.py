#!/usr/bin/env python3
"""
api.py  –  FastAPI backend for Mini NotebookLM React UI
Run with:  uvicorn api:app --reload --port 8000

Changes (2026-05-10):
  Added /api/evaluate   — on-demand RAGAS evaluation
  Added /api/query/stream SSE event type "ragas" — grounding score inline
  Added /api/ragas/history  — last N evaluation results for the RAGAS panel
"""

import os
import asyncio
import tempfile
import json
from collections import deque
from pathlib import Path
from typing import AsyncGenerator, Deque, Optional

import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.master_pipeline import MasterPipeline
from src.generation.persona_config import PersonaConfig
from src.evaluation.ragas_evaluator import RAGASEvaluator

# ── app ─────────────────────────────────────────────────────────────────────────────
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

# ── global singletons ──────────────────────────────────────────────────────────────────
pipeline:        MasterPipeline = MasterPipeline(mode="chat")
_persona_config: PersonaConfig  = PersonaConfig()
_evaluator:      RAGASEvaluator = RAGASEvaluator()          # shared, lazy-loads embedder
_ragas_history:  Deque[dict]    = deque(maxlen=50)          # rolling window of evaluations

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODELS = [
    {"name": "all-MiniLM-L6-v2",      "dim": 384,  "max_tokens": 256,  "label": "MiniLM",   "speed": "fast",   "note": "Local · fastest"},
    {"name": "all-mpnet-base-v2",      "dim": 768,  "max_tokens": 384,  "label": "MPNet",    "speed": "medium", "note": "Local · balanced"},
    {"name": "e5-large-v2",            "dim": 1024, "max_tokens": 512,  "label": "E5-Large", "speed": "slow",   "note": "Local · most accurate"},
    {"name": "text-embedding-3-small", "dim": 1536, "max_tokens": 8191, "label": "OAI Small","speed": "fast",   "note": "OpenAI API key required"},
    {"name": "text-embedding-3-large", "dim": 3072, "max_tokens": 8191, "label": "OAI Large","speed": "medium", "note": "OpenAI API key required"},
]


# ── request / response models ──────────────────────────────────────────────────────────────────

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
    persona:        Optional[str] = None
    tone:           Optional[str] = None
    length:         Optional[str] = None
    custom_persona: Optional[str] = None
    reset:          bool = False

class EvaluateRequest(BaseModel):
    question:     str
    answer:       str
    context_chunks: list = []
    ground_truth: Optional[str] = None


# ── helpers ─────────────────────────────────────────────────────────────────────────────

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
    """Run RAGAS eval in thread pool (CPU-bound embedding), cache in history."""
    loop = asyncio.get_event_loop()
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
    return result_dict


# ── existing routes (unchanged) ─────────────────────────────────────────────────────────────

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
        pipeline.set_llm(provider=req.provider.lower(), model=req.model, api_key=req.api_key)
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
        analyzer   = ContentAnalyzer()
        STRATEGIES = ["recursive", "paragraph", "page", "semantic", "hierarchical"]
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body        = await request.json()
            url         = body.get("url", "").strip()
            text        = body.get("text", "").strip()
            source_type = body.get("source_type", "text").lower().strip()
            if url:
                try:
                    fetched = await asyncio.get_event_loop().run_in_executor(
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
            form_data   = await request.form()
            source_type = str(form_data.get("source_type", "pdf")).lower().strip()
            suffix = Path(file.filename).suffix or ".tmp"
            tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(UPLOAD_DIR))
            tmp.write(await file.read())
            tmp.close()
            source_content = Path(tmp.name).read_text(errors="ignore")
            source_name    = file.filename
        analyzer  = ContentAnalyzer()
        analysis  = analyzer.analyze(source_content, source_type=source_type)
        chunker   = AdaptiveChunker()
        recommended = chunker.recommend_strategy(source_content, source_type)
        chunks    = chunker.chunk(source_content, strategy=recommended)
        previews  = [{"index": i, "text": c.get("content", "")[:200]} for i, c in enumerate(chunks[:3])]
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
            suffix = Path(file.filename).suffix or ".tmp"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(UPLOAD_DIR))
            tmp.write(await file.read())
            tmp.close()
            tmp_path = tmp.name
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pipeline.ingest(file_path=tmp_path, url=None,
                    source_type=source_type, chunking_strategy=chunking_strategy,
                    embedding_model=embedding_model),
            )
        elif url:
            result = await asyncio.get_event_loop().run_in_executor(
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
            try: os.unlink(tmp_path)
            except Exception: pass


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


# ── query routes ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/query")
async def query(req: QueryRequest):
    """
    Non-streaming query with automatic RAGAS evaluation.
    Returns full result + ragas field.
    """
    try:
        _apply_tuning(req)
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipeline.generate(
                req.query, stream=False,
                persona_config=_persona_config if req.mode == "chat" else None,
            ),
        )
        # Auto-evaluate after response
        context_chunks = result.get("context_chunks") or result.get("retrieved_chunks", [])
        ragas = await _run_evaluation(
            question=req.query,
            answer=result.get("answer", ""),
            context_chunks=context_chunks,
        )
        result["ragas"] = ragas
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """
    Streaming SSE query.

    Event types emitted:
      {type: "token",    content: "..."}               — token chunk
      {type: "metadata", citations: [...], ...}         — end-of-stream metadata
      {type: "ragas",    faithfulness: 0.87, grade: "Good", overall_score: 0.81, ...}
      {type: "done"}
      {type: "error",    detail: "..."}

    The "ragas" event is emitted AFTER "done" so the UI can display the
    grounding score as a small badge below the completed answer.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            loop = asyncio.get_event_loop()
            _apply_tuning(req)

            result = await loop.run_in_executor(
                None,
                lambda: pipeline.generate(
                    req.query, stream=True,
                    persona_config=_persona_config if req.mode == "chat" else None,
                ),
            )

            full_answer = ""
            retrieved   = []

            if hasattr(result, "__iter__") and not isinstance(result, dict):
                for chunk in result:
                    if isinstance(chunk, str):
                        full_answer += chunk
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    elif isinstance(chunk, dict):
                        retrieved = chunk.get("retrieved_chunks", [])
                        yield f"data: {json.dumps({'type': 'metadata', **chunk})}\n\n"
            else:
                answer = result.get("answer", "") if isinstance(result, dict) else str(result)
                full_answer = answer
                for word in answer.split(" "):
                    yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                if isinstance(result, dict):
                    retrieved = result.get("context_chunks") or result.get("retrieved_chunks", [])
                    meta = {k: result[k] for k in
                            ["citations","chunks","retrieved_chunks","sources_used"]
                            if k in result}
                    if meta:
                        yield f"data: {json.dumps({'type': 'metadata', **meta})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # Evaluate asynchronously AFTER streaming is complete
            try:
                ragas = await _run_evaluation(
                    question=req.query,
                    answer=full_answer,
                    context_chunks=retrieved,
                )
                # Emit only the lightweight inline fields first (shown below the message)
                inline = {
                    "type":          "ragas",
                    "faithfulness":  ragas["faithfulness"],
                    "overall_score": ragas["overall_score"],
                    "grade":         ragas["grade"],
                    "answer_relevance": ragas["answer_relevance"],
                    "context_precision": ragas["context_precision"],
                }
                yield f"data: {json.dumps(inline)}\n\n"
            except Exception as eval_exc:
                # Evaluation errors must never break the chat
                pass

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── RAGAS endpoints ────────────────────────────────────────────────────────────────────────────

@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    """
    On-demand RAGAS evaluation for any question/answer/context triple.
    Can also accept ground_truth for recall + similarity metrics.
    Used by the RAGAS panel's "Re-evaluate" button.
    """
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
    """
    Return the last `limit` RAGAS evaluations for the history table in the
    RAGAS Dashboard panel.
    """
    items = list(_ragas_history)[:limit]
    if not items:
        return {"history": [], "avg": {}}
    avg = {
        "faithfulness":       round(sum(i["faithfulness"]       for i in items) / len(items), 3),
        "answer_relevance":   round(sum(i["answer_relevance"]   for i in items) / len(items), 3),
        "context_precision":  round(sum(i["context_precision"]  for i in items) / len(items), 3),
        "overall_score":      round(sum(i["overall_score"]      for i in items) / len(items), 3),
    }
    return {"history": items, "avg": avg, "total": len(items)}


@app.get("/api/ragas/summary")
def ragas_summary():
    """
    Lightweight summary used by the RAGAS badge in the status bar.
    Returns averages across the current session.
    """
    items = list(_ragas_history)
    if not items:
        return {"avg_faithfulness": None, "avg_overall": None, "total_evaluated": 0}
    return {
        "avg_faithfulness":  round(sum(i["faithfulness"]   for i in items) / len(items), 3),
        "avg_overall":       round(sum(i["overall_score"]  for i in items) / len(items), 3),
        "avg_relevance":     round(sum(i["answer_relevance"] for i in items) / len(items), 3),
        "avg_precision":     round(sum(i["context_precision"] for i in items) / len(items), 3),
        "total_evaluated":   len(items),
    }


@app.post("/api/new-chat")
def new_chat():
    try:
        pipeline.chat_history.clear()
        return {"status": "reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
def get_history():
    try:
        history = pipeline.chat_history.get_history_context("", max_messages=100)
        return {"history": history}
    except Exception as e:
        return {"history": [], "error": str(e)}


@app.get("/api/graph")
def get_graph():
    try:
        kg    = pipeline.storage_manager.knowledge_graph
        nodes = list(kg.graph.nodes(data=True))[:120]
        edges = list(kg.graph.edges(data=True))[:300]
        return {
            "nodes": [{"id": n, **d} for n, d in nodes],
            "edges": [{"from": u, "to": v, **d} for u, v, d in edges],
        }
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}
