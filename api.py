#!/usr/bin/env python3
"""
api.py — FastAPI backend for Mini NotebookLM
Run: uvicorn api:app --reload --port 8000

Fix pass v1.4.3
---------------
* Fix #7 : QueryRequest gains do_expand field; forwarded into retrieval state
* All prior fixes (v1.4.2) retained
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONUTF8", "1")

import asyncio
import json
import logging
import re
import tempfile
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Deque, List, Optional

from dotenv import load_dotenv
load_dotenv()

import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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

_MODE_MAP = {
    "chat":          "chat",
    "deep":          "research",
    "research":      "research",
    "deep_research": "research",
    "study":         "study",
    "analyze":       "chat",
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

def _fetch_url_content(url: str) -> str:
    """Fetch plain text from a URL using LangChain WebBaseLoader."""
    try:
        from langchain_community.document_loaders import WebBaseLoader
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        docs = WebBaseLoader(url, requests_kwargs={"headers": headers}).load()
        return "\n\n".join(d.page_content for d in docs)
    except Exception as exc:
        logger.warning("[_fetch_url_content] Failed to fetch %s: %s", url, exc)
        return ""

# ── Globals ────────────────────────────────────────────────────────────────────

pipeline:       MiniNotebookLM
_persona:       PersonaConfig
_ragas_history: Deque[dict]


def safe_json_dumps(obj: Any) -> str:
    import numpy as np
    
    def convert(item: Any) -> Any:
        if isinstance(item, dict):
            return {k: convert(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert(i) for i in item]
        elif isinstance(item, tuple):
            return tuple(convert(i) for i in item)
        elif hasattr(item, "dict") and callable(getattr(item, "dict")):
            try:
                return convert(item.dict())
            except Exception:
                pass
        elif hasattr(item, "model_dump") and callable(getattr(item, "model_dump")):
            try:
                return convert(item.model_dump())
            except Exception:
                pass
        
        # Check by class name to be safe with any numpy/pandas types
        tname = type(item).__name__.lower()
        if "float" in tname:
            try:
                return float(item)
            except Exception:
                pass
        if "int" in tname or "uint" in tname:
            try:
                return int(item)
            except Exception:
                pass
        if "bool" in tname:
            try:
                return bool(item)
            except Exception:
                pass
        if "ndarray" in tname or hasattr(item, "tolist"):
            try:
                return convert(item.tolist())
            except Exception:
                pass

        try:
            json.dumps(item)
            return item
        except TypeError:
            return str(item)

    return json.dumps(convert(obj))


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

app = FastAPI(title="Mini NotebookLM API", version="1.4.3", lifespan=lifespan)

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
    model:    str       = "llama-3.3-70b-versatile"
    api_key:  SecretStr

class QueryRequest(BaseModel):
    query:        str
    mode:         str             = "chat"
    stream:       bool            = True
    temperature:  Optional[float] = None
    top_p:        Optional[float] = None
    max_tokens:   Optional[int]   = None
    ground_truth: Optional[str]   = None
    source_ids:   List[str]       = []
    # FIX #7: expose do_expand so clients can disable query expansion
    # (e.g. for short factual queries where expansion degrades precision)
    do_expand:    bool            = True

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

class SearchAgentRequest(BaseModel):
    query: str
    max_results: int = 5

class IngestAgentRequest(BaseModel):
    url: str
    source_id: Optional[str] = None
    source_name: Optional[str] = None

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
        from src.storage.sqlite_manager import SQLiteManager
        db = SQLiteManager()
        return db.list_sources()
    except Exception as exc:
        logger.warning("[_list_sources_safe] Failed to read sources from SQLite: %s", exc)
        return []

def _ingest_via_router(
    source_type:     str,
    source_id:       str,
    file_path:       Optional[str] = None,
    content:         Optional[str] = None,
    strategy:        str           = "paragraph_based",
    source_name:     Optional[str] = None,
    embedding_model: Optional[str] = None,
    start_page:      int           = 1,
) -> dict:
    from src.ingestion.ingestion_router import ingest
    return ingest(
        source_type     = source_type,
        source_id       = source_id,
        file_path       = file_path,
        content         = content,
        strategy        = strategy,
        embedding_model = embedding_model,
        start_page      = start_page,
        source_name     = source_name,
    )
# ── Agent Web Search ──────────────────────────────────────────────────────────

@app.post("/api/agent/search")
async def agent_search(req: SearchAgentRequest):
    try:
        from src.agents.web_search_agent import WebSearchAgent
        agent = WebSearchAgent(max_results=req.max_results)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, agent.search, req.query)
        return {"results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/agent/fetch-and-ingest")
async def agent_fetch_and_ingest(req: IngestAgentRequest):
    try:
        from src.agents.web_search_agent import WebSearchAgent
        agent = WebSearchAgent()
        loop = asyncio.get_running_loop()
        content = await loop.run_in_executor(None, agent.fetch_content, req.url)
        if content.startswith("Error:"):
            raise HTTPException(status_code=422, detail=content)
            
        sid = (req.source_id or "").strip() or "web_" + req.url.split("//")[-1][:16].replace("/", "_")
        sname = req.source_name or req.url

        # Now ingest it as a website/text source
        result = await loop.run_in_executor(
            None,
            lambda: _ingest_via_router(
                source_type="website",
                source_id=sid,
                content=content,
                source_name=sname
            )
        )
        return {"status": "ingested", "source_id": sid, "result": result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    s = pipeline.status()
    return {"status": "ok", "version": "1.4.3", **s}

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
    PDF_STRATEGIES = [
        "paragraph_based", "sentence_based", "fixed_size",
        "semantic", "recursive", "page_based",
    ]
    content_type = request.headers.get("content-type", "")
    try:
        from src.ingestion.preprocessing.content_analyzer import ContentAnalyzer
        from src.ingestion.chunking.chunker_registry import ChunkerRegistry
        from langchain_core.documents import Document

        start_page = 1
        if "application/json" in content_type:
            body        = await request.json()
            url         = body.get("url",  "").strip()
            text        = body.get("text", "").strip()
            source_type = body.get("source_type", "text").lower().strip()
            try:
                start_page = int(body.get("start_page", 1))
            except (ValueError, TypeError):
                start_page = 1
                
            if url:
                loop           = asyncio.get_running_loop()
                source_content = await loop.run_in_executor(None, lambda: _fetch_url_content(url))
                source_name    = url
                if not source_content:
                    raise HTTPException(status_code=422, detail=f"Could not fetch content from URL: {url}")
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
            try:
                start_page = int(form_data.get("start_page", 1))
            except (ValueError, TypeError):
                start_page = 1
                
            raw         = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds 50 MB")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".tmp", dir=str(UPLOAD_DIR))
            tmp.write(raw); tmp.flush(); tmp.close()
            
            if source_type == "pdf" or ext == ".pdf":
                from src.ingestion.pdf_pipeline import analyze_pdf
                try:
                    analysis_result = analyze_pdf(tmp.name, file.filename or "upload", start_page=start_page)
                    os.unlink(tmp.name)
                    # Add standard UI mapping helper keys to match what Streamlit expects
                    analysis_result["source_name"] = file.filename or "upload"
                    analysis_result["source_type"] = "pdf"
                    analysis_result["recommended_strategy"] = "paragraph_based"
                    analysis_result["available_strategies"] = PDF_STRATEGIES
                    analysis_result["embedding_models"] = EMBEDDING_MODELS
                    return analysis_result
                except Exception as e:
                    logger.warning("Failed to analyze PDF using analyze_pdf, falling back: %s", e)
                    from langchain_community.document_loaders import PyMuPDFLoader
                    try:
                        loader = PyMuPDFLoader(tmp.name)
                        docs = loader.load()
                        source_content = "\n\n".join(d.page_content for d in docs)
                    except Exception as loader_exc:
                        logger.warning("Failed to extract PDF text using PyMuPDFLoader: %s", loader_exc)
                        source_content = Path(tmp.name).read_text(errors="ignore")
            else:
                source_content = Path(tmp.name).read_text(errors="ignore")
                
            source_name    = file.filename or "upload"
            os.unlink(tmp.name)

        doc = Document(page_content=source_content, metadata={"source_id": source_name, "source_type": source_type})
        analyzer = ContentAnalyzer()
        analysis = analyzer.analyze([doc])[0]
        
        # Build token stats
        word_count = analysis.get("word_count", 0)
        token_estimate = int(word_count * 1.33)
        token_stats = {
            "word_count": word_count,
            "char_count": analysis.get("char_count", 0),
            "token_estimate": token_estimate
        }
        analysis["token_stats"] = token_stats

        # Recommend strategy
        if source_type == "pdf":
            recommended = "paragraph"
        elif analysis.get("has_headings"):
            recommended = "hierarchical"
        elif word_count > 10000:
            recommended = "recursive"
        else:
            recommended = "paragraph"

        try:
            chunker = ChunkerRegistry.get(recommended)
            chunks = chunker.chunk_documents([doc])
        except Exception:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents([doc])

        previews = [{"index": i, "text": c.page_content[:200]} for i, c in enumerate(chunks[:3])]
        return {
            "source_name": source_name, "source_type": source_type,
            "recommended_strategy": recommended, "available_strategies": STRATEGIES,
            "chunk_count_estimate":  len(chunks),
            "analysis": analysis, "previews": previews,
            "token_stats": token_stats,
            "embedding_models": EMBEDDING_MODELS,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to analyze source: %s", exc)
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
    source_name:       Optional[str]        = Form(None),
    start_page:        Optional[int]        = Form(1),
):
    import hashlib
    import re
    raw_sid = (source_id or "").strip()
    if not raw_sid or raw_sid == "yt_https" or raw_sid == "yt_http" or "://" in raw_sid or ":" in raw_sid or "/" in raw_sid:
        if url:
            if "youtube.com" in url or "youtu.be" in url:
                match = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})', url)
                yt_id = match.group(1) if match else hashlib.md5(url.encode()).hexdigest()[:8]
                raw_sid = f"yt_{yt_id}"
            else:
                domain = url.split("//")[-1].split("/")[0]
                hash_part = hashlib.md5(url.encode()).hexdigest()[:6]
                raw_sid = f"web_{domain}_{hash_part}"
        elif file:
            raw_sid = file.filename or str(uuid.uuid4())[:8]
        else:
            raw_sid = str(uuid.uuid4())[:8]

    # Sanitize: replace any non-alphanumeric, non-dash, non-underscore with underscore
    sid = re.sub(r'[^a-zA-Z0-9_\-]', '_', raw_sid)
    sid = re.sub(r'_+', '_', sid).strip('_')
    if not sid or sid in ("yt_http", "yt_https"):
        sid = str(uuid.uuid4())[:8]

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
            tmp.write(raw); tmp.flush(); tmp.close()
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

        loop = asyncio.get_running_loop()

        try:
            result = await loop.run_in_executor(
                None,
                lambda: _ingest_via_router(
                    source_type     = source_type,
                    source_id       = sid,
                    file_path       = resolved_path,
                    content         = resolved_content,
                    strategy        = strategy,
                    source_name     = source_name or file.filename if file else source_name,
                    embedding_model = embedding_model,
                    start_page      = start_page,
                ),
            )
            return {"status": "ingested", "source_id": sid, "router": "ingestion_router", "result": result}
        except ImportError:
            logger.warning("/api/ingest: ingestion_router not available, falling back")

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
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


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
        from src.storage.sqlite_manager import SQLiteManager
        import shutil
        db = SQLiteManager()
        source = db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")
        
        # Delete from SQLite database
        db.delete_chunks_by_source(source_id)
        db.delete_source(source_id)
        
        # Delete FAISS vectorstore files from disk
        store_path = os.path.join("data/vectorstores", source_id)
        if os.path.exists(store_path):
            shutil.rmtree(store_path)
            logger.info("[delete_source] Deleted FAISS directory: %s", store_path)
            
        return {"status": "deleted", "source_id": source_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[delete_source] Failed to delete source: %s", exc)
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
                # FIX #7: forward do_expand so retrieval graph respects it
                do_expand=req.do_expand,
            ),
        )
        context_chunks = gen_result.chunks_used or []
        ragas = await _run_ragas(safe_query, gen_result.answer, context_chunks, req.ground_truth)
        res_dict = {
            "answer": gen_result.answer,
            "citations": gen_result.citations,
            "follow_ups": gen_result.follow_ups,
            "sources_used": gen_result.sources_used,
            "chunks_used": len(gen_result.chunks_used),
            "tokens_estimate": gen_result.tokens_estimate,
            "ragas": ragas,
        }
        serializable_data = json.loads(safe_json_dumps(res_dict))
        return JSONResponse(content=serializable_data)
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
            import queue
            import threading
            loop = asyncio.get_running_loop()
            q = queue.Queue()

            def run_ask_stream():
                try:
                    for event in pipeline.ask_stream(
                        safe_query,
                        mode=internal_mode,
                        persona=_persona if internal_mode == "chat" else None,
                        evaluate=False,
                        ground_truth=req.ground_truth,
                        source_ids=_sids,
                        do_expand=req.do_expand,
                    ):
                        q.put(event)
                    q.put(None)
                except Exception as exc:
                    q.put(exc)

            threading.Thread(target=run_ask_stream, daemon=True).start()

            while True:
                event = await loop.run_in_executor(None, q.get)
                if event is None:
                    break
                if isinstance(event, Exception):
                    raise event

                etype = event.get("type")
                if etype == "token":
                    full_answer += event.get("content", "")
                elif etype == "metadata":
                    chunks_used = event.get("chunks_used", [])

                # Serialize event with fallback for metadata
                if etype == "metadata":
                    try:
                        yield f"data: {safe_json_dumps(event)}\n\n"
                    except Exception as ser_exc:
                        logger.warning("Metadata serialization failed (%s), sending simplified metadata", ser_exc)
                        # Build a safe fallback without chunks_used (the most likely offender)
                        safe_event = {}
                        for k, v in event.items():
                            if k == "chunks_used":
                                safe_event[k] = []
                                continue
                            try:
                                json.dumps(v)
                                safe_event[k] = v
                            except (TypeError, ValueError):
                                safe_event[k] = str(v)
                        yield f"data: {json.dumps(safe_event)}\n\n"
                else:
                    try:
                        yield f"data: {safe_json_dumps(event)}\n\n"
                    except Exception as ser_exc:
                        logger.warning("Event serialization failed for type=%s: %s", etype, ser_exc)

            if full_answer and chunks_used:
                async def _eval():
                    try:
                        ragas = await _run_ragas(safe_query, full_answer, chunks_used, req.ground_truth)
                        pipeline.last_ragas = ragas
                    except Exception as exc:
                        logger.warning("Post-stream RAGAS failed: %s", exc)
                asyncio.create_task(_eval())

        except Exception as exc:
            yield f"data: {safe_json_dumps({'type': 'error', 'detail': str(exc)})}\n\n"

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
