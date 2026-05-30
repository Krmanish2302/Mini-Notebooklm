"""
master_pipeline.py — MiniNotebookLM: the single orchestration class.

Wires together:
  ┌─────────────┐    ┌───────────────┐    ┌──────────────────┐
  │  Ingestion  │───▶│   Retrieval   │───▶│   Generation     │
  │  (chunking, │    │ (hybrid RRF:  │    │  (LangGraph DAG: │
  │  embedding, │    │  dense+BM25)  │    │  prompt→LLM      │
  │  vector DB) │    │               │    │  →parse→cite)    │
  └─────────────┘    └───────────────┘    └──────────────────┘
                                                    │
                                           ┌────────▼────────┐
                                           │   Evaluation    │
                                           │   (RAGAS,       │
                                           │    optional)    │
                                           └─────────────────┘

Fix #7 (do_expand): ask() accepts and forwards do_expand into generate()
so the retrieval graph's expand_query node can be disabled per-request.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.generation import generate, PersonaConfig, LLMRegistry
from src.generation.prompt_builder import PromptBuilder, sanitize_query

logger = logging.getLogger(__name__)

DEFAULT_VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "data/vectorstores/default")

# ── Optional integrations (graceful degradation if not installed) ─────────────────

def _try_import_ingestion():
    try:
        from src.ingestion import IngestionPipeline
        return IngestionPipeline
    except ImportError:
        logger.warning("[MiniNotebookLM] src.ingestion not found — ingest() disabled.")
        return None

def _try_import_retrieval():
    try:
        from src.retrieval import HybridRetriever
        return HybridRetriever
    except ImportError:
        logger.warning("[MiniNotebookLM] src.retrieval not found — retrieve() returns [].")
        return None

def _try_import_evaluator():
    try:
        from src.evaluation import RagasEvaluator
        return RagasEvaluator
    except ImportError:
        logger.warning("[MiniNotebookLM] src.evaluation not found — RAGAS disabled.")
        return None


# ── Config dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    All pipeline knobs in one place.
    Override via env vars or pass explicitly to MiniNotebookLM().
    """
    # LLM
    llm_provider:    str   = field(default_factory=lambda: os.getenv("LLM_PROVIDER",    "groq"))
    llm_model:       str   = field(default_factory=lambda: os.getenv("LLM_MODEL",       "llama-3.1-70b-versatile"))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))
    llm_max_tokens:  int   = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS",    "1024")))

    # Retrieval
    vectorstore_path:   str  = field(default_factory=lambda: os.getenv("VECTORSTORE_PATH", "data/vectorstores/default"))
    retrieval_k:        int  = field(default_factory=lambda: int(os.getenv("RETRIEVAL_K",    "6")))
    retrieval_rewrite:  bool = field(default_factory=lambda: os.getenv("RETRIEVAL_REWRITE", "true").lower() == "true")
    retrieval_strategy: str  = field(default_factory=lambda: os.getenv("RETRIEVAL_STRATEGY", "auto"))

    # Generation
    default_mode:    str  = field(default_factory=lambda: os.getenv("GEN_MODE", "chat"))
    stream:          bool = field(default_factory=lambda: os.getenv("GEN_STREAM", "false").lower() == "true")

    # Evaluation
    auto_evaluate:   bool = field(default_factory=lambda: os.getenv("AUTO_EVALUATE", "false").lower() == "true")

    # History
    max_history_turns: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY_TURNS", "8")))


# ── Result dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    answer:           str
    citations:        List[Dict[str, Any]]       = field(default_factory=list)
    follow_ups:       List[str]                  = field(default_factory=list)
    sources_used:     List[str]                  = field(default_factory=list)
    chunks_used:      List[Dict[str, Any]]        = field(default_factory=list)
    tokens_estimate:  int                        = 0
    ragas:            Optional[Dict[str, Any]]   = None
    retrieval_query:  str                        = ""
    mode:             str                        = "chat"
    error:            Optional[str]              = None

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


# ── Main pipeline class ─────────────────────────────────────────────────────────────────

class MiniNotebookLM:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        self._ingestion_cls  = _try_import_ingestion()
        self._retrieval_cls  = _try_import_retrieval()
        self._evaluator_cls  = _try_import_evaluator()

        self._ingestion:  Any = None
        self._retrieval:  Any = None
        self._evaluator:  Any = None

        self._history: List[Dict[str, str]] = []
        self.last_ragas: Optional[Dict[str, Any]] = None

        logger.info(
            "[MiniNotebookLM] Init — provider=%s model=%s mode=%s",
            self.config.llm_provider, self.config.llm_model, self.config.default_mode,
        )

    # ── Ingestion ──────────────────────────────────────────────────────────────────

    def ingest(
        self,
        source: str,
        *,
        source_id:     Optional[str] = None,
        chunk_size:    int           = 512,
        chunk_overlap: int           = 64,
        **kwargs,
    ) -> Dict[str, Any]:
        if not self._ingestion_cls:
            raise RuntimeError("src.ingestion is not installed.")

        if self._ingestion is None:
            self._ingestion = self._ingestion_cls()

        result = self._ingestion.ingest(
            source,
            source_id=source_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
        logger.info("[ingest] %s → %s chunks", source, result.get("chunks_added", "?"))

        vpath = result.get("vectorstore_path") or result.get("vector_store_path")
        if vpath:
            self.config.vectorstore_path = vpath
            self._retrieval = None

        return result

    def ingest_many(self, sources: List[str], **kwargs) -> List[Dict[str, Any]]:
        return [self.ingest(s, **kwargs) for s in sources]

    # ── Retrieval ──────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:      str,
        k:          Optional[int]       = None,
        rewrite:    Optional[bool]      = None,
        strategy:   Optional[str]       = None,
        source_ids: Optional[List[str]] = None,
    ) -> tuple[List[Document], str]:
        k       = k       if k       is not None else self.config.retrieval_k
        rewrite = rewrite if rewrite is not None else self.config.retrieval_rewrite

        if rewrite:
            try:
                ret_query = PromptBuilder.get_retrieval_query(query, rewrite=True)
            except Exception:
                ret_query = query
        else:
            ret_query = query

        if not self._retrieval_cls:
            logger.warning("[retrieve] No retriever — returning empty docs.")
            return [], ret_query

        if self._retrieval is None:
            self._retrieval = self._retrieval_cls(
                vectorstore_path=self.config.vectorstore_path,
                top_k=k,
            )

        docs = self._retrieval.retrieve(ret_query, top_k=k, source_ids=source_ids or None)
        logger.info(
            "[retrieve] query_len=%d source_ids=%s → %d docs",
            len(query), source_ids, len(docs),
        )
        return docs, ret_query

    # ── Generation ──────────────────────────────────────────────────────────────────

    def ask(
        self,
        query:         str,
        *,
        mode:          Optional[str]           = None,
        persona:       Optional[PersonaConfig] = None,
        documents:     Optional[List[Document]] = None,
        k:             Optional[int]            = None,
        rewrite:       Optional[bool]           = None,
        evaluate:      Optional[bool]           = None,
        ground_truth:  Optional[str]            = None,
        stream:        Optional[bool]           = None,
        clear_history: bool                    = False,
        source_ids:    Optional[List[str]]      = None,
        # FIX #7: do_expand forwarded all the way from API -> master_pipeline -> generate()
        do_expand:     bool                    = True,
    ) -> GenerationResult:
        mode     = mode     or self.config.default_mode
        evaluate = evaluate if evaluate is not None else self.config.auto_evaluate
        stream   = stream   if stream   is not None else self.config.stream

        if clear_history:
            self._history.clear()

        safe_query = sanitize_query(query)

        # ── 1. Retrieve ───────────────────────────────────────────────────────────────
        ret_query = safe_query
        if documents is None:
            _sids = source_ids if source_ids else None
            documents, ret_query = self.retrieve(
                safe_query, k=k, rewrite=rewrite, source_ids=_sids
            )

        # ── 2. Build conversation history string ───────────────────────────────────
        history_str = self._format_history()

        # ── 3. Generate via LangGraph ───────────────────────────────────────────────
        try:
            gen_result = generate(
                query=safe_query,
                documents=documents,
                mode=mode,
                history=history_str,
                persona=persona or PersonaConfig(),
                stream=stream,
                do_expand=do_expand,  # FIX #7
            )
        except Exception as exc:
            logger.exception("[ask] Generation failed: %s", exc)
            return GenerationResult(
                answer=f"⚠️ Generation failed: {exc}",
                error=str(exc),
                mode=mode,
                retrieval_query=ret_query,
            )

        answer = gen_result.get("answer", "")

        # ── 4. Update history ─────────────────────────────────────────────────────────
        self._history.append({"role": "user",      "content": safe_query})
        self._history.append({"role": "assistant", "content": answer})
        self._trim_history()

        # ── 5. RAGAS evaluation (optional) ────────────────────────────────────────────
        ragas_result: Optional[Dict[str, Any]] = None
        if evaluate and self._evaluator_cls:
            if self._evaluator is None:
                self._evaluator = self._evaluator_cls()
            try:
                ragas_result = self._evaluator.evaluate(
                    query=safe_query,
                    answer=answer,
                    contexts=[
                        (d.page_content if hasattr(d, "page_content") else d.get("content", ""))
                        for d in documents
                    ],
                    ground_truth=ground_truth,
                )
                self.last_ragas = ragas_result
            except Exception as exc:
                logger.warning("[ask] RAGAS evaluation failed: %s", exc)

        return GenerationResult(
            answer=answer,
            citations=gen_result.get("citations",       []),
            follow_ups=gen_result.get("follow_ups",     []),
            sources_used=gen_result.get("sources_used", []),
            chunks_used=gen_result.get("chunks_used",   []),
            tokens_estimate=gen_result.get("tokens_estimate", 0),
            ragas=ragas_result,
            retrieval_query=ret_query,
            mode=mode,
        )

    # ── Convenience aliases ────────────────────────────────────────────────────────────────

    def chat(self, query: str, **kwargs) -> GenerationResult:
        return self.ask(query, mode="chat", **kwargs)

    def study(self, query: str, **kwargs) -> GenerationResult:
        return self.ask(query, mode="study", **kwargs)

    def research(self, query: str, **kwargs) -> GenerationResult:
        return self.ask(query, mode="research", **kwargs)

    # ── History management ──────────────────────────────────────────────────────────────

    def _format_history(self) -> str:
        lines = []
        for turn in self._history[-(self.config.max_history_turns * 2):]:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        return "\n\n".join(lines)

    def _trim_history(self) -> None:
        max_messages = self.config.max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def clear_history(self) -> None:
        self._history.clear()

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)

    # ── State inspection ─────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        llm_ok = False
        try:
            LLMRegistry.get(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
            )
            llm_ok = True
        except Exception:
            pass

        return {
            "version":          "0.4.3",
            "llm_provider":     self.config.llm_provider,
            "llm_model":        self.config.llm_model,
            "llm_ok":           llm_ok,
            "vectorstore_path": self.config.vectorstore_path,
            "ingestion_ready":  self._ingestion  is not None,
            "retrieval_ready":  self._retrieval  is not None,
            "evaluator_ready":  self._evaluator  is not None,
            "history_turns":    len(self._history) // 2,
            "auto_evaluate":    self.config.auto_evaluate,
            "default_mode":     self.config.default_mode,
        }

    def __repr__(self) -> str:
        return (
            f"MiniNotebookLM("
            f"provider={self.config.llm_provider!r}, "
            f"model={self.config.llm_model!r}, "
            f"mode={self.config.default_mode!r}, "
            f"history_turns={len(self._history)//2}"
            f")"
        )
