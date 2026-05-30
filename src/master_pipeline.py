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

Design principles
-----------------
1. Stateless generate() — every call is independent; no mutable LLM state.
2. LLM is constructed once via LLMRegistry (lru_cache) — not re-instantiated per call.
3. retrieval_query (HyDE/expanded) is ONLY passed to the retriever — never to the LLM.
4. PersonaConfig is a value object; callers pass it per-call or set a default.
5. RAGAS evaluation is opt-in — set evaluate=True or configure via env.
6. The pipeline never imports Streamlit — UI concerns live in src.ui.
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

# ── Optional integrations (graceful degradation if not installed) ─────────────

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


# ── Config dataclass ──────────────────────────────────────────────────────────

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
    retrieval_k:        int  = field(default_factory=lambda: int(os.getenv("RETRIEVAL_K",    "6")))
    retrieval_rewrite:  bool = field(default_factory=lambda: os.getenv("RETRIEVAL_REWRITE", "true").lower() == "true")
    retrieval_strategy: str  = field(default_factory=lambda: os.getenv("RETRIEVAL_STRATEGY", "auto"))  # auto|hyde|expand|both

    # Generation
    default_mode:    str  = field(default_factory=lambda: os.getenv("GEN_MODE", "chat"))
    stream:          bool = field(default_factory=lambda: os.getenv("GEN_STREAM", "false").lower() == "true")

    # Evaluation
    auto_evaluate:   bool = field(default_factory=lambda: os.getenv("AUTO_EVALUATE", "false").lower() == "true")

    # History
    max_history_turns: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY_TURNS", "8")))


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """
    Structured result returned by MiniNotebookLM.ask().

    Attributes
    ----------
    answer          : Clean answer string with inline [S1]… citations.
    citations       : List of resolved citation dicts.
    follow_ups      : 0-3 suggested follow-up questions.
    sources_used    : Citation labels that appeared in the answer ([S1], [S2], …).
    chunks_used     : Full chunk dicts for cited sources.
    tokens_estimate : Rough token count of the answer.
    ragas           : RAGAS evaluation dict — None if evaluate=False.
    retrieval_query : The rewritten query used for retrieval (NOT shown to LLM).
    mode            : Generation mode used ("chat"|"study"|"research").
    error           : Non-None if the pipeline encountered a recoverable error.
    """
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


# ── Main pipeline class ───────────────────────────────────────────────────────

class MiniNotebookLM:
    """
    One-stop pipeline: ingest → retrieve → generate → (optionally) evaluate.

    Quick start
    -----------
        nb = MiniNotebookLM()
        nb.ingest("notes.pdf")
        result = nb.ask("What is the main idea?")
        print(result.answer)

    With persona
    ------------
        from src.generation import PersonaConfig
        persona = PersonaConfig(persona="analyst", tone="formal", length="long")
        result  = nb.ask("Summarise section 3", persona=persona)

    With RAGAS
    ----------
        result = nb.ask("What is attention?", evaluate=True,
                        ground_truth="Attention is a mechanism that...")
        print(result.ragas)

    With source filtering
    ---------------------
        # Only retrieve from specific ingested sources
        result = nb.ask("What are the conclusions?",
                        source_ids=["lecture_notes", "paper_02"])
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Lazy-loaded sub-systems
        self._ingestion_cls  = _try_import_ingestion()
        self._retrieval_cls  = _try_import_retrieval()
        self._evaluator_cls  = _try_import_evaluator()

        self._ingestion:  Any = None
        self._retrieval:  Any = None
        self._evaluator:  Any = None

        # Conversation history: List[{"role": "user"|"assistant", "content": str}]
        self._history: List[Dict[str, str]] = []

        # Last RAGAS result — accessible by UI without re-running
        self.last_ragas: Optional[Dict[str, Any]] = None

        logger.info(
            "[MiniNotebookLM] Init — provider=%s model=%s mode=%s",
            self.config.llm_provider, self.config.llm_model, self.config.default_mode,
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        source: str,
        *,
        source_id:     Optional[str] = None,
        chunk_size:    int           = 512,
        chunk_overlap: int           = 64,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Ingest a document (file path, URL, or raw text) into the vector store.

        Returns metadata dict: {source_id, chunks_added, source_type, …}
        """
        if not self._ingestion_cls:
            raise RuntimeError("src.ingestion is not installed. Run: pip install -e '.[ingestion]'")

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
        return result

    def ingest_many(self, sources: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Ingest multiple sources. Returns list of per-source metadata."""
        return [self.ingest(s, **kwargs) for s in sources]

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:      str,
        k:          Optional[int]       = None,
        rewrite:    Optional[bool]      = None,
        strategy:   Optional[str]       = None,
        source_ids: Optional[List[str]] = None,
    ) -> tuple[List[Document], str]:
        """
        Retrieve relevant chunks for *query*.

        Parameters
        ----------
        query      : Raw user question.
        k          : Override retrieval_k for this call only.
        rewrite    : Override retrieval_rewrite for this call only.
        strategy   : Override retrieval_strategy (auto|hyde|expand|both).
        source_ids : If provided, only return docs from these source IDs.
                     Passed directly to HybridRetriever.retrieve().

        Returns
        -------
        (documents, retrieval_query)
            documents       — List[Document] ranked by RRF score
            retrieval_query — the rewritten query used for embedding
                              (never passed to LLM)
        """
        k        = k        if k        is not None else self.config.retrieval_k
        rewrite  = rewrite  if rewrite  is not None else self.config.retrieval_rewrite
        strategy = strategy if strategy is not None else self.config.retrieval_strategy

        # Build retrieval query (HyDE / expand / both) — for embedder only
        if rewrite:
            ret_query = PromptBuilder.get_retrieval_query(query, rewrite=True)
        else:
            ret_query = query

        if not self._retrieval_cls:
            logger.warning("[retrieve] No retriever — returning empty docs.")
            return [], ret_query

        if self._retrieval is None:
            self._retrieval = self._retrieval_cls()

        # Pass source_ids to HybridRetriever so both FAISS and BM25 are filtered
        docs = self._retrieval.retrieve(ret_query, top_k=k, source_ids=source_ids or None)
        logger.info(
            "[retrieve] query_len=%d source_ids=%s → %d docs",
            len(query), source_ids, len(docs),
        )
        return docs, ret_query

    # ── Generation ────────────────────────────────────────────────────────────

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
    ) -> GenerationResult:
        """
        Full pipeline: retrieve → generate → (optionally) evaluate.

        Parameters
        ----------
        query         : User question (raw; sanitized internally).
        mode          : "chat" | "study" | "research". Default from config.
        persona       : PersonaConfig — controls tone/style in chat mode.
        documents     : Pre-fetched docs. If None, retrieve() is called.
        k             : Override retrieval_k for this call only.
        rewrite       : Override retrieval_rewrite for this call only.
        evaluate      : Run RAGAS. Default from config.auto_evaluate.
        ground_truth  : Reference answer for RAGAS context_recall + answer_similarity.
        stream        : Stream LLM tokens. Default from config.stream.
        clear_history : Wipe conversation history before this turn.
        source_ids    : Restrict retrieval to these source IDs (UI checkboxes).
                        Empty list / None means use all ingested sources.

        Returns
        -------
        GenerationResult — see dataclass docstring above.
        """
        mode     = mode     or self.config.default_mode
        evaluate = evaluate if evaluate is not None else self.config.auto_evaluate
        stream   = stream   if stream   is not None else self.config.stream

        if clear_history:
            self._history.clear()

        safe_query = sanitize_query(query)

        # ── 1. Retrieve ───────────────────────────────────────────────────────
        ret_query = safe_query
        if documents is None:
            # Treat empty list same as None (search all sources)
            _sids = source_ids if source_ids else None
            documents, ret_query = self.retrieve(
                safe_query, k=k, rewrite=rewrite, source_ids=_sids
            )

        # ── 2. Build conversation history string ──────────────────────────────
        history_str = self._format_history()

        # ── 3. Generate via LangGraph ─────────────────────────────────────────
        try:
            gen_result = generate(
                query=safe_query,          # original query, NOT ret_query
                documents=documents,
                mode=mode,
                history=history_str,
                persona=persona or PersonaConfig(),
                stream=stream,
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

        # ── 4. Update history ─────────────────────────────────────────────────
        self._history.append({"role": "user",      "content": safe_query})
        self._history.append({"role": "assistant", "content": answer})
        self._trim_history()

        # ── 5. RAGAS evaluation (optional) ────────────────────────────────────
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

    # ── Convenience aliases ────────────────────────────────────────────────────

    def chat(self, query: str, **kwargs) -> GenerationResult:
        """Alias for ask() with mode='chat'."""
        return self.ask(query, mode="chat", **kwargs)

    def study(self, query: str, **kwargs) -> GenerationResult:
        """Alias for ask() with mode='study'."""
        return self.ask(query, mode="study", **kwargs)

    def research(self, query: str, **kwargs) -> GenerationResult:
        """Alias for ask() with mode='research'."""
        return self.ask(query, mode="research", **kwargs)

    # ── History management ─────────────────────────────────────────────────────

    def _format_history(self) -> str:
        """Format history list → User:/Assistant: block string for PromptBuilder."""
        lines = []
        for turn in self._history[-(self.config.max_history_turns * 2):]:
            role    = "User"      if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        return "\n\n".join(lines)

    def _trim_history(self) -> None:
        """Keep history within max_history_turns (user+assistant pairs)."""
        max_messages = self.config.max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def clear_history(self) -> None:
        """Wipe conversation history."""
        self._history.clear()

    @property
    def history(self) -> List[Dict[str, str]]:
        """Read-only view of conversation history."""
        return list(self._history)

    # ── State inspection ──────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of pipeline status — useful for health-check endpoints."""
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
            "version":         "0.4.1",
            "llm_provider":    self.config.llm_provider,
            "llm_model":       self.config.llm_model,
            "llm_ok":          llm_ok,
            "ingestion_ready": self._ingestion  is not None,
            "retrieval_ready": self._retrieval  is not None,
            "evaluator_ready": self._evaluator  is not None,
            "history_turns":   len(self._history) // 2,
            "auto_evaluate":   self.config.auto_evaluate,
            "default_mode":    self.config.default_mode,
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
