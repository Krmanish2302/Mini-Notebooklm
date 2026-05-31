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
    llm_model:       str   = field(default_factory=lambda: os.getenv("LLM_MODEL",       "llama-3.3-70b-versatile"))
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
    chunks_used:      List[Dict[str, Any]]       = field(default_factory=list)
    tokens_estimate:  int                        = 0
    ragas:            Optional[Dict[str, Any]]   = None
    retrieval_query:  str                        = ""
    mode:             str                        = "chat"
    error:            Optional[str]              = None
    sub_queries:      List[str]                  = field(default_factory=list)
    quiz_cards:       List[Dict[str, Any]]       = field(default_factory=list)
    summary_bullets:  List[str]                  = field(default_factory=list)
    learning_path:    List[Dict[str, Any]]       = field(default_factory=list)

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

        # Initialise SQLite-based RAGHistoryStore
        try:
            from src.storage.rag_history_store import RAGHistoryStore
            from src.storage.sqlite_manager import SQLiteManager
            from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
            db = SQLiteManager()
            embedder = EmbeddingRegistry.get()
            self.history_store = RAGHistoryStore(db, embedder)
        except Exception as e:
            logger.warning("[MiniNotebookLM] Failed to initialize RAGHistoryStore: %s", e)
            self.history_store = None

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
            # Invalidate cached retriever for this source so next query loads fresh data
            if hasattr(self, "_retriever_cache"):
                sid = result.get("source_id") or os.path.basename(vpath)
                self._retriever_cache.pop(sid, None)

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
        mode:       Optional[str]       = None,
    ) -> tuple[List[Document], str] | tuple[List[Document], str, Optional[dict]]:
        k       = k       if k       is not None else self.config.retrieval_k
        rewrite = rewrite if rewrite is not None else self.config.retrieval_rewrite

        # If mode is chat, bypass rewriting
        if mode == "chat":
            rewrite = False

        if rewrite:
            try:
                ret_query = PromptBuilder.get_retrieval_query(query, rewrite=True)
            except Exception:
                ret_query = query
        else:
            ret_query = query

        if not self._retrieval_cls:
            logger.warning("[retrieve] No retriever — returning empty docs.")
            return [], ret_query, None

        # Determine target source IDs to query
        target_sids = source_ids
        if not target_sids:
            try:
                from src.storage.sqlite_manager import SQLiteManager
                db = SQLiteManager()
                target_sids = [s["source_id"] for s in db.list_sources(active_only=True)]
            except Exception:
                try:
                    target_sids = [d for d in os.listdir("data/vectorstores") if os.path.isdir(os.path.join("data/vectorstores", d))]
                except Exception:
                    target_sids = []

        if not target_sids:
            logger.warning("[retrieve] No active sources found to query.")
            return [], ret_query, None

        # Resolve source names from SQLite
        source_names = {}
        try:
            from src.storage.sqlite_manager import SQLiteManager
            db = SQLiteManager()
            for s in db.list_sources():
                source_names[s["source_id"]] = s["name"]
        except Exception as e:
            logger.warning("[retrieve] Failed to load source names from SQLite: %s", e)

        # Lazily initialise retriever cache
        if not hasattr(self, "_retriever_cache"):
            self._retriever_cache: Dict[str, Any] = {}

        if mode == "chat":
            # Semantic query caching check
            cache_hit = None
            if self.history_store is not None:
                try:
                    cache_hit = self.history_store.check_semantic_cache("default", query, threshold=0.90)
                except Exception as e:
                    logger.warning("[retrieve] Semantic cache check failed: %s", e)
            if cache_hit is not None:
                return [], ret_query, cache_hit

            # Run parallel Dense similarity search + Sparse BM25 search + History turn retrieval
            from concurrent.futures import ThreadPoolExecutor
            dense_futures = {}
            sparse_futures = {}
            history_future = None

            with ThreadPoolExecutor(max_workers=8) as executor:
                if self.history_store is not None:
                    history_future = executor.submit(
                        self.history_store.retrieve_history_docs,
                        session_id="default",
                        current_query=query,
                        top_k=2
                    )

                for sid in target_sids:
                    sid_path = os.path.join("data/vectorstores", sid)
                    if os.path.exists(sid_path):
                        try:
                            if sid not in self._retriever_cache:
                                self._retriever_cache[sid] = self._retrieval_cls(
                                    vectorstore_path=sid_path, top_k=k
                                )
                            r = self._retriever_cache[sid]
                            if r._ensemble is None:
                                r._ensemble = r._build(k)
                            
                            if getattr(r, "dense_retriever", None) is not None:
                                dense_futures[sid] = executor.submit(r.dense_retriever.invoke, query)
                            if getattr(r, "bm25_retriever", None) is not None:
                                sparse_futures[sid] = executor.submit(r.bm25_retriever.invoke, query)
                        except Exception as e:
                            logger.warning("[retrieve] Failed to submit parallel searches for '%s': %s", sid, e)

            # Collect history chunks (containing only assistant responses)
            history_docs = []
            if history_future is not None:
                try:
                    history_docs = history_future.result()
                except Exception as e:
                    logger.warning("[retrieve] Parallel history retrieval failed: %s", e)

            # Collect dense and sparse documents
            all_dense = []
            for sid, fut in dense_futures.items():
                try:
                    docs = fut.result()
                    for doc in docs:
                        if "source_id" not in doc.metadata:
                            doc.metadata["source_id"] = sid
                        doc.metadata["source_name"] = source_names.get(sid, doc.metadata.get("source_name", sid))
                    all_dense.extend(docs)
                except Exception as e:
                    logger.warning("[retrieve] Parallel dense search failed for '%s': %s", sid, e)

            all_sparse = []
            for sid, fut in sparse_futures.items():
                try:
                    docs = fut.result()
                    for doc in docs:
                        if "source_id" not in doc.metadata:
                            doc.metadata["source_id"] = sid
                        doc.metadata["source_name"] = source_names.get(sid, doc.metadata.get("source_name", sid))
                    all_sparse.extend(docs)
                except Exception as e:
                    logger.warning("[retrieve] Parallel sparse search failed for '%s': %s", sid, e)

            # Extract distinct rank lists for RRF
            def get_cid(d):
                return d.metadata.get("chunk_id") or d.metadata.get("id") or str(hash(d.page_content))

            dense_ids, seen_dense = [], set()
            for doc in all_dense:
                cid = get_cid(doc)
                if cid not in seen_dense:
                    seen_dense.add(cid)
                    dense_ids.append(cid)

            sparse_ids, seen_sparse = [], set()
            for doc in all_sparse:
                cid = get_cid(doc)
                if cid not in seen_sparse:
                    seen_sparse.add(cid)
                    sparse_ids.append(cid)

            # RRF Fusion (dense and sparse only)
            RRF_K = 60
            scores = {}
            for rank, cid in enumerate(dense_ids):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            for rank, cid in enumerate(sparse_ids):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

            fused_cids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
            doc_map = {get_cid(d): d for d in all_dense + all_sparse}
            fused_docs = [doc_map[cid] for cid in fused_cids if cid in doc_map]

            # Reranking on fused documents
            try:
                from src.retrieval.reranker import Reranker
                reranker = Reranker()
                reranked_docs = reranker.rerank(query, fused_docs, top_n=k)
            except Exception as e:
                logger.warning("[retrieve] Parallel reranking failed: %s", e)
                reranked_docs = fused_docs[:k]

            # Combine reranked documents and history chunks (history does not participate in RRF or reranking)
            combined_docs = reranked_docs + history_docs

            # Contextual compression on the combined documents
            try:
                from src.retrieval.contextual_compressor import ContextualCompressor
                compressor = ContextualCompressor()
                compressed_docs = compressor.compress(query, combined_docs)
            except Exception as e:
                logger.warning("[retrieve] Parallel contextual compression failed: %s", e)
                compressed_docs = combined_docs

            logger.info(
                "[retrieve] Chat mode parallel retrieval: dense=%d sparse=%d history=%d fused=%d reranked=%d compressed=%d",
                len(all_dense), len(all_sparse), len(history_docs), len(fused_docs), len(reranked_docs), len(compressed_docs)
            )
            return compressed_docs, ret_query, None

        # Retrieve documents from each target source (reuse cached retrievers) - Non-chat mode
        all_docs = []
        for sid in target_sids:
            sid_path = os.path.join("data/vectorstores", sid)
            if os.path.exists(sid_path):
                try:
                    if sid not in self._retriever_cache:
                        self._retriever_cache[sid] = self._retrieval_cls(
                            vectorstore_path=sid_path, top_k=k
                        )
                    r = self._retriever_cache[sid]
                    docs = r.retrieve(ret_query, top_k=k)
                    for doc in docs:
                        if "source_id" not in doc.metadata:
                            doc.metadata["source_id"] = sid
                        doc.metadata["source_name"] = source_names.get(sid, doc.metadata.get("source_name", sid))
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning("[retrieve] Failed to retrieve from source '%s': %s", sid, e)

        # De-duplicate by document content and sort by score descending
        seen = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)

        unique_docs = sorted(unique_docs, key=lambda x: x.metadata.get("score", 0.0), reverse=True)
        
        logger.info(
            "[retrieve] query_len=%d target_sids=%s → %d docs",
            len(query), target_sids, len(unique_docs[:k]),
        )
        return unique_docs[:k], ret_query, None


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
        cache_hit = None
        if documents is None:
            _sids = source_ids if source_ids else None
            ret_res = self.retrieve(
                safe_query, k=k, rewrite=rewrite, source_ids=_sids, mode=mode
            )
            if len(ret_res) == 3:
                documents, ret_query, cache_hit = ret_res
            else:
                documents, ret_query = ret_res
                cache_hit = None

        # Handle Semantic Cache Hit:
        if cache_hit is not None:
            answer = f"[Cached history to similar query (similarity score: {cache_hit['similarity']:.2f})]\n{cache_hit['answer']}"
            return GenerationResult(
                answer=answer,
                retrieval_query=ret_query,
                mode=mode,
            )

        # Handle No Chunks Fallback:
        if mode == "chat" and not documents:
            return GenerationResult(
                answer="Not in my notes, bro.",
                retrieval_query=ret_query,
                mode=mode,
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

        # Save to SQLite-based RAGHistoryStore
        if self.history_store is not None:
            try:
                self.history_store.add_turn("default", safe_query, answer)
            except Exception as e:
                logger.warning("[MiniNotebookLM] Failed to save turn to RAGHistoryStore: %s", e)

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

        # ── 6. Mode-specific study/research materials ─────────────────────────────────
        sub_queries_res = []
        quiz_cards_res = []
        summary_bullets_res = []
        if mode == "study" and documents:
            try:
                from src.retrieval.study_mode import StudyMode
                sm = StudyMode()
                quiz_cards_res = sm.flashcards(documents)
                summary_bullets_res = sm.summary_bullets(documents)
            except Exception as exc:
                logger.warning("[ask] Study mode helper failed: %s", exc)
        elif mode in ("research", "deep_research") and documents:
            try:
                import re
                from src.generation.llm_registry import LLMRegistry
                llm = LLMRegistry.get(temperature=0.5)
                sub_q_prompt = f"Given the user query: '{query}', suggest 3 specific, focused sub-queries to investigate in the context of the document. Return ONLY the 3 queries, one per line, starting with a bullet point (- or *)."
                res = llm.invoke(sub_q_prompt)
                content = res.content or ""
                lines = [re.sub(r'^\s*[-•*\d.)]+\s*', '', line).strip() for line in content.splitlines() if line.strip()]
                sub_queries_res = [l for l in lines if l][:3]
            except Exception as exc:
                logger.warning("[ask] Sub-queries generation failed: %s", exc)

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
            sub_queries=sub_queries_res,
            quiz_cards=quiz_cards_res,
            summary_bullets=summary_bullets_res,
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
