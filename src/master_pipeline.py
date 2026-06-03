"""
master_pipeline.py — MiniNotebookLM: the single orchestration class.

Wires together Ingestion, Retrieval, and Generation under a unified parallelized interface.
"""
from __future__ import annotations

import logging
import os
import time
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.generation import generate, PersonaConfig, LLMRegistry
from src.generation.prompt_builder import PromptBuilder, sanitize_query
from src.generation.response_parser import ResponseParser
from src.generation.response_generator import ResponseGenerator

from src.storage.sqlite_manager import SQLiteManager
from src.storage.rag_history_store import RAGHistoryStore
from src.storage.knowledge_graph_updater import KnowledgeGraphUpdater

from src.retrieval.retrieval_graph import retrieval_app
from src.retrieval.state import RetrievalState
from src.retrieval.reorder import reorder_chunks
from src.retrieval.nodes.build_context_node import format_study_graph_context

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
    graph_context:    List[Dict[str, Any]]       = field(default_factory=list)

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

        # Reusable ThreadPoolExecutor for background tasks & parallel retrieval
        self._executor = ThreadPoolExecutor(max_workers=16)

        # Single instance level SQLiteManager database connection
        try:
            self._db = SQLiteManager()
        except Exception as e:
            logger.warning("[MiniNotebookLM] Failed to initialize SQLiteManager: %s", e)
            self._db = None

        # Initialise SQLite-based RAGHistoryStore using shared SQLiteManager
        try:
            from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
            embedder = EmbeddingRegistry.get()
            if self._db:
                self.history_store = RAGHistoryStore(self._db, embedder)
            else:
                self.history_store = None
        except Exception as e:
            logger.warning("[MiniNotebookLM] Failed to initialize RAGHistoryStore: %s", e)
            self.history_store = None

        logger.info(
            "[MiniNotebookLM] Init — provider=%s model=%s mode=%s",
            self.config.llm_provider, self.config.llm_model, self.config.default_mode,
        )

    def __del__(self):
        if hasattr(self, "_executor"):
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass

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
        results = [None] * len(sources)
        futures = {self._executor.submit(self.ingest, source, **kwargs): i for i, source in enumerate(sources)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("[ingest_many] Failed to ingest source %s: %s", sources[idx], exc)
                results[idx] = {"error": str(exc), "source": sources[idx]}
        return results

    # ── Retrieval ──────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:      str,
        k:          Optional[int]       = None,
        rewrite:    Optional[bool]      = None,
        strategy:   Optional[str]       = None,
        source_ids: Optional[List[str]] = None,
        mode:       Optional[str]       = None,
        do_expand:  bool                = True,
    ) -> tuple[List[Document], str, Optional[dict], dict]:
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
            return [], ret_query, None, {
                "dense_count": 0, "sparse_count": 0, "history_count": 0,
                "rrf_fused": 0, "reranked": 0, "reordered": 0, "cached_hit": False
            }

        # Determine target source IDs to query (reusing class-level SQLiteManager)
        target_sids = source_ids
        if not target_sids:
            if self._db:
                try:
                    target_sids = [s["source_id"] for s in self._db.list_sources(active_only=True)]
                except Exception as e:
                    logger.warning("[retrieve] Failed to list active sources: %s", e)
                    target_sids = []
            else:
                try:
                    target_sids = [d for d in os.listdir("data/vectorstores") if os.path.isdir(os.path.join("data/vectorstores", d))]
                except Exception:
                    target_sids = []

        if not target_sids:
            logger.warning("[retrieve] No active sources found to query.")
            return [], ret_query, None, {
                "dense_count": 0, "sparse_count": 0, "history_count": 0,
                "rrf_fused": 0, "reranked": 0, "reordered": 0, "cached_hit": False
            }

        # Resolve source names from SQLite (reusing class-level SQLiteManager)
        source_names = {}
        if self._db:
            try:
                for s in self._db.list_sources():
                    source_names[s["source_id"]] = s["name"]
            except Exception as e:
                logger.warning("[retrieve] Failed to load source names from SQLite: %s", e)

        # Lazily initialise retriever cache
        if not hasattr(self, "_retriever_cache"):
            self._retriever_cache: Dict[str, Any] = {}

        # Semantic query caching check (disabled/dropped)
        cache_hit = None

        # Retrieve documents from each target source using LangGraph retrieval_app
        all_docs = []
        all_parents = []
        all_graph_context = []
        all_current_concepts = []
        
        # Accumulate stats
        total_dense = 0
        total_sparse = 0
        total_history = 0
        total_rrf = 0
        total_reranked = 0
        total_reordered = 0

        # Parallelize active sources retrieval invocation
        futures = {}
        for sid in target_sids:
            sid_path = os.path.join("data/vectorstores", sid)
            if os.path.exists(sid_path):
                state = RetrievalState(
                    query=query,
                    vectorstore_path=sid_path,
                    top_k=k,
                    use_rerank=True,
                    use_reordering=True,
                    do_expand=do_expand,
                    mode=mode,
                    source_ids=[sid]
                )
                futures[self._executor.submit(retrieval_app.invoke, state)] = sid

        for future in as_completed(futures):
            sid = futures[future]
            try:
                res = future.result()
                
                # Gather docs from result
                docs = res.get("documents", [])
                for doc in docs:
                    if "source_id" not in doc.metadata:
                        doc.metadata["source_id"] = sid
                    doc.metadata["source_name"] = source_names.get(sid, doc.metadata.get("source_name", sid))
                all_docs.extend(docs)
                
                # Gather history_docs returned by study mode node
                hdocs = res.get("history_docs", [])
                for hd in hdocs:
                    if "source_id" not in hd.metadata:
                        hd.metadata["source_id"] = "history"
                    hd.metadata["is_history"] = True
                all_docs.extend(hdocs)
                
                # Accumulate current concepts
                ccon = res.get("current_concepts", [])
                if ccon:
                    all_current_concepts.extend(ccon)
                
                # Accumulate stats
                meta = res.get("metadata") or {}
                total_dense += meta.get("dense_count", 0)
                total_sparse += meta.get("sparse_count", 0)
                total_history += meta.get("history_count", 0)
                total_rrf += meta.get("rrf_fused", 0)
                total_reranked += meta.get("reranked", 0)
                total_reordered += meta.get("reordered", 0)

                # Gather parents
                parents = res.get("reordered_parents", [])
                all_parents.extend(parents)
                
                # Gather graph context
                gctx = res.get("graph_context", [])
                if gctx:
                    all_graph_context.extend(gctx)
                    
            except Exception as e:
                logger.warning("[retrieve] LangGraph retrieval failed for source '%s': %s", sid, e)

        # De-duplicate unique documents and separate history
        seen = set()
        history_docs = []
        source_docs = []

        for doc in all_docs:
            is_hist = doc.metadata.get("is_history") or doc.metadata.get("source_id") == "history"
            cid = doc.metadata.get("parent_id") or doc.metadata.get("chunk_id") or hash(doc.page_content[:200])
            if cid not in seen:
                seen.add(cid)
                if is_hist:
                    history_docs.append(doc)
                else:
                    source_docs.append(doc)

        if mode == "chat":
            # Chunks are already reranked and reordered inside chat_retrieve node
            combined_docs = history_docs[:2] + source_docs[:k]

            logger.info(
                "[retrieve] Chat mode LangGraph retrieval: dense=%d sparse=%d history=%d fused=%d reranked=%d reordered=%d",
                total_dense, total_sparse, total_history, total_rrf, total_reranked, total_reordered
            )

            retrieval_stats = {
                "dense_count": total_dense,
                "sparse_count": total_sparse,
                "history_count": total_history,
                "rrf_fused": total_rrf,
                "reranked": total_reranked,
                "reordered": total_reordered,
                "cached_hit": False,
            }
            self._last_graph_context = []
            return combined_docs, ret_query, None, retrieval_stats

        # Deduplicate current concepts
        seen_ccon = set()
        unique_ccon = []
        for cc in all_current_concepts:
            cc_name = cc.get("name", "").strip().lower()
            if cc_name and cc_name not in seen_ccon:
                seen_ccon.add(cc_name)
                unique_ccon.append(cc)

        if mode == "study":
            graph_text = format_study_graph_context(all_graph_context, unique_ccon)
            
            reordered_source_docs = source_docs[:k]
            combined_docs = []
            if graph_text:
                graph_doc = Document(
                    page_content=graph_text,
                    metadata={"source_id": "SQLite Study Graph"}
                )
                combined_docs.append(graph_doc)
            
            combined_docs.extend(history_docs[:2])
            combined_docs.extend(reordered_source_docs)
            
            logger.info(
                "[retrieve] Study mode LangGraph retrieval: dense=%d sparse=%d history=%d fused=%d reranked=%d reordered=%d",
                total_dense, total_sparse, total_history, total_rrf, total_reranked, len(reordered_source_docs)
            )
            
            retrieval_stats = {
                "dense_count": total_dense,
                "sparse_count": total_sparse,
                "history_count": len(history_docs),
                "rrf_fused": total_rrf,
                "reranked": total_reranked,
                "reordered": len(reordered_source_docs),
                "cached_hit": False,
            }
            self._last_graph_context = all_graph_context
            self._last_current_concepts = unique_ccon
            return combined_docs, ret_query, None, retrieval_stats

        # Deep Research mode
        all_docs = source_docs[:k]

        logger.info(
            "[retrieve] non-chat mode retrieval: target_sids=%s → %d docs, mode=%s",
            target_sids, len(all_docs), mode
        )

        retrieval_stats = {
            "dense_count": len(all_docs),
            "sparse_count": 0,
            "history_count": 0,
            "rrf_fused": 0,
            "reranked": 0,
            "reordered": len(all_docs),
            "cached_hit": False,
        }
        self._last_graph_context = all_graph_context
        return all_docs, ret_query, None, retrieval_stats


    # ── Helpers for ask & ask_stream logic deduplication ────────────────────────────

    def _run_retrieval(
        self,
        query: str,
        mode: str,
        k: Optional[int],
        rewrite: Optional[bool],
        source_ids: Optional[List[str]],
        documents: Optional[List[Document]],
        do_expand: bool,
    ) -> tuple[List[Document], str, Optional[dict], dict]:
        ret_query = query
        cache_hit = None
        
        if documents is None:
            _sids = source_ids if source_ids else None
            documents, ret_query, cache_hit, retrieval_stats = self.retrieve(
                query, k=k, rewrite=rewrite, source_ids=_sids, mode=mode, do_expand=do_expand
            )
        else:
            retrieval_stats = {
                "dense_count": len(documents),
                "sparse_count": 0,
                "history_count": 0,
                "rrf_fused": 0,
                "reranked": 0,
                "reordered": 0,
                "cached_hit": False,
            }
        return documents, ret_query, cache_hit, retrieval_stats

    def _post_process(
        self,
        query: str,
        answer: str,
        documents: List[Document],
        mode: str,
        evaluate: bool,
        ground_truth: Optional[str],
        sub_queries_future: Optional[Any] = None,
    ) -> tuple[Optional[dict], List[str]]:
        # 1. Safe Knowledge Graph Enrichment in Study Mode using shared SQLiteManager
        if mode == "study" and documents:
            try:
                updater = KnowledgeGraphUpdater(self._db)
                clean_docs = [d for d in documents if d.metadata.get("source_id") != "SQLite Study Graph"]
                updater.enrich_graph(query, answer, clean_docs)
            except Exception as e:
                logger.warning("[MiniNotebookLM] Safe graph enrichment failed: %s", e)

        # 2. Update history
        self._history.append({"role": "user",      "content": query})
        self._history.append({"role": "assistant", "content": answer})
        self._trim_history()

        # Save to SQLite-based RAGHistoryStore
        if self.history_store is not None:
            try:
                self.history_store.add_turn("default", query, answer)
            except Exception as e:
                logger.warning("[MiniNotebookLM] Failed to save turn to RAGHistoryStore: %s", e)

        # 3. RAGAS evaluation (optional)
        ragas_result = None
        if evaluate and self._evaluator_cls:
            if self._evaluator is None:
                self._evaluator = self._evaluator_cls()
            try:
                ragas_result = self._evaluator.evaluate(
                    query=query,
                    answer=answer,
                    contexts=[
                        (d.page_content if hasattr(d, "page_content") else d.get("content", ""))
                        for d in documents
                    ],
                    ground_truth=ground_truth,
                )
                self.last_ragas = ragas_result
            except Exception as exc:
                logger.warning("[_post_process] RAGAS evaluation failed: %s", exc)

        # 4. Resolve background sub-queries if research mode future is supplied
        sub_queries_res = []
        if sub_queries_future:
            try:
                res = sub_queries_future.result(timeout=15)
                content = res.content or ""
                lines = [re.sub(r'^\s*[-•*\d.)]+\s*', '', line).strip() for line in content.splitlines() if line.strip()]
                sub_queries_res = [l for l in lines if l][:3]
            except Exception as exc:
                logger.warning("[_post_process] Sub-queries background generation failed: %s", exc)

        return ragas_result, sub_queries_res


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
        do_expand:     bool                    = True,
    ) -> GenerationResult:
        mode     = mode     or self.config.default_mode
        evaluate = evaluate if evaluate is not None else self.config.auto_evaluate
        stream   = stream   if stream   is not None else self.config.stream

        if clear_history:
            self._history.clear()

        safe_query = sanitize_query(query)

        # ── 1. Retrieve ───────────────────────────────────────────────────────────────
        documents, ret_query, cache_hit, retrieval_stats = self._run_retrieval(
            safe_query, mode, k, rewrite, source_ids, documents, do_expand
        )

        # Handle Semantic Cache Hit:
        if cache_hit is not None:
            answer = f"[Cached history to similar query (similarity score: {cache_hit['similarity']:.2f})]\n{cache_hit['answer']}"
            return GenerationResult(
                answer=answer,
                retrieval_query=ret_query,
                mode=mode,
            )

        # Handle No Chunks Fallback:
        if not documents:
            return GenerationResult(
                answer="Not in my notes, bro.",
                retrieval_query=ret_query,
                mode=mode,
            )

        # ── 2. Build conversation history string ───────────────────────────────────
        history_str = self._format_history()

        # ── 3. Kick off sub-queries concurrently in background thread if research mode ──
        sub_queries_future = None
        if mode in ("research", "deep_research"):
            try:
                llm_sub = LLMRegistry.get(temperature=0.5)
                sub_q_prompt = f"Given the user query: '{query}', suggest 3 specific, focused sub-queries to investigate in the context of the document. Return ONLY the 3 queries, one per line, starting with a bullet point (- or *)."
                sub_queries_future = self._executor.submit(llm_sub.invoke, sub_q_prompt)
            except Exception as exc:
                logger.warning("[ask] Failed to kick off sub-queries in background: %s", exc)

        # ── 4. Generate via LangGraph ───────────────────────────────────────────────
        try:
            gen_result = generate(
                query=safe_query,
                documents=documents,
                mode=mode,
                history=history_str,
                persona=persona or PersonaConfig(),
                stream=stream,
                do_expand=do_expand,
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

        # ── 5. Unified Post-Process & Evaluation ──────────────────────────────────────
        ragas_result, sub_queries_res = self._post_process(
            safe_query, answer, documents, mode, evaluate, ground_truth, sub_queries_future
        )

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
            quiz_cards=[],
            summary_bullets=[],
            graph_context=getattr(self, "_last_graph_context", []),
        )

    def ask_stream(
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
        clear_history: bool                    = False,
        source_ids:    Optional[List[str]]      = None,
        do_expand:     bool                    = True,
    ):
        """
        Stream the generation token by token.
        Yields events matching the event format for SSE streaming.
        """
        mode     = mode     or self.config.default_mode
        evaluate = evaluate if evaluate is not None else self.config.auto_evaluate

        if clear_history:
            self._history.clear()

        safe_query = sanitize_query(query)

        # ── 1. Retrieve ───────────────────────────────────────────────────────────────
        documents, ret_query, cache_hit, retrieval_stats = self._run_retrieval(
            safe_query, mode, k, rewrite, source_ids, documents, do_expand
        )

        # Handle Semantic Cache Hit:
        if cache_hit is not None:
            answer = f"[Cached history to similar query (similarity score: {cache_hit['similarity']:.2f})]\n{cache_hit['answer']}"
            
            # Yield cached answer token by token
            for word in answer.split(" "):
                if word:
                    yield {"type": "token", "content": word + " "}
            
            meta = {
                "type": "metadata",
                "citations": [],
                "sources_used": [],
                "follow_ups": [],
                "sub_queries": [],
                "quiz_cards": [],
                "summary_bullets": [],
                "learning_path": [],
                "pipeline_mode": mode,
                "ttft_ms": 0,
                "total_time_ms": 0,
                "retrieval_stats": retrieval_stats,
                "chunk_strategy": "cache",
                "model_name": self.config.llm_model,
            }
            yield meta
            yield {"type": "done"}
            return

        # Handle No Chunks Fallback:
        if not documents:
            answer = "Not in my notes, bro."
            for word in answer.split(" "):
                if word:
                    yield {"type": "token", "content": word + " "}
            meta = {
                "type": "metadata",
                "citations": [],
                "sources_used": [],
                "follow_ups": [],
                "sub_queries": [],
                "quiz_cards": [],
                "summary_bullets": [],
                "learning_path": [],
                "pipeline_mode": mode,
                "ttft_ms": 0,
                "total_time_ms": 0,
                "retrieval_stats": retrieval_stats,
                "chunk_strategy": "none",
                "model_name": self.config.llm_model,
            }
            yield meta
            yield {"type": "done"}
            return

        # ── 2. Build prompt ─────────────────────────────────────────────────────────
        history_str = self._format_history()
        if mode == "study":
            prompt = PromptBuilder.build_study_prompt(safe_query, documents, history_str)
        elif mode == "research":
            prompt = PromptBuilder.build_research_prompt(safe_query, documents, history_str)
        else:
            prompt = PromptBuilder.build_chat_prompt(safe_query, documents, history_str, persona or PersonaConfig())

        # ── 3. Kick off sub-queries concurrently in background thread if research mode ──
        sub_queries_future = None
        if mode in ("research", "deep_research"):
            try:
                llm_sub = LLMRegistry.get(temperature=0.5)
                sub_q_prompt = f"Given the user query: '{query}', suggest 3 specific, focused sub-queries to investigate in the context of the document. Return ONLY the 3 queries, one per line, starting with a bullet point (- or *)."
                sub_queries_future = self._executor.submit(llm_sub.invoke, sub_q_prompt)
            except Exception as exc:
                logger.warning("[ask_stream] Failed to kick off sub-queries in background: %s", exc)

        # ── 4. Call LLM with streaming ──────────────────────────────────────────────
        start_time = time.time()
        ttft_ms = 0
        tokens = []

        try:
            llm = LLMRegistry.get(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )

            for chunk in llm.stream([HumanMessage(content=prompt)]):
                content = chunk.content or ""
                if content:
                    if ttft_ms == 0:
                        ttft_ms = int((time.time() - start_time) * 1000)
                    tokens.append(content)
                    yield {"type": "token", "content": content}

            total_time_ms = int((time.time() - start_time) * 1000)
            raw_output = "".join(tokens)

            # ── 5. Parse & Assemble Response ──────────────────────────────────────────
            # Extract/parse followups
            parsed = ResponseParser.parse(raw_output)

            # Build chunk details for ResponseGenerator
            chunks = []
            for i, doc in enumerate(documents, 1):
                if hasattr(doc, "page_content"):
                    content = doc.page_content
                    meta    = doc.metadata or {}
                else:
                    content = doc.get("content", "")
                    meta    = {k: v for k, v in doc.items() if k != "content"}
                chunks.append({"citation_label": f"S{i}", "content": content, **meta})

            generator = ResponseGenerator(chunks)
            assembled = generator.assemble(
                raw_llm_output=raw_output,
                query=safe_query,
            )

            # Get final values
            answer = assembled.get("answer", raw_output)
            citations = assembled.get("citations", [])
            follow_ups = parsed.follow_ups or assembled.get("follow_ups", [])
            sources_used = assembled.get("sources_used", [])
            chunks_used = assembled.get("chunks_used", [])

            # ── 6. Unified Post-Process & Evaluation ──────────────────────────────────
            ragas_result, sub_queries_res = self._post_process(
                safe_query, answer, documents, mode, evaluate, ground_truth, sub_queries_future
            )

            # ── 7. Yield Metadata ───────────────────────────────────────────────────
            chunk_strategy = "unknown"
            if documents:
                chunk_strategy = documents[0].metadata.get("chunking_strategy", "unknown")

            meta = {
                "type": "metadata",
                "citations": citations,
                "sources_used": sources_used,
                "follow_ups": follow_ups,
                "sub_queries": sub_queries_res,
                "quiz_cards": [],
                "summary_bullets": [],
                "learning_path": [],
                "pipeline_mode": mode,
                "ttft_ms": ttft_ms,
                "total_time_ms": total_time_ms,
                "retrieval_stats": retrieval_stats,
                "chunk_strategy": chunk_strategy,
                "model_name": self.config.llm_model,
                "chunks_used": chunks_used,
                "graph_context": getattr(self, "_last_graph_context", []),
                "ragas": ragas_result,
            }
            yield meta
            yield {"type": "done"}

        except Exception as exc:
            logger.exception("[ask_stream] Streaming generation failed: %s", exc)
            yield {"type": "error", "detail": str(exc)}

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
