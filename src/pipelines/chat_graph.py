"""
chat_graph.py — LangGraph ChatGraph

Graph topology:
    embed_query → retrieve → compress → rerank → build_prompt → generate → END

Fixes applied:
  FIX-P01: source_ids filter — retrieve node now filters FAISS results to
            chunks whose metadata["source_id"] is in the allowed set.
            ChatState gains a `source_ids` field (empty list = search all).
  FIX-P02: ChatGraph.chat() accepts + forwards source_ids.

History: RAG-based (no MemorySaver / ConversationBufferWindowMemory).
  - Past turns are stored + embedded in SQLite via RAGHistoryStore.
  - On each query the N most-relevant turns are retrieved by cosine
    similarity and injected into the prompt as a plain-text history block.
  - The ChatState carries `rag_history` (str) built by the caller before
    invoking the graph.

LLM is supplied via ChatState["llm"] so any LangChain BaseChatModel works.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
from src.storage.faiss_store import MultiFAISSStore
from src.storage.sqlite_manager import SQLiteManager
from src.storage.source_manager import SourceManager
from src.storage.rag_history_store import RAGHistoryStore

logger = logging.getLogger(__name__)

RRF_K = 60

# ── Persona system prompts ──────────────────────────────────────────────────

_SYSTEM_CHAT = (
    "You're Carl Sagan if he were a chill classmate. "
    "Simple words, real-world analogies, poetic touch. "
    "Answer ONLY from the SOURCES in the context. "
    "Cite as [S1], [S2]\u2026 "
    "If it's not there, say: 'Not in my notes, bro.'"
)

_SYSTEM_DEEP = (
    "You're Carl Sagan if he were a chill classmate. "
    "Go deep \u2014 structured, thorough, cite everything [S1]\u2026 "
    "Use ONLY the sources. Never invent facts. "
    "If it's not there, say: 'Not in my notes, bro.'"
)


# ── State ───────────────────────────────────────────────────────────────────

class ChatState(TypedDict, total=False):
    query:               str
    rag_history:         str                  # pre-built by caller via RAGHistoryStore
    source_ids:          List[str]            # FIX-P01: empty = search all sources
    retrieved:           List[Dict[str, Any]]
    history_chunks:      List[Dict[str, Any]] # history chunks (assistant response only)
    cached_response:     str                  # semantic query cache hit
    session_id:          str
    _query_vectors:      Dict[int, Any]       # internal
    _formatted_messages: Any                  # internal (List[BaseMessage])
    response:            str
    mode:                str                  # "chat" | "deep_research"
    llm:                 Any                  # BaseChatModel
    top_k:               int
    score_threshold:     float
    error:               Optional[str]



# ── RRF helper ─────────────────────────────────────────────────────────────────

def _rrf_fuse(results_by_dim: Dict[int, List[tuple]], k: int = RRF_K) -> List[str]:
    scores: Dict[str, float] = {}
    for dim_results in results_by_dim.values():
        for rank, (chunk_id, _score) in enumerate(dim_results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


# ── Node factories ─────────────────────────────────────────────────────────────────

def _make_embed_query(faiss_store: MultiFAISSStore):
    def node(state: ChatState) -> ChatState:
        try:
            active_dims = faiss_store.active_dims()
            if not active_dims:
                return {**state, "retrieved": [], "_query_vectors": {}}
            qvecs: Dict[int, Any] = {}
            for dim in active_dims:
                model = EmbeddingRegistry.get_by_dim(dim)
                if model:
                    qvecs[dim] = model.embed_query(state["query"])
            return {**state, "_query_vectors": qvecs}
        except Exception as exc:
            logger.error("[ChatGraph:embed_query] %s", exc)
            return {**state, "error": str(exc)}
    return node


def _make_retrieve(faiss_store: MultiFAISSStore, sqlite: SQLiteManager, rag_history_store: Optional[RAGHistoryStore] = None):
    def node(state: ChatState) -> ChatState:
        if state.get("error"):
            return state
        try:
            query        = state["query"]
            top_k        = state.get("top_k", 8)
            candidate_k  = top_k * 3
            allowed_ids: set = set(state.get("source_ids") or [])

            if state.get("mode") == "chat":
                # Parallel Dense + Sparse + History chunk retrieval & Semantic query caching check
                from concurrent.futures import ThreadPoolExecutor
                import concurrent.futures

                cache_hit = None
                dense_raw = {}
                sparse_docs = []
                history_docs = []

                # Resolve embedding vectors first for dense search
                qvecs = {}
                try:
                    active_dims = faiss_store.active_dims()
                    for dim in active_dims:
                        model = EmbeddingRegistry.get_by_dim(dim)
                        if model:
                            qvecs[dim] = model.embed_query(query)
                except Exception as exc:
                    logger.warning("[ChatGraph:retrieve] Embedding query failed: %s", exc)

                # Query database if empty list of sources to get all active source IDs
                if not allowed_ids:
                    try:
                        allowed_ids = {s["source_id"] for s in sqlite.list_sources(active_only=True)}
                    except Exception:
                        allowed_ids = set()

                session_id = state.get("session_id", "default")

                with ThreadPoolExecutor(max_workers=4) as executor:
                    # 1. Cache hit check
                    cache_fut = None
                    if rag_history_store is not None:
                        cache_fut = executor.submit(rag_history_store.check_semantic_cache, session_id, query, 0.90)

                    # 2. Dense search
                    dense_fut = None
                    if qvecs:
                        dense_fut = executor.submit(faiss_store.search, qvecs, k=candidate_k)

                    # 3. Sparse search (BM25 over documents in allowed sources)
                    sparse_fut = None
                    all_docs = []
                    for sid in allowed_ids:
                        try:
                            all_docs.extend(sqlite.get_documents_by_source(sid))
                        except Exception:
                            pass
                    if all_docs:
                        from langchain_community.retrievers import BM25Retriever
                        try:
                            bm25 = BM25Retriever.from_documents(all_docs, k=candidate_k)
                            sparse_fut = executor.submit(bm25.invoke, query)
                        except Exception as e:
                            logger.warning("[ChatGraph:retrieve] BM25 init failed: %s", e)

                    # 4. History retrieval (max 2-3 turns, matched against response_embedding)
                    history_fut = None
                    if rag_history_store is not None:
                        history_fut = executor.submit(rag_history_store.retrieve_history_docs, session_id, query, 2)

                    # Retrieve futures
                    if cache_fut is not None:
                        try:
                            cache_hit = cache_fut.result()
                        except Exception as e:
                            logger.warning("[ChatGraph:retrieve] Cache check future failed: %s", e)

                    if cache_hit is not None:
                        # Semantic cache hit -> stop retrieval and set cached_response
                        cached_answer = f"[Cached history to similar query (similarity score: {cache_hit['similarity']:.2f})]\n{cache_hit['answer']}"
                        return {
                            **state,
                            "cached_response": cached_answer,
                            "retrieved": [],
                            "history_chunks": [],
                        }

                    if dense_fut is not None:
                        try:
                            dense_raw = dense_fut.result()
                        except Exception as e:
                            logger.warning("[ChatGraph:retrieve] Dense search future failed: %s", e)

                    if sparse_fut is not None:
                        try:
                            sparse_docs = sparse_fut.result()
                        except Exception as e:
                            logger.warning("[ChatGraph:retrieve] Sparse search future failed: %s", e)

                    if history_fut is not None:
                        try:
                            history_docs = history_fut.result()
                        except Exception as e:
                            logger.warning("[ChatGraph:retrieve] History future failed: %s", e)

                # RRF over Dense search results (multi-dimensional)
                dense_ranked_ids = _rrf_fuse(dense_raw)
                
                # filter dense results to active sources only
                dense_ids = []
                for cid in dense_ranked_ids:
                    chunk_source = "_".join(cid.rsplit("_", 1)[:-1])
                    if allowed_ids and chunk_source not in allowed_ids:
                        continue
                    dense_ids.append(cid)

                # Sparse results
                sparse_ids = []
                for doc in sparse_docs:
                    cid = doc.metadata.get("chunk_id") or doc.metadata.get("id") or str(hash(doc.page_content))
                    chunk_source = "_".join(cid.rsplit("_", 1)[:-1])
                    if allowed_ids and chunk_source not in allowed_ids:
                        continue
                    sparse_ids.append(cid)

                # RRF Fusion of Dense & Sparse search results
                RRF_K = 60
                scores = {}
                for rank, cid in enumerate(dense_ids):
                    scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
                for rank, cid in enumerate(sparse_ids):
                    scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

                fused_cids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:candidate_k]

                # Resolve fused chunk details from SQLite
                chunks = []
                for cid in fused_cids:
                    chunk_info = sqlite.get_chunk_with_source(cid)
                    if chunk_info:
                        chunks.append({
                            "id":          cid,
                            "content":     chunk_info["content"],
                            "source_id":   chunk_info["source_id"],
                            "source_name": chunk_info["source_name"],
                            "page":        chunk_info["metadata"].get("page_number", chunk_info["metadata"].get("page", "")),
                        })

                # Convert history_docs list of Documents to matching dictionaries
                h_chunks = []
                for doc in history_docs:
                    h_chunks.append({
                        "id":          doc.metadata.get("chunk_id", "history"),
                        "content":     doc.page_content,
                        "source_id":   "history",
                        "source_name": "Chat History",
                        "page":        "",
                    })

                logger.debug(
                    "[ChatGraph:retrieve] Parallel retrieval completed: dense=%d sparse=%d history=%d fused=%d",
                    len(dense_ids), len(sparse_ids), len(h_chunks), len(chunks)
                )
                return {
                    **state,
                    "retrieved": chunks,
                    "history_chunks": h_chunks,
                }

            # Original non-chat retrieval flow:
            qvecs = state.get("_query_vectors", {})
            if not qvecs:
                return {**state, "retrieved": []}

            raw       = faiss_store.search(qvecs, k=candidate_k)
            fused_ids = _rrf_fuse(raw)

            chunks = []
            for cid in fused_ids[:candidate_k]:
                # FIX-P01: parse source_id prefix from chunk_id ("<source_id>_<index>")
                if allowed_ids:
                    chunk_source = "_".join(cid.rsplit("_", 1)[:-1])
                    if chunk_source not in allowed_ids:
                        continue

                chunk_info = sqlite.get_chunk_with_source(cid)
                if chunk_info:
                    chunks.append({
                        "id":          cid,
                        "content":     chunk_info["content"],
                        "source_id":   chunk_info["source_id"],
                        "source_name": chunk_info["source_name"],
                        "page":        chunk_info["metadata"].get("page_number", chunk_info["metadata"].get("page", "")),
                    })

            logger.debug(
                "[ChatGraph:retrieve] fused=%d after_filter=%d (allowed_sources=%s)",
                len(fused_ids), len(chunks),
                list(allowed_ids) if allowed_ids else "all",
            )
            return {**state, "retrieved": chunks}
        except Exception as exc:
            logger.error("[ChatGraph:retrieve] %s", exc)
            return {**state, "error": str(exc)}

    return node


def _make_reorder():
    def node(state: ChatState) -> ChatState:
        if state.get("error"):
            return state
        if state.get("cached_response"):
            return state
        retrieved = state.get("retrieved", [])
        if not retrieved:
            return state
        try:
            from src.retrieval.reorder import reorder_chunks

            # Separate history_chunks and source chunks from retrieved based on source_id == "history"
            history_chunks = [c for c in retrieved if c.get("source_id") == "history"]
            source_chunks = [c for c in retrieved if c.get("source_id") != "history"]

            # Extract rerank_score (or default score) for source chunks and call reorder_chunks on them
            chunks_with_scores = []
            for c in source_chunks:
                score = c.get("rerank_score", c.get("score", 0.0))
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 0.0
                chunks_with_scores.append((c, score))

            reordered_sources = reorder_chunks(chunks_with_scores)

            # Re-assemble retrieved documents by prepending history_chunks before reordered_sources
            combined = history_chunks + reordered_sources

            logger.debug(
                "[ChatGraph:reorder] %d sources reordered, %d history chunks prepended",
                len(reordered_sources), len(history_chunks)
            )
            return {**state, "retrieved": combined}
        except Exception as exc:
            logger.warning("[ChatGraph:reorder] fallback — %s", exc)
            return state
    return node


def _make_rerank(reranker):
    def node(state: ChatState) -> ChatState:
        if state.get("error"):
            return state
        if state.get("cached_response"):
            return state
        chunks    = state.get("retrieved", [])
        top_k     = state.get("top_k", 8)
        threshold = state.get("score_threshold", 0.0)
        
        if not chunks:
            h_chunks = state.get("history_chunks", [])
            return {**state, "retrieved": h_chunks[:top_k]}

        try:
            if reranker is not None:
                reranked = reranker.rerank(state["query"], chunks, top_k=len(chunks))
                filtered = [c for c in reranked if c.get("rerank_score", 1.0) >= threshold]
                logger.debug(
                    "[ChatGraph:rerank] %d → %d (threshold=%.2f)",
                    len(reranked), len(filtered), threshold,
                )
                reranked_docs = filtered
            else:
                reranked_docs = chunks

            # Merge reranked documents with parallel history chunks (history chunks contain assistant responses only)
            h_chunks = state.get("history_chunks", [])
            combined = reranked_docs + h_chunks
            return {**state, "retrieved": combined[:top_k]}
        except Exception as exc:
            logger.warning("[ChatGraph:rerank] fallback — %s", exc)
            h_chunks = state.get("history_chunks", [])
            combined = chunks + h_chunks
            return {**state, "retrieved": combined[:top_k]}
    return node


def _make_build_prompt():
    def node(state: ChatState) -> ChatState:
        if state.get("error"):
            return state
        if state.get("cached_response"):
            return state
        try:
            retrieved_chunks = state.get("retrieved", [])
            source_block = "\n\n".join(
                f"[S{i+1}] {c['content']}"
                for i, c in enumerate(retrieved_chunks)
            ) or "[No sources retrieved]"

            # Bypassed in chat mode because history turns are retrieved and compressed as part of retrieved documents
            rag_history  = "" if state.get("mode") == "chat" else state.get("rag_history", "")
            history_line = f"\n\nCONVERSATION HISTORY:\n{rag_history}" if rag_history else ""

            system = _SYSTEM_DEEP if state.get("mode") == "deep_research" else _SYSTEM_CHAT

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system + "\n\nSOURCES:\n{context}{history}"),
                ("human", "{query}"),
            ])
            messages = prompt_template.format_messages(
                context=source_block,
                history=history_line,
                query=state["query"],
            )
            return {**state, "_formatted_messages": messages}
        except Exception as exc:
            logger.error("[ChatGraph:build_prompt] %s", exc)
            return {**state, "error": str(exc)}
    return node


def _make_generate():
    def node(state: ChatState) -> ChatState:
        if state.get("error"):
            return {
                **state,
                "response": "I encountered an error — please try again.",
            }
        
        # Handle Cache Hit shortcut
        if state.get("cached_response"):
            return {
                **state,
                "response": state["cached_response"]
            }

        # Handle No Chunks Fallback
        if state.get("mode") == "chat" and not state.get("retrieved", []):
            return {
                **state,
                "response": "Not in my notes, bro."
            }

        try:
            llm = state.get("llm")
            if llm is None:
                raise ValueError("No LLM configured in ChatState.")

            formatted_messages = state.get("_formatted_messages")
            if formatted_messages is None:
                raise ValueError("No formatted messages — build_prompt may have failed.")

            ai_msg   = llm.invoke(formatted_messages)
            response = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
            return {**state, "response": response}
        except Exception as exc:
            logger.error("[ChatGraph:generate] %s", exc)
            return {
                **state,
                "error": str(exc),
                "response": "I encountered an error — please try again.",
            }
    return node



# ── Graph builder ─────────────────────────────────────────────────────────────────

def build_chat_graph(
    faiss_store:  MultiFAISSStore,
    sqlite:       SQLiteManager,
    compressor=None,
    reranker=None,
    rag_history_store=None,
) -> Any:
    """Build and compile the chat LangGraph. No checkpointer needed — history is RAG-based."""
    builder = StateGraph(ChatState)
    builder.add_node("embed_query",  _make_embed_query(faiss_store))
    builder.add_node("retrieve",     _make_retrieve(faiss_store, sqlite, rag_history_store))
    builder.add_node("reorder",      _make_reorder())
    builder.add_node("rerank",       _make_rerank(reranker))
    builder.add_node("build_prompt", _make_build_prompt())
    builder.add_node("generate",     _make_generate())

    builder.set_entry_point("embed_query")
    builder.add_edge("embed_query",  "retrieve")
    builder.add_edge("retrieve",     "rerank")
    builder.add_edge("rerank",       "reorder")
    builder.add_edge("reorder",      "build_prompt")
    builder.add_edge("build_prompt", "generate")
    builder.add_edge("generate",     END)

    return builder.compile()


# ── ChatGraph façade ────────────────────────────────────────────────────────────────

class ChatGraph:
    """
    High-level façade over the compiled LangGraph.

    History flow:
        1. Before each `chat()` call, RAGHistoryStore.retrieve_history()
           is called to get the top-k relevant past turns as a string.
        2. That string is passed into the graph as ChatState["rag_history"].
        3. After the graph returns, the new turn is saved via
           RAGHistoryStore.add_turn().

    FIX-P01: source_ids is now forwarded into the graph via ChatState,
             so the retrieve node can filter results to active sources only.

    Usage:
        cg = ChatGraph(faiss_store, sqlite, source_manager, rag_history_store)
        cg.set_llm(LLMRegistry.get())
        result = cg.chat("What is photosynthesis?", source_ids=["bio_101"])
        print(result["response"])
    """

    def __init__(
        self,
        faiss_store:       MultiFAISSStore,
        sqlite:            SQLiteManager,
        source_manager:    SourceManager,
        rag_history_store: Optional[RAGHistoryStore] = None,
        mode:              str  = "chat",
        compressor=None,
        reranker=None,
    ):
        self.faiss_store       = faiss_store
        self.sqlite            = sqlite
        self.source_manager    = source_manager
        self.rag_history_store = rag_history_store
        self.mode              = mode
        self._llm: Optional[Any] = None
        self.graph             = build_chat_graph(faiss_store, sqlite, compressor, reranker, rag_history_store)
        self._session_id: str  = "default"

    def set_llm(self, llm: Any) -> None:
        self._llm = llm

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def set_session(self, session_id: str) -> None:
        """Switch to a different user/session context Scoping."""
        self._session_id = session_id

    def chat(
        self,
        query:           str,
        top_k:           int        = 8,
        score_threshold: float      = 0.0,
        history_top_k:   int        = 4,
        source_ids:      List[str]  = None,   # FIX-P01: forwarded into retrieve node
    ) -> Dict[str, Any]:
        """
        Run one chat turn.

        Parameters
        ----------
        query           : the user's question
        top_k           : number of retrieved source chunks
        score_threshold : minimum rerank score to keep a chunk
        history_top_k   : number of past turns to inject into prompt
        source_ids      : restrict retrieval to these source IDs (empty = all)
        """
        # 1. Retrieve RAG-based history
        rag_history = ""
        if self.rag_history_store is not None:
            try:
                rag_history = self.rag_history_store.retrieve_history(
                    self._session_id, query, top_k=history_top_k
                )
            except Exception as exc:
                logger.warning("[ChatGraph] history retrieval failed: %s", exc)

        # 2. Run graph  (FIX-P01: pass source_ids into state)
        initial_state: ChatState = {
            "query":           query,
            "rag_history":     rag_history,
            "mode":            self.mode,
            "llm":             self._llm,
            "top_k":           top_k,
            "score_threshold": score_threshold,
            "source_ids":      list(source_ids) if source_ids else [],
            "session_id":      self._session_id,
        }
        result = self.graph.invoke(initial_state)

        # 3. Persist this turn (only if not a cache hit to prevent duplicates)
        if self.rag_history_store is not None and not result.get("cached_response"):
            try:
                self.rag_history_store.add_turn(
                    self._session_id,
                    query,
                    result.get("response", ""),
                )
            except Exception as exc:
                logger.warning("[ChatGraph] history save failed: %s", exc)

        return result
