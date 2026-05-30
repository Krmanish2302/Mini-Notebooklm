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


def _make_retrieve(faiss_store: MultiFAISSStore, sqlite: SQLiteManager):
    def node(state: ChatState) -> ChatState:
        if state.get("error"):
            return state
        try:
            qvecs = state.get("_query_vectors", {})
            if not qvecs:
                return {**state, "retrieved": []}

            top_k        = state.get("top_k", 8)
            candidate_k  = top_k * 3
            # FIX-P01: build allowed set (empty = no filter = search all)
            allowed_ids: set = set(state.get("source_ids") or [])

            raw       = faiss_store.search(qvecs, k=candidate_k)
            fused_ids = _rrf_fuse(raw)

            chunks = []
            for cid in fused_ids[:candidate_k]:
                # FIX-P01: parse source_id prefix from chunk_id ("<source_id>_<index>")
                if allowed_ids:
                    # chunk_id format: "<source_id>_<int>"  e.g. "bio_101_42"
                    # source_id may itself contain underscores, so split from right once
                    chunk_source = "_".join(cid.rsplit("_", 1)[:-1])
                    if chunk_source not in allowed_ids:
                        continue

                content = sqlite.get_chunk_content(cid)
                if content:
                    chunks.append({"id": cid, "content": content})

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


def _make_compress(compressor):
    def node(state: ChatState) -> ChatState:
        if state.get("error") or compressor is None:
            return state
        chunks = state.get("retrieved", [])
        if not chunks:
            return state
        try:
            compressed = compressor.compress(chunks, state["query"])
            logger.debug("[ChatGraph:compress] %d → %d", len(chunks), len(compressed))
            return {**state, "retrieved": compressed}
        except Exception as exc:
            logger.warning("[ChatGraph:compress] fallback — %s", exc)
            return state
    return node


def _make_rerank(reranker):
    def node(state: ChatState) -> ChatState:
        if state.get("error"):
            return state
        chunks    = state.get("retrieved", [])
        top_k     = state.get("top_k", 8)
        threshold = state.get("score_threshold", 0.0)
        if not chunks:
            return state
        try:
            if reranker is not None:
                reranked = reranker.rerank(state["query"], chunks, top_k=len(chunks))
                filtered = [c for c in reranked if c.get("rerank_score", 1.0) >= threshold]
                logger.debug(
                    "[ChatGraph:rerank] %d → %d (threshold=%.2f)",
                    len(reranked), len(filtered), threshold,
                )
                return {**state, "retrieved": filtered[:top_k]}
            return {**state, "retrieved": chunks[:top_k]}
        except Exception as exc:
            logger.warning("[ChatGraph:rerank] fallback — %s", exc)
            return {**state, "retrieved": chunks[:top_k]}
    return node


def _make_build_prompt():
    def node(state: ChatState) -> ChatState:
        if state.get("error"):
            return state
        try:
            source_block = "\n\n".join(
                f"[S{i+1}] {c['content']}"
                for i, c in enumerate(state.get("retrieved", []))
            ) or "[No sources retrieved]"

            rag_history  = state.get("rag_history", "")
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
) -> Any:
    """Build and compile the chat LangGraph. No checkpointer needed — history is RAG-based."""
    builder = StateGraph(ChatState)
    builder.add_node("embed_query",  _make_embed_query(faiss_store))
    builder.add_node("retrieve",     _make_retrieve(faiss_store, sqlite))
    builder.add_node("compress",     _make_compress(compressor))
    builder.add_node("rerank",       _make_rerank(reranker))
    builder.add_node("build_prompt", _make_build_prompt())
    builder.add_node("generate",     _make_generate())

    builder.set_entry_point("embed_query")
    builder.add_edge("embed_query",  "retrieve")
    builder.add_edge("retrieve",     "compress")
    builder.add_edge("compress",     "rerank")
    builder.add_edge("rerank",       "build_prompt")
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
        self.graph             = build_chat_graph(faiss_store, sqlite, compressor, reranker)
        self._session_id: str  = "default"

    def set_llm(self, llm: Any) -> None:
        self._llm = llm

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def set_session(self, session_id: str) -> None:
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
        }
        result = self.graph.invoke(initial_state)

        # 3. Persist this turn
        if self.rag_history_store is not None:
            try:
                self.rag_history_store.add_turn(
                    self._session_id,
                    query,
                    result.get("response", ""),
                )
            except Exception as exc:
                logger.warning("[ChatGraph] history save failed: %s", exc)

        return result
