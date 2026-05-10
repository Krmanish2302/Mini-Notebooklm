"""
chat_graph.py  —  LangGraph ChatGraph

The query → retrieve → generate pipeline as a LangGraph StateGraph.

Graph topology:
    embed_query → retrieve → compress → rerank → build_prompt → generate → done

Multi-index retrieval:
    The retrieve node queries ALL active FAISS indexes in parallel (one per
    active embedding model dimension), then applies Reciprocal Rank Fusion
    to merge results into a single ranked list.

Post-retrieval (Chat mode):
    NODE 2.5 — ContextualCompressor: strips irrelevant sentences from each chunk.
    NODE 2.6 — Reranker (BAAI/bge-reranker-base): cross-encoder rescoring.
    Score threshold: chunks below `score_threshold` are dropped before the LLM.
    Top-K: user-controlled, stored in ChatState as `top_k` (default 8).

Mid-chat ingest:
    The ChatGraph exposes add_source_background() which delegates to
    IngestGraph.run_background().  The chat loop is never blocked.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
from src.storage.faiss_store import MultiFAISSStore
from src.storage.sqlite_manager import SQLiteManager
from src.storage.source_manager import SourceManager
from src.pipelines.ingest_graph import IngestGraph

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF smoothing constant


# ── State ─────────────────────────────────────────────────────────────────────

class ChatState(TypedDict, total=False):
    messages:       List[BaseMessage]
    query:          str
    retrieved:      List[Dict[str, Any]]
    prompt:         str
    response:       str
    mode:           str
    llm:            Any
    # Retrieval controls (user-configurable)
    top_k:          int    # how many final chunks to send to LLM (default 8)
    score_threshold: float  # drop chunks with rerank_score below this (default 0.0)
    error:          Optional[str]


# ── RRF Fusion ────────────────────────────────────────────────────────────────

def _rrf_fuse(
    results_by_dim: Dict[int, List[tuple]],
    k: int = RRF_K,
) -> List[str]:
    """
    Reciprocal Rank Fusion across multiple FAISS result lists.

    Args:
        results_by_dim: {dim: [(chunk_id, score), ...]} — sorted by score desc
        k:              RRF smoothing constant (default 60)

    Returns:
        List of chunk_ids sorted by fused RRF score (best first).
    """
    scores: Dict[str, float] = {}
    for dim_results in results_by_dim.values():
        for rank, (chunk_id, _score) in enumerate(dim_results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def _make_node_embed_query(faiss_store: MultiFAISSStore):
    def node_embed_query(state: ChatState) -> ChatState:
        """
        Embed the query using every model corresponding to an active FAISS dim.
        Stores query vectors as a transient dict on state.
        """
        try:
            active_dims = faiss_store.active_dims()
            if not active_dims:
                return {**state, "retrieved": [], "error": None}
            query_vectors: Dict[int, Any] = {}
            for dim in active_dims:
                model = EmbeddingRegistry.get_by_dim(dim)
                if model:
                    query_vectors[dim] = model.embed_query(state["query"])
            state = {**state, "_query_vectors": query_vectors}  # type: ignore[typeddict-item]
            return state
        except Exception as exc:
            logger.error("ChatGraph[embed_query]: %s", exc)
            return {**state, "error": str(exc)}
    return node_embed_query


def _make_node_retrieve(faiss_store: MultiFAISSStore, sqlite: SQLiteManager):
    def node_retrieve(state: ChatState) -> ChatState:
        """
        Search all active FAISS indexes, apply RRF fusion, hydrate chunk content
        from SQLite.  Fetches top_k * 3 candidates so compression+rerank have
        enough headroom to filter down to the user's desired top_k.
        """
        if state.get("error"):
            return state
        try:
            qvecs = state.get("_query_vectors", {})  # type: ignore[typeddict-item]
            if not qvecs:
                return {**state, "retrieved": []}

            top_k = state.get("top_k", 8)
            # Fetch 3x candidates so compression/rerank have room to filter
            candidate_k = top_k * 3

            raw_results = faiss_store.search(qvecs, k=candidate_k)
            fused_ids = _rrf_fuse(raw_results, k=RRF_K)

            chunks = []
            for cid in fused_ids[:candidate_k]:
                content = sqlite.get_chunk_content(cid)
                if content:
                    chunks.append({"id": cid, "content": content})

            return {**state, "retrieved": chunks}
        except Exception as exc:
            logger.error("ChatGraph[retrieve]: %s", exc)
            return {**state, "error": str(exc)}
    return node_retrieve


def _make_node_compress(compressor):
    """
    NODE 2.5 — Contextual Compression.

    Uses ContextualCompressor to strip each chunk down to only the sentences
    relevant to the query.  Irrelevant chunks are dropped entirely.
    If no compressor is injected, this node is a transparent pass-through.
    """
    def node_compress(state: ChatState) -> ChatState:
        if state.get("error"):
            return state
        chunks = state.get("retrieved", [])
        if not chunks or compressor is None:
            return state
        try:
            compressed = compressor.compress(chunks, state["query"])
            logger.debug(
                "ChatGraph[compress]: %d → %d chunks after compression",
                len(chunks), len(compressed),
            )
            return {**state, "retrieved": compressed}
        except Exception as exc:
            logger.warning("ChatGraph[compress] failed, using raw chunks: %s", exc)
            return state  # graceful fallback — use uncompressed chunks
    return node_compress


def _make_node_rerank(reranker):
    """
    NODE 2.6 — Cross-Encoder Reranking + Score Threshold + Top-K trim.

    1. Reranks retrieved chunks with BAAI/bge-reranker-base.
    2. Drops any chunk whose rerank_score < state['score_threshold'].
    3. Returns the best state['top_k'] chunks to the prompt builder.

    If no reranker is injected, falls back to simple top-k slice.
    """
    def node_rerank(state: ChatState) -> ChatState:
        if state.get("error"):
            return state
        chunks = state.get("retrieved", [])
        if not chunks:
            return state

        top_k = state.get("top_k", 8)
        threshold = state.get("score_threshold", 0.0)

        try:
            if reranker is not None:
                reranked = reranker.rerank(state["query"], chunks, top_k=len(chunks))
                # Apply score threshold
                filtered = [
                    c for c in reranked
                    if c.get("rerank_score", 1.0) >= threshold
                ]
                logger.debug(
                    "ChatGraph[rerank]: %d → %d chunks after threshold=%.2f",
                    len(reranked), len(filtered), threshold,
                )
                final = filtered[:top_k]
            else:
                # No reranker: just apply top_k slice
                final = chunks[:top_k]

            return {**state, "retrieved": final}
        except Exception as exc:
            logger.warning("ChatGraph[rerank] failed, using top_k slice: %s", exc)
            return {**state, "retrieved": chunks[:top_k]}
    return node_rerank


def _make_node_build_prompt():
    def node_build_prompt(state: ChatState) -> ChatState:
        """Build prompt from retrieved context + message history."""
        if state.get("error"):
            return state
        try:
            context = "\n\n---\n\n".join(
                c["content"] for c in state.get("retrieved", [])
            )
            mode = state.get("mode", "chat")
            if mode == "deep_research":
                system = (
                    "You are a research assistant. Answer in depth using ONLY "
                    "the provided context. Cite sources inline."
                )
            else:
                system = (
                    "You are a helpful assistant. Answer concisely using ONLY "
                    "the provided context. If unsure, say so."
                )
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system + "\n\nContext:\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{query}"),
            ])
            history = state.get("messages", [])[:-1]
            formatted = prompt_template.format_messages(
                context=context,
                history=history,
                query=state["query"],
            )
            prompt_str = "\n".join(m.content for m in formatted)
            return {**state, "prompt": prompt_str}
        except Exception as exc:
            logger.error("ChatGraph[build_prompt]: %s", exc)
            return {**state, "error": str(exc)}
    return node_build_prompt


def _make_node_generate():
    def node_generate(state: ChatState) -> ChatState:
        """Call the LLM and append response to message history."""
        if state.get("error"):
            return state
        try:
            llm = state.get("llm")
            if not llm:
                raise ValueError("LLM not configured in ChatState.")
            response_text = llm.invoke(state["prompt"])
            if hasattr(response_text, "content"):
                response_text = response_text.content
            messages = list(state.get("messages", []))
            messages.append(AIMessage(content=response_text))
            return {**state, "response": response_text, "messages": messages}
        except Exception as exc:
            logger.error("ChatGraph[generate]: %s", exc)
            return {**state, "error": str(exc), "response": "I encountered an error. Please try again."}
    return node_generate


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_chat_graph(
    faiss_store: MultiFAISSStore,
    sqlite: SQLiteManager,
    compressor=None,
    reranker=None,
) -> StateGraph:
    """
    Build and compile the LangGraph ChatGraph.

    Args:
        faiss_store : MultiFAISSStore instance
        sqlite      : SQLiteManager instance
        compressor  : ContextualCompressor instance (optional — pass-through if None)
        reranker    : Reranker instance (optional — top_k slice if None)
    """
    builder = StateGraph(ChatState)

    builder.add_node("embed_query",  _make_node_embed_query(faiss_store))
    builder.add_node("retrieve",     _make_node_retrieve(faiss_store, sqlite))
    builder.add_node("compress",     _make_node_compress(compressor))
    builder.add_node("rerank",       _make_node_rerank(reranker))
    builder.add_node("build_prompt", _make_node_build_prompt())
    builder.add_node("generate",     _make_node_generate())

    builder.set_entry_point("embed_query")
    builder.add_edge("embed_query",  "retrieve")
    builder.add_edge("retrieve",     "compress")
    builder.add_edge("compress",     "rerank")
    builder.add_edge("rerank",       "build_prompt")
    builder.add_edge("build_prompt", "generate")
    builder.add_edge("generate",     END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ── ChatGraph façade ──────────────────────────────────────────────────────────

class ChatGraph:
    """
    High-level façade over the compiled LangGraph ChatGraph.

    Usage:
        cg = ChatGraph(
            faiss_store, sqlite, source_manager,
            compressor=ContextualCompressor(llm),
            reranker=Reranker(),
        )
        cg.set_llm(llm_instance)

        # Standard chat — default top_k=8, threshold=0.0
        response = cg.chat("What is attention?", session_id="abc")

        # User bumps top_k to 12, drops weak chunks
        response = cg.chat("Explain RLHF", session_id="abc",
                           top_k=12, score_threshold=0.3)

        # Mid-chat background ingest
        cg.add_source_background(
            source_config={...},
            on_complete=lambda sid: st.toast(f"✅ Ready: {sid}"),
        )
    """

    def __init__(
        self,
        faiss_store: MultiFAISSStore,
        sqlite: SQLiteManager,
        source_manager: SourceManager,
        mode: str = "chat",
        compressor=None,
        reranker=None,
    ):
        self.faiss_store = faiss_store
        self.sqlite = sqlite
        self.source_manager = source_manager
        self.mode = mode
        self._llm: Optional[Any] = None
        self.graph = build_chat_graph(faiss_store, sqlite, compressor, reranker)
        self._ingest_graph = IngestGraph(source_manager)

    def set_llm(self, llm: Any) -> None:
        """Set the LangChain LLM instance (any ChatModel)."""
        self._llm = llm

    def set_mode(self, mode: str) -> None:
        """Switch mode without destroying session history."""
        self.mode = mode

    def chat(
        self,
        query: str,
        session_id: str = "default",
        top_k: int = 8,
        score_threshold: float = 0.0,
        stream: bool = False,
    ) -> str:
        """
        Send a message and get a response.

        Args:
            query           : user query
            session_id      : LangGraph thread id (per-user session)
            top_k           : number of chunks to pass to LLM (user-configurable)
            score_threshold : drop chunks with rerank_score below this value.
                              Range 0.0–1.0; 0.0 = keep all, 0.5 = strict.
        """
        if not self._llm:
            raise ValueError("LLM not set. Call set_llm() first.")

        config = {"configurable": {"thread_id": session_id}}

        existing = self.graph.get_state(config)
        history = existing.values.get("messages", []) if existing.values else []
        history = list(history) + [HumanMessage(content=query)]

        initial_state: ChatState = {
            "messages":       history,
            "query":          query,
            "mode":           self.mode,
            "llm":            self._llm,
            "top_k":          top_k,
            "score_threshold": score_threshold,
        }

        result = self.graph.invoke(initial_state, config=config)

        if result.get("error"):
            logger.warning("ChatGraph error: %s", result["error"])

        return result.get("response", "")

    # ── mid-chat background ingest ────────────────────────────────────────────

    def add_source_background(
        self,
        source_config: Dict[str, Any],
        on_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Ingest a new source in the background while chat continues.

        Args:
            source_config: {
                "file_path":         str (for pdf/csv),
                "url":               str (for website/youtube),
                "source_type":       str,
                "chunking_strategy": str,
                "embedding_model":   str,
            }
            on_complete: callback(source_id) when ingest finishes
            on_error:    callback(error_str) on failure
        """
        self._ingest_graph.run_background(
            source_config=source_config,
            on_complete=on_complete,
            on_error=on_error,
        )
        logger.info(
            "ChatGraph: background ingest dispatched for %s",
            source_config.get("source_type"),
        )
