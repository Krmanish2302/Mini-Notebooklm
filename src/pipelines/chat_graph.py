"""
chat_graph.py  —  LangGraph ChatGraph

The query → retrieve → generate pipeline as a LangGraph StateGraph.

Graph topology:
    embed_query → retrieve → build_prompt → generate → done

Multi-index retrieval:
    The retrieve node queries ALL active FAISS indexes in parallel (one per
    active embedding model dimension), then applies Reciprocal Rank Fusion
    to merge results into a single ranked list before passing to the LLM.

Mid-chat ingest:
    The ChatGraph exposes add_source_background() which delegates to
    IngestGraph.run_background().  The chat loop is never blocked.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypedDict, Iterator

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
    # LangGraph managed message history
    messages:       List[BaseMessage]
    # Current query
    query:          str
    # Retrieved context
    retrieved:      List[Dict[str, Any]]
    # Built prompt string
    prompt:         str
    # LLM response
    response:       str
    # Mode: chat | deep_research | study
    mode:           str
    # LLM callable  (set once, persists in state)
    llm:            Any
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
            # Build query vectors for each active dim
            # EmbeddingRegistry.get_by_dim returns the model registered for that dim
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
        from SQLite.
        """
        if state.get("error"):
            return state
        try:
            qvecs = state.get("_query_vectors", {})  # type: ignore[typeddict-item]
            if not qvecs:
                return {**state, "retrieved": []}

            # Search all dims
            raw_results = faiss_store.search(qvecs, k=10)

            # RRF fusion
            fused_ids = _rrf_fuse(raw_results, k=RRF_K)

            # Hydrate from SQLite (content, metadata)
            chunks = []
            for cid in fused_ids[:8]:  # top 8 after fusion
                content = sqlite.get_chunk_content(cid)
                if content:
                    chunks.append({"id": cid, "content": content})

            return {**state, "retrieved": chunks}
        except Exception as exc:
            logger.error("ChatGraph[retrieve]: %s", exc)
            return {**state, "error": str(exc)}
    return node_retrieve


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
            # Format into a string for the LLM invoke call
            history = state.get("messages", [])[:-1]  # exclude latest human msg
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
) -> StateGraph:
    """Build and compile the LangGraph ChatGraph."""
    builder = StateGraph(ChatState)

    builder.add_node("embed_query",  _make_node_embed_query(faiss_store))
    builder.add_node("retrieve",     _make_node_retrieve(faiss_store, sqlite))
    builder.add_node("build_prompt", _make_node_build_prompt())
    builder.add_node("generate",     _make_node_generate())

    builder.set_entry_point("embed_query")
    builder.add_edge("embed_query",  "retrieve")
    builder.add_edge("retrieve",     "build_prompt")
    builder.add_edge("build_prompt", "generate")
    builder.add_edge("generate",     END)

    checkpointer = MemorySaver()  # in-memory per-session history
    return builder.compile(checkpointer=checkpointer)


# ── ChatGraph façade ──────────────────────────────────────────────────────────

class ChatGraph:
    """
    High-level façade over the compiled LangGraph ChatGraph.

    Usage:
        cg = ChatGraph(faiss_store, sqlite, source_manager)
        cg.set_llm(llm_instance)

        # Normal chat
        response = cg.chat("What is attention mechanism?", session_id="abc")

        # Mid-chat: user drops a new PDF while chatting
        cg.add_source_background(
            source_config={"file_path": "new.pdf", "source_type": "pdf",
                           "chunking_strategy": "recursive",
                           "embedding_model": "all-MiniLM-L6-v2"},
            on_complete=lambda sid: st.toast(f"✅ New source ready: {sid}"),
        )
        # Chat continues immediately — ingest runs in background thread
    """

    def __init__(
        self,
        faiss_store: MultiFAISSStore,
        sqlite: SQLiteManager,
        source_manager: SourceManager,
        mode: str = "chat",
    ):
        self.faiss_store = faiss_store
        self.sqlite = sqlite
        self.source_manager = source_manager
        self.mode = mode
        self._llm: Optional[Any] = None
        self.graph = build_chat_graph(faiss_store, sqlite)
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
        stream: bool = False,
    ) -> str:
        """
        Send a message and get a response.
        LangGraph MemorySaver persists message history per session_id.
        """
        if not self._llm:
            raise ValueError("LLM not set. Call set_llm() first.")

        config = {"configurable": {"thread_id": session_id}}

        # Get existing state to append to message history
        existing = self.graph.get_state(config)
        history = existing.values.get("messages", []) if existing.values else []
        history = list(history) + [HumanMessage(content=query)]

        initial_state: ChatState = {
            "messages": history,
            "query":    query,
            "mode":     self.mode,
            "llm":      self._llm,
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
        logger.info("ChatGraph: background ingest dispatched for %s", source_config.get("source_type"))
