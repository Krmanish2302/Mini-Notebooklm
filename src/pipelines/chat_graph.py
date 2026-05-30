"""
chat_graph.py — LangGraph ChatGraph

Graph topology:
    embed_query → retrieve → compress → rerank → build_prompt → generate → END

Any node failure → error is propagated; generate node returns fallback message.
Uses LangChain-native ChatPromptTemplate + MessagesPlaceholder.
LLM is supplied via ChatState["llm"] so any LangChain BaseChatModel works.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
from src.storage.faiss_store import MultiFAISSStore
from src.storage.sqlite_manager import SQLiteManager
from src.storage.source_manager import SourceManager

logger = logging.getLogger(__name__)

RRF_K = 60

# ── Persona system prompts ────────────────────────────────────────────────────

_SYSTEM_CHAT = (
    "You're Carl Sagan if he were a chill classmate. "
    "Simple words, real-world analogies, poetic touch. "
    "Answer ONLY from the SOURCES in the context. "
    "Cite as [S1], [S2]… "
    "If it's not there, say: 'Not in my notes, bro.'"
)

_SYSTEM_DEEP = (
    "You're Carl Sagan if he were a chill classmate. "
    "Go deep — structured, thorough, cite everything [S1]… "
    "Use ONLY the sources. Never invent facts. "
    "If it's not there, say: 'Not in my notes, bro.'"
)


# ── State ─────────────────────────────────────────────────────────────────────

class ChatState(TypedDict, total=False):
    messages:        List[BaseMessage]
    query:           str
    retrieved:       List[Dict[str, Any]]
    _query_vectors:  Dict[int, Any]      # internal — not exposed outside graph
    prompt:          str
    response:        str
    mode:            str                 # "chat" | "deep_research"
    llm:             Any                 # BaseChatModel
    top_k:           int
    score_threshold: float
    error:           Optional[str]


# ── RRF helper ────────────────────────────────────────────────────────────────

def _rrf_fuse(results_by_dim: Dict[int, List[tuple]], k: int = RRF_K) -> List[str]:
    """Reciprocal Rank Fusion across multiple embedding-dimension result sets."""
    scores: Dict[str, float] = {}
    for dim_results in results_by_dim.values():
        for rank, (chunk_id, _score) in enumerate(dim_results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


# ── Node factories ─────────────────────────────────────────────────────────────

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
            top_k      = state.get("top_k", 8)
            candidate_k = top_k * 3
            raw        = faiss_store.search(qvecs, k=candidate_k)
            fused_ids  = _rrf_fuse(raw)
            chunks = []
            for cid in fused_ids[:candidate_k]:
                content = sqlite.get_chunk_content(cid)
                if content:
                    chunks.append({"id": cid, "content": content})
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

            system = _SYSTEM_DEEP if state.get("mode") == "deep_research" else _SYSTEM_CHAT

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system + "\n\nSOURCES:\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{query}"),
            ])
            # history = all messages except the latest HumanMessage
            history = [m for m in state.get("messages", [])[:-1]]
            messages = prompt_template.format_messages(
                context=source_block,
                history=history,
                query=state["query"],
            )
            return {**state, "_formatted_messages": messages}   # type: ignore[typeddict-item]
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

            formatted_messages = state.get("_formatted_messages")  # type: ignore[typeddict-item]
            if formatted_messages is None:
                raise ValueError("No formatted messages — build_prompt may have failed.")

            ai_msg   = llm.invoke(formatted_messages)
            response = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)

            messages = list(state.get("messages", []))
            messages.append(AIMessage(content=response))
            return {**state, "response": response, "messages": messages}
        except Exception as exc:
            logger.error("[ChatGraph:generate] %s", exc)
            return {
                **state,
                "error": str(exc),
                "response": "I encountered an error — please try again.",
            }
    return node


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_chat_graph(
    faiss_store:  MultiFAISSStore,
    sqlite:       SQLiteManager,
    compressor=None,
    reranker=None,
) -> Any:
    """Build and compile the chat LangGraph."""
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

    return builder.compile(checkpointer=MemorySaver())


# ── ChatGraph façade ──────────────────────────────────────────────────────────

class ChatGraph:
    """
    High-level façade over the compiled LangGraph.

    Usage:
        cg = ChatGraph(faiss_store, sqlite, source_manager)
        cg.set_llm(LLMRegistry.get())
        result = cg.chat("What is photosynthesis?")
        print(result["response"])
    """

    def __init__(
        self,
        faiss_store:    MultiFAISSStore,
        sqlite:         SQLiteManager,
        source_manager: SourceManager,
        mode:           str  = "chat",
        compressor=None,
        reranker=None,
    ):
        self.faiss_store    = faiss_store
        self.sqlite         = sqlite
        self.source_manager = source_manager
        self.mode           = mode
        self._llm: Optional[Any] = None
        self.graph = build_chat_graph(faiss_store, sqlite, compressor, reranker)
        self._thread_id: str = "default"

    def set_llm(self, llm: Any) -> None:
        self._llm = llm

    def set_mode(self, mode: str) -> None:
        assert mode in ("chat", "deep_research"), f"Unknown mode: {mode}"
        self.mode = mode

    def set_thread(self, thread_id: str) -> None:
        """Switch conversation thread (for multi-user / multi-session support)."""
        self._thread_id = thread_id

    def chat(
        self,
        query:          str,
        history:        Optional[List[BaseMessage]] = None,
        top_k:          int   = 8,
        score_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        if self._llm is None:
            raise RuntimeError("Call set_llm() before chat().")

        messages = list(history or [])
        messages.append(HumanMessage(content=query))

        initial_state: ChatState = {
            "messages":        messages,
            "query":           query,
            "mode":            self.mode,
            "llm":             self._llm,
            "top_k":           top_k,
            "score_threshold": score_threshold,
        }

        config = {"configurable": {"thread_id": self._thread_id}}
        final  = self.graph.invoke(initial_state, config=config)

        return {
            "response":  final.get("response", ""),
            "retrieved": final.get("retrieved", []),
            