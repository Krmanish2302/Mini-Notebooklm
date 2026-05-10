"""
chat_graph.py  —  LangGraph ChatGraph

Graph topology:
    embed_query → retrieve → compress → rerank → build_prompt → generate → done

Persona: Carl Sagan as a chill classmate. Strictly source-grounded.
See src/generation/prompt_builder.py for the shared persona + grounding block.
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

RRF_K = 60

# ── Persona + grounding (token-efficient, Carl Sagan style) ───────────────
_SYSTEM_CHAT = (
    "You're Carl Sagan if he were a chill classmate. "
    "Simple words, real-world analogies, poetic touch. "
    "Answer ONLY from the SOURCES in the context. "
    "Cite as [S1], [S2]… "
    "If it’s not there, say: 'Not in my notes, bro.'"
)

_SYSTEM_DEEP = (
    "You're Carl Sagan if he were a chill classmate. "
    "Go deep — structured, thorough, cite everything [S1]… "
    "Use ONLY the sources. Never invent facts. "
    "If it’s not there, say: 'Not in my notes, bro.'"
)


# ── State ─────────────────────────────────────────────────────────────────────

class ChatState(TypedDict, total=False):
    messages:        List[BaseMessage]
    query:           str
    retrieved:       List[Dict[str, Any]]
    prompt:          str
    response:        str
    mode:            str
    llm:             Any
    top_k:           int
    score_threshold: float
    error:           Optional[str]


# ── RRF ───────────────────────────────────────────────────────────────────────

def _rrf_fuse(results_by_dim: Dict[int, List[tuple]], k: int = RRF_K) -> List[str]:
    scores: Dict[str, float] = {}
    for dim_results in results_by_dim.values():
        for rank, (chunk_id, _score) in enumerate(dim_results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def _make_node_embed_query(faiss_store: MultiFAISSStore):
    def node_embed_query(state: ChatState) -> ChatState:
        try:
            active_dims = faiss_store.active_dims()
            if not active_dims:
                return {**state, "retrieved": [], "error": None}
            query_vectors: Dict[int, Any] = {}
            for dim in active_dims:
                model = EmbeddingRegistry.get_by_dim(dim)
                if model:
                    query_vectors[dim] = model.embed_query(state["query"])
            return {**state, "_query_vectors": query_vectors}  # type: ignore[typeddict-item]
        except Exception as exc:
            logger.error("ChatGraph[embed_query]: %s", exc)
            return {**state, "error": str(exc)}
    return node_embed_query


def _make_node_retrieve(faiss_store: MultiFAISSStore, sqlite: SQLiteManager):
    def node_retrieve(state: ChatState) -> ChatState:
        if state.get("error"):
            return state
        try:
            qvecs = state.get("_query_vectors", {})  # type: ignore[typeddict-item]
            if not qvecs:
                return {**state, "retrieved": []}
            top_k = state.get("top_k", 8)
            candidate_k = top_k * 3
            raw_results = faiss_store.search(qvecs, k=candidate_k)
            fused_ids = _rrf_fuse(raw_results)
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
    def node_compress(state: ChatState) -> ChatState:
        if state.get("error") or compressor is None:
            return state
        chunks = state.get("retrieved", [])
        if not chunks:
            return state
        try:
            compressed = compressor.compress(chunks, state["query"])
            logger.debug("ChatGraph[compress]: %d → %d", len(chunks), len(compressed))
            return {**state, "retrieved": compressed}
        except Exception as exc:
            logger.warning("ChatGraph[compress] fallback: %s", exc)
            return state
    return node_compress


def _make_node_rerank(reranker):
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
                filtered = [c for c in reranked if c.get("rerank_score", 1.0) >= threshold]
                logger.debug(
                    "ChatGraph[rerank]: %d → %d (threshold=%.2f)",
                    len(reranked), len(filtered), threshold,
                )
                return {**state, "retrieved": filtered[:top_k]}
            return {**state, "retrieved": chunks[:top_k]}
        except Exception as exc:
            logger.warning("ChatGraph[rerank] fallback: %s", exc)
            return {**state, "retrieved": chunks[:top_k]}
    return node_rerank


def _make_node_build_prompt():
    def node_build_prompt(state: ChatState) -> ChatState:
        if state.get("error"):
            return state
        try:
            # Build compact numbered source block — [S1], [S2]…
            source_block = "\n\n".join(
                f"[S{i+1}] {c['content']}"
                for i, c in enumerate(state.get("retrieved", []))
            )
            mode = state.get("mode", "chat")
            system = _SYSTEM_DEEP if mode == "deep_research" else _SYSTEM_CHAT

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system + "\n\nSOURCES:\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{query}"),
            ])
            history = state.get("messages", [])[:-1]
            formatted = prompt_template.format_messages(
                context=source_block,
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
    return builder.compile(checkpointer=MemorySaver())


# ── ChatGraph façade ──────────────────────────────────────────────────────────

class ChatGraph:
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
        self._llm = llm

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def chat(
        self,
        query: str,
        session_id: str = "default",
        top_k: int = 8,
        score_threshold: float = 0.0,
        stream: bool = False,
    ) -> str:
        if not self._llm:
            raise ValueError("LLM not set. Call set_llm() first.")
        config = {"configurable": {"thread_id": session_id}}
        existing = self.graph.get_state(config)
        history = existing.values.get("messages", []) if existing.values else []
        history = list(history) + [HumanMessage(content=query)]
        initial_state: ChatState = {
            "messages":        history,
            "query":           query,
            "mode":            self.mode,
            "llm":             self._llm,
            "top_k":           top_k,
            "score_threshold": score_threshold,
        }
        result = self.graph.invoke(initial_state, config=config)
        if result.get("error"):
            logger.warning("ChatGraph error: %s", result["error"])
        return result.get("response", "")

    def add_source_background(
        self,
        source_config: Dict[str, Any],
        on_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._ingest_graph.run_background(
            source_config=source_config,
            on_complete=on_complete,
            on_error=on_error,
        )
        logger.info("ChatGraph: background ingest dispatched for %s", source_config.get("source_type"))
