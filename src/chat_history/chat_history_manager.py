"""
chat_history_manager.py — Unified history dispatcher.

LangChain upgrade notes
-----------------------
* Implements BaseChatMemory interface:
    - memory_variables  → ["history"]
    - load_memory_variables(inputs) → {"history": str}
    - save_context(inputs, outputs) → persists user + ai turns
    - clear()

* This makes ChatHistoryManager a drop-in for any LangChain chain:

    from langchain.chains import ConversationChain
    from langchain_core.runnables.history import RunnableWithMessageHistory

    chain = RunnableWithMessageHistory(
        runnable=your_chain,
        get_session_history=lambda sid: ChatHistoryManager(sid, mode="chat"),
    )

* Mode → backend routing (unchanged):
    "chat"     → RAGChatHistory
    "research" → RAGChatHistory
    "study"    → GraphHistory  (requires knowledge_graph)

* Class-name fix:
    Original imported 'GraphChatHistory' which didn't match graph_history.py export.
    Now imports 'GraphHistory' (canonical); GraphChatHistory alias handles old code.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage
from pydantic import Field

from .rag_history   import RAGChatHistory
from .graph_history import GraphHistory

logger = logging.getLogger(__name__)


class ChatHistoryManager(BaseMemory):
    """
    Unified session history manager.

    Implements LangChain BaseChatMemory so it drops into any chain or
    RunnableWithMessageHistory without modification.

    Parameters
    ----------
    session_id      : str
    mode            : "chat" | "research" | "study"
    knowledge_graph : KnowledgeGraph — required when mode=="study"
    embedding_model : HuggingFace model name for RAGChatHistory
    """

    # Pydantic v2 fields (required by BaseMemory)
    session_id:      str                       = Field(...)
    mode:            str                       = Field(default="chat")
    memory_key:      str                       = Field(default="history")
    return_messages: bool                      = Field(default=False)

    # Private — not exposed as Pydantic fields
    _backend: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        session_id:      str,
        mode:            str  = "chat",
        knowledge_graph: Any  = None,
        embedding_model: str  = "all-MiniLM-L6-v2",
        **kwargs,
    ) -> None:
        super().__init__(
            session_id=session_id,
            mode=mode,
            **kwargs,
        )
        if mode == "study":
            if knowledge_graph is None:
                raise ValueError("knowledge_graph required for mode='study'")
            object.__setattr__(self, "_backend", GraphHistory(session_id, knowledge_graph))
        else:
            object.__setattr__(
                self, "_backend",
                RAGChatHistory(session_id, embedding_model=embedding_model)
            )
        logger.info(
            "[ChatHistoryManager] session=%s mode=%s backend=%s",
            session_id, mode, type(self._backend).__name__,
        )

    # ── BaseChatMemory interface ───────────────────────────────────────────────

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called by LangChain chains before each LLM call.
        Returns {"history": <formatted string or message list>}.
        """
        query = inputs.get("input", inputs.get("query", ""))
        ctx   = self.get_history_context(query=query)
        if self.return_messages:
            return {self.memory_key: self._backend.lc_messages}
        return {self.memory_key: ctx}

    def save_context(
        self,
        inputs:  Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        """
        Called by LangChain chains after each LLM call.
        Persists the user turn and the AI response.
        """
        human_text = inputs.get("input", inputs.get("query", ""))
        ai_text    = outputs.get("output", outputs.get("answer", ""))
        if human_text:
            self.add_message("user", human_text)
        if ai_text:
            self.add_message("assistant", ai_text)

    def clear(self) -> None:
        self._backend.clear()

    # ── Domain API (used by pipeline nodes) ──────────────────────────────────

    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the active backend."""
        if self.mode == "study":
            self._backend.add_message(
                role, content,
                concepts=kwargs.get("concepts"),
                sources_used=kwargs.get("sources_used"),
            )
        else:
            self._backend.add_message(
                role, content,
                sources_used=kwargs.get("sources_used"),
            )

    def get_history_context(self, query: str = "", max_messages: int = 10) -> str:
        """
        Formatted history string for injection into LLM prompts.
        Delegates to the active backend.
        """
        if self.mode == "study":
            connections = self._backend.get_concept_connections()
            if connections:
                concepts = ", ".join(c["concept"] for c in connections)
                return f"Previously discussed concepts: {concepts}"
            return ""
        else:
            if query:
                return self._backend.format_for_prompt(query)
            recent = self._backend.get_recent_messages(max_messages)
            return "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    # ── LangChain interop ─────────────────────────────────────────────────────

    @property
    def lc_messages(self) -> List[BaseMessage]:
        """Direct access to LangChain BaseMessage list."""
        return self._backend.lc_messages

    @property
    def backend(self) -> Any:
        """Direct access to the underlying RAGChatHistory or GraphHistory."""
        return self._backend