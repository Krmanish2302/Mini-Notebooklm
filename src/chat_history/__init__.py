"""
src/chat_history — Session-scoped chat history backends.

Public API:
    RAGChatHistory       — vector-RAG history for Chat / Deep Research modes
    GraphHistory         — concept-graph history for Study mode
    ChatHistoryManager   — unified dispatcher; implements BaseChatMemory
                           → drop-in for RunnableWithMessageHistory

Usage:
    from src.chat_history import ChatHistoryManager, RAGChatHistory, GraphHistory
"""
from .rag_history          import RAGChatHistory          # noqa: F401
from .graph_history        import GraphHistory             # noqa: F401
from .chat_history_manager import ChatHistoryManager       # noqa: F401

__all__ = ["RAGChatHistory", "GraphHistory", "ChatHistoryManager"]