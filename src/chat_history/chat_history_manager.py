from typing import Dict, Any, Optional
from .rag_history import RAGChatHistory
from .graph_history import GraphChatHistory
from src.graph.graph_storage import GraphStorage

class ChatHistoryManager:
    """
    Unified interface for chat history management.
    Automatically selects appropriate backend based on mode.
    """
    
    def __init__(self, session_id: str, mode: str = "chat", graph_storage: GraphStorage = None):
        self.session_id = session_id
        self.mode = mode
        
        if mode == "study":
            if graph_storage is None:
                raise ValueError("Graph storage required for Study Mode")
            self.backend = GraphChatHistory(session_id, graph_storage)
        else:
            self.backend = RAGChatHistory(session_id)
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add message to history."""
        if self.mode == "study":
            self.backend.add_message(role, content, 
                                   concepts=kwargs.get("concepts"),
                                   sources_used=kwargs.get("sources_used"))
        else:
            self.backend.add_message(role, content, 
                                   sources_used=kwargs.get("sources_used"))
    
    def get_history_context(self, query: str = "", max_messages: int = 10) -> str:
        """Get formatted history for LLM context."""
        if self.mode == "study":
            # For study mode, return concept connections
            connections = self.backend.get_concept_connections()
            if connections:
                return f"Previous concepts discussed: {', '.join([c['concept'] for c in connections])}"
            return ""
        else:
            # For chat/deep research, return relevant messages
            if query:
                return self.backend.get_relevant_history(query)
            else:
                recent = self.backend.get_recent_messages(max_messages)
                return "\n".join([f"{m['role']}: {m['content']}" for m in recent])
    
    def clear(self):
        """Clear history."""
        self.backend.clear()