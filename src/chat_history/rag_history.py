from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGChatHistory:
    """
    Vector-based chat history for Chat and Deep Research modes.
    Stores messages and retrieves relevant past messages using similarity.
    """
    
    def __init__(self, session_id: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.session_id = session_id
        self.messages: List[Dict[str, Any]] = []
        self.embedder = SentenceTransformer(embedding_model)
        self.message_embeddings: List[np.ndarray] = []
    
    def add_message(self, role: str, content: str, sources_used: List[str] = None):
        """Add message to history."""
        message = {
            "id": f"{self.session_id}_{len(self.messages)}",
            "session_id": self.session_id,
            "role": role,
            "content": content,
            "sources_used": sources_used or [],
            "index": len(self.messages)
        }
        self.messages.append(message)
        
        # Embed for retrieval
        embedding = self.embedder.encode(content)
        self.message_embeddings.append(embedding)
    
    def get_relevant_history(self, query: str, k: int = 5) -> str:
        """Retrieve relevant past messages for query context."""
        if not self.messages:
            return ""
        
        query_emb = self.embedder.encode(query)
        
        # Calculate similarities
        similarities = [
            np.dot(query_emb, msg_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(msg_emb))
            for msg_emb in self.message_embeddings
        ]
        
        # Get top-k most relevant
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        relevant_messages = []
        for idx in top_indices:
            msg = self.messages[idx]
            relevant_messages.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(relevant_messages)
    
    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get last N messages."""
        return self.messages[-n:]
    
    def clear(self):
        """Clear all history."""
        self.messages = []
        self.message_embeddings = []