from typing import List, Dict, Any, Optional
from src.graph.graph_storage import GraphStorage

class GraphChatHistory:
    """
    Graph-based chat history for Study Mode.
    Tracks concept relationships and learning sequences.
    """
    
    def __init__(self, session_id: str, graph_storage: GraphStorage):
        self.session_id = session_id
        self.graph = graph_storage
        self.messages: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: str, concepts: List[str] = None, 
                   sources_used: List[str] = None):
        """Add message with concept tracking."""
        message_id = f"{self.session_id}_{len(self.messages)}"
        
        message = {
            "id": message_id,
            "session_id": self.session_id,
            "role": role,
            "content": content,
            "concepts": concepts or [],
            "sources_used": sources_used or [],
            "index": len(self.messages)
        }
        self.messages.append(message)
        
        # Add to graph as node
        self.graph.add_chunk({
            "id": message_id,
            "content": content,
            "modality": "chat_message",
            "source_id": self.session_id,
            "metadata": {"role": role, "concepts": concepts}
        })
        
        # Link to previous message
        if len(self.messages) > 1:
            prev_id = self.messages[-2]["id"]
            self.graph.add_relationship(
                prev_id, message_id, "followed_by", weight=1.0
            )
        
        # Link concepts
        if concepts:
            for concept in concepts:
                # Create concept node if not exists
                concept_id = f"concept_{concept}_{self.session_id}"
                if concept_id not in self.graph.graph:
                    self.graph.add_chunk({
                        "id": concept_id,
                        "content": concept,
                        "modality": "concept",
                        "source_id": self.session_id
                    })
                self.graph.add_relationship(
                    message_id, concept_id, "mentions", weight=0.8
                )
    
    def get_learning_path(self, concept: str) -> List[Dict[str, Any]]:
        """Get learning sequence for a concept."""
        concept_id = f"concept_{concept}_{self.session_id}"
        if concept_id not in self.graph.graph:
            return []
        
        # Find all messages mentioning this concept
        related = self.graph.get_related(concept_id, depth=2)
        return related
    
    def get_concept_connections(self) -> List[Dict[str, Any]]:
        """Get all concept relationships in this session."""
        connections = []
        for node_id, data in self.graph.graph.nodes(data=True):
            if data.get("modality") == "concept":
                related = self.graph.get_related(node_id, depth=1)
                connections.append({
                    "concept": data.get("content", ""),
                    "related": related
                })
        return connections