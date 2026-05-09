from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseChunker(ABC):
    """Abstract base for all chunkers."""
    
    @abstractmethod
    def chunk(self, content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split content into chunks. Returns list of chunk dicts."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        pass