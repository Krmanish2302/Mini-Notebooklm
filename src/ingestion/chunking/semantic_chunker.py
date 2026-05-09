from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from .base_chunker import BaseChunker
from typing import List, Dict, Any

class SemanticChunker(BaseChunker):
    """Chunks based on semantic similarity using embeddings."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 breakpoint_threshold: float = 0.85):
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.splitter = LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold
        )
    
    def chunk(self, content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        docs = self.splitter.create_documents([content])
        chunks = []
        for i, doc in enumerate(docs):
            chunks.append({
                "id": f"{metadata.get('source_id', 'unknown')}_chunk_{i}",
                "content": doc.page_content,
                "metadata": {**(metadata or {}), "chunk_index": i},
                "modality": metadata.get("modality", "text")
            })
        return chunks
    
    def get_strategy_name(self) -> str:
        return "semantic"