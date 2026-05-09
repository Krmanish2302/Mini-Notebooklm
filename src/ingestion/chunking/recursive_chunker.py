from langchain.text_splitter import RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker
from typing import List, Dict, Any


class RecursiveChunker(BaseChunker):
    """Default chunker using LangChain's recursive splitter."""

    def __init__(self, chunk_size: int = 384, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(
        self, content: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        docs = self.splitter.create_documents([content])
        chunks = []
        for i, doc in enumerate(docs):
            chunks.append(
                {
                    "id": f"{metadata.get('source_id', 'unknown')}_chunk_{i}",
                    "content": doc.page_content,
                    "metadata": {**(metadata or {}), "chunk_index": i},
                    "modality": metadata.get("modality", "text"),
                }
            )
        return chunks

    def get_strategy_name(self) -> str:
        return "recursive"
