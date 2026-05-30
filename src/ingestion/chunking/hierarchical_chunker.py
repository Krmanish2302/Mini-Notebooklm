"""
hierarchical_chunker.py

Two-level parent/child chunking using LangChain splitters.
Returns all chunks (large + small) tagged with hierarchy level.
"""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker


class HierarchicalChunker(BaseChunker):
    def __init__(self, parent_size: int = 2000, child_size: int = 400, overlap: int = 50):
        self._parent = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=overlap)
        self._child  = RecursiveCharacterTextSplitter(chunk_size=child_size,  chunk_overlap=overlap)

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        parents = self._parent.split_documents(docs)
        result  = []
        for i, parent in enumerate(parents):
            parent.metadata["hierarchy_level"] = "parent"
            parent.metadata["parent_index"]    = i
            result.append(parent)
            children = self._child.split_documents([parent])
            for j, child in enumerate(children):
                child.metadata["hierarchy_level"] = "child"
                child.metadata["parent_index"]    = i
                child.metadata["child_index"]     = j
            result.extend(children)
        return result