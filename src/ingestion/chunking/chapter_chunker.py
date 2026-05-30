"""
chapter_chunker.py

Split documents on Markdown heading boundaries (# / ## / ###).
Falls back to RecursiveCharacterTextSplitter if no headings found.
"""
from __future__ import annotations
import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker


class ChapterChunker(BaseChunker):
    _HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    _FALLBACK = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self._HEADERS)
        result   = []
        for doc in docs:
            if re.search(r"^#{1,3}\s", doc.page_content, re.MULTILINE):
                splits = splitter.split_text(doc.page_content)
                for s in splits:
                    s.metadata = {**doc.metadata, **s.metadata, "chunker": "chapter"}
                result.extend(splits)
            else:
                fallback = self._FALLBACK.split_documents([doc])
                for c in fallback:
                    c.metadata["chunker"] = "chapter_fallback"
                result.extend(fallback)
        return result