"""youtube_cleaner.py — Clean YouTube transcript LangChain Documents."""
from __future__ import annotations
import re
from typing import List
from langchain_core.documents import Document


class YoutubeCleaner:
    def clean(self, docs: List[Document]) -> List[Document]:
        result = []
        for doc in docs:
            text = doc.page_content
            text = re.sub(r"\[.*?\]", "", text)           # remove [Music], [Applause]
            text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)  # remove repeated words
            text = re.sub(r"\s{2,}", " ", text)
            text = text.strip()
            if len(text.split()) >= 10:
                result.append(Document(page_content=text, metadata=doc.metadata))
        return result