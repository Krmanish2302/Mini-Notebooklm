"""pdf_cleaner.py — Clean extracted PDF text in LangChain Documents."""
from __future__ import annotations
import re
from typing import List
from langchain_core.documents import Document


class PDFCleaner:
    def clean(self, docs: List[Document]) -> List[Document]:
        result = []
        for doc in docs:
            text = doc.page_content
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]{2,}", " ", text)
            text = re.sub(r"(?i)\bpage\s*\d+\b", "", text)
            text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
            text = text.strip()
            if len(text.split()) >= 10:
                result.append(Document(page_content=text, metadata=doc.metadata))
        return result