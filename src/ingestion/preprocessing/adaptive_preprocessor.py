"""
adaptive_preprocessor.py

Routes each Document to the correct source cleaner based on source_type,
then returns cleaned LangChain Documents.
"""
from __future__ import annotations
import re
from typing import List
from langchain_core.documents import Document


class AdaptivePreprocessor:
    def preprocess(self, docs: List[Document]) -> List[Document]:
        result = []
        for doc in docs:
            stype = doc.metadata.get("source_type", "text")
            if stype == "pdf":
                content = self._clean_pdf(doc.page_content)
            elif stype == "website":
                content = self._clean_website(doc.page_content)
            else:
                content = doc.page_content.strip()

            if len(content.split()) >= 10:
                result.append(Document(
                    page_content=content,
                    metadata={**doc.metadata, "preprocessed": True},
                ))
        return result

    @staticmethod
    def _clean_pdf(text: str) -> str:
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"(?i)\bpage\s*\d+\b", "", text)
        return text.strip()

    @staticmethod
    def _clean_website(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()