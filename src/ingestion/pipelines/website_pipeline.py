"""website_pipeline.py — LangChain WebBaseLoader wrapper."""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader


class WebsitePipeline:
    @staticmethod
    def process(url: str, source_id: str) -> List[Document]:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        docs = WebBaseLoader(url, requests_kwargs={"headers": headers}).load()
        for doc in docs:
            doc.metadata["source_id"]   = source_id
            doc.metadata["source_type"] = "website"
        return docs