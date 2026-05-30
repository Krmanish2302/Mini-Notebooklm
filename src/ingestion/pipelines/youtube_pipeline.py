"""youtube_pipeline.py — LangChain YoutubeLoader wrapper."""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader


class YoutubePipeline:
    @staticmethod
    def process(url: str, source_id: str) -> List[Document]:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source_id"]   = source_id
            doc.metadata["source_type"] = "youtube"
        return docs