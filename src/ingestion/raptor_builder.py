"""
raptor_builder.py

RAPTOR (Recursive Abstractive Processing for Tree-Organised Retrieval)
builds a multi-level summary tree on top of the base chunk layer.

How it works
------------
1. Take the leaf-level Document chunks produced by the ingestion pipeline.
2. Group them into batches of RAPTOR_BATCH_SIZE.
3. Summarise each batch with an LLM → create parent-level Documents.
4. Repeat until only one root Document remains.
5. Add ALL levels (leaves + parent summaries) to the FAISS vectorstore
   so both fine-grained and abstract queries are answered well.

Integration
-----------
Call build_raptor_tree() AFTER embed_and_index node, passing the
FAISS vectorstore path and the leaf chunks from state:

    from src.ingestion.raptor_builder import build_raptor_tree

    build_raptor_tree(
        chunks=state["chunks"],
        vectorstore_path=state["vectorstore_path"],
        source_id=state["source_id"],
    )
"""
from __future__ import annotations

import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts   import ChatPromptTemplate
from langchain_openai          import ChatOpenAI

logger = logging.getLogger(__name__)

RAPTOR_BATCH_SIZE  = int(os.getenv("RAPTOR_BATCH_SIZE",  "10"))
RAPTOR_MAX_LEVELS  = int(os.getenv("RAPTOR_MAX_LEVELS",  "3"))
SUMMARY_MODEL      = os.getenv("RAPTOR_SUMMARY_MODEL",   "gpt-4o-mini")

_SUMMARISE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert summariser. Produce a concise, dense summary of the "
     "following text passages that preserves all key facts and concepts."),
    ("human", "{text}"),
])


def _summarise_batch(texts: List[str], llm: ChatOpenAI) -> str:
    chain    = _SUMMARISE_PROMPT | llm
    combined = "\n\n---\n\n".join(texts)
    response = chain.invoke({"text": combined})
    return response.content


def build_raptor_tree(
    chunks: List[Document],
    vectorstore_path: str,
    source_id: str,
) -> None:
    """
    Build RAPTOR summary tree and add all levels to the FAISS index.

    Args:
        chunks:           Leaf-level chunks from the ingestion pipeline.
        vectorstore_path: Directory where the FAISS index is saved.
        source_id:        Source identifier.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore  = FAISS.load_local(
        vectorstore_path, embeddings, allow_dangerous_deserialization=True
    )
    llm          = ChatOpenAI(model=SUMMARY_MODEL, temperature=0)

    current_level: List[Document] = chunks
    all_summary_docs: List[Document] = []

    for level in range(1, RAPTOR_MAX_LEVELS + 1):
        if len(current_level) <= 1:
            logger.info("[raptor] Level %d: only 1 doc, stopping tree.", level)
            break

        next_level: List[Document] = []
        batches = [
            current_level[i : i + RAPTOR_BATCH_SIZE]
            for i in range(0, len(current_level), RAPTOR_BATCH_SIZE)
        ]

        for batch_idx, batch in enumerate(batches):
            texts   = [d.page_content for d in batch]
            summary = _summarise_batch(texts, llm)
            summary_doc = Document(
                page_content=summary,
                metadata={
                    "source_id":    source_id,
                    "raptor_level": level,
                    "batch_index":  batch_idx,
                    "source_type":  "raptor_summary",
                },
            )
            next_level.append(summary_doc)
            all_summary_docs.append(summary_doc)

        logger.info(
            "[raptor] Level %d: %d batches → %d summaries",
            level, len(batches), len(next_level),
        )
        current_level = next_level

    if all_summary_docs:
        vectorstore.add_documents(all_summary_docs)
        vectorstore.save_local(vectorstore_path)
        logger.info(
            "[raptor] Added %d summary docs across all levels to '%s'",
            len(all_summary_docs), vectorstore_path,
        )
    else:
        logger.info("[raptor] No summary docs generated — skipping update.")
