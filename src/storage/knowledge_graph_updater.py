"""
knowledge_graph_updater.py

Implements post-generation safe knowledge graph enrichment. Parses LLM output
to extract grounded concepts/relationships and updates SQLite.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.generation.llm_registry import LLMRegistry
from src.storage.sqlite_manager import SQLiteManager

logger = logging.getLogger(__name__)

_ENRICH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI research assistant specializing in constructing structured learning maps. "
               "Based ONLY on the retrieved source documents and the generated response, extract key technical "
               "concepts and their learning relationships (e.g. prerequisites, examples, contrasts).\n\n"
               "Valid relation types:\n"
               "- 'prerequisite_of': Concept A is required knowledge before Concept B.\n"
               "- 'contrast_with': Concept A highlights key differences or contrasts with Concept B.\n"
               "- 'example_of': Concept A is a concrete example/illustration of Concept B.\n"
               "- 'related_to': Concept A is related to Concept B in other general ways.\n\n"
               "Format your response as a strict JSON block with 'nodes' and 'edges' arrays. "
               "Ground all relationships strictly in the provided sources and generated text. Do not make up facts.\n"
               "JSON Format:\n"
               "{{\n"
               "  \"nodes\": [\n"
               "    {{\"name\": \"Concept Name\", \"type\": \"concept\", \"description\": \"brief explanation\"}}\n"
               "  ],\n"
               "  \"edges\": [\n"
               "    {{\"source\": \"Concept A\", \"target\": \"Concept B\", \"relation\": \"prerequisite_of\", \"confidence\": 0.9, \"provenance\": {{\"source_id\": \"doc_id\", \"page\": \"page_number\"}}}}\n"
               "  ]]\n"
               "}}\n"
               "Output ONLY valid JSON inside markdown code fence ```json ... ```. No extra text."),
    ("human", "Retrieved Sources:\n{sources_text}\n\nGenerated Response:\n{answer}\n\nJSON Output:")
])

class KnowledgeGraphUpdater:
    def __init__(self, db: SQLiteManager | None = None):
        self.db = db or SQLiteManager()

    def enrich_graph(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Parses evidence to extract grounded concepts and learning relations, 
        persisting them to SQLite.
        """
        if not retrieved_docs or not answer:
            return {"nodes_added": 0, "edges_added": 0}

        # Format sources text for the LLM
        sources_text_list = []
        source_id_set = set()
        for idx, doc in enumerate(retrieved_docs):
            sid = doc.metadata.get("source_id", "unknown")
            source_id_set.add(sid)
            page = doc.metadata.get("page", "")
            page_str = f" p.{page}" if page else ""
            sources_text_list.append(f"[{sid}{page_str}]: {doc.page_content.strip()}")

        sources_text = "\n\n".join(sources_text_list)

        try:
            llm = LLMRegistry.get()
            chain = _ENRICH_PROMPT | llm | StrOutputParser()
            raw_output = chain.invoke({
                "sources_text": sources_text,
                "answer": answer
            })

            # Extract JSON block
            json_match = re.search(r"```json\s*(.*?)\s*```", raw_output, re.DOTALL)
            json_str = json_match.group(1) if json_match else raw_output.strip()
            
            # Clean up potentially loose JSON starts
            if not json_str.startswith("{") and "{" in json_str:
                json_str = json_str[json_str.find("{"):]
            if not json_str.endswith("}") and "}" in json_str:
                json_str = json_str[:json_str.rfind("}")+1]

            data = json.loads(json_str)
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])

            nodes_added = 0
            edges_added = 0

            # ── 1. Save Nodes ────────────────────────────────────────────────
            for node in nodes:
                name = node.get("name", "").strip()
                if not name:
                    continue
                node_id = name.lower().replace(" ", "_")
                node_type = node.get("type", "concept")
                desc = node.get("description", "")
                
                self.db.add_graph_node(
                    node_id=node_id,
                    name=name,
                    node_type=node_type,
                    metadata={"description": desc}
                )
                nodes_added += 1

            # ── 2. Save Edges ────────────────────────────────────────────────
            for edge in edges:
                src_name = edge.get("source", "").strip()
                tgt_name = edge.get("target", "").strip()
                relation = edge.get("relation", "related_to").strip().lower()
                conf = float(edge.get("confidence", 1.0))
                prov = edge.get("provenance", {})

                if not src_name or not tgt_name:
                    continue

                # Ensure source and target node IDs exist in the DB
                src_id = src_name.lower().replace(" ", "_")
                tgt_id = tgt_name.lower().replace(" ", "_")

                # Guarantee nodes exist in DB
                self.db.add_graph_node(src_id, src_name, "concept")
                self.db.add_graph_node(tgt_id, tgt_name, "concept")

                # Validate provenance: only allow grounded source_ids
                prov_source = prov.get("source_id", "").strip()
                if prov_source and prov_source not in source_id_set:
                    # Look for soft matches (e.g. filename substring matching)
                    matched_sid = None
                    for sid in source_id_set:
                        if prov_source in sid or sid in prov_source:
                            matched_sid = sid
                            break
                    prov["source_id"] = matched_sid if matched_sid else list(source_id_set)[0]

                self.db.add_graph_edge(
                    source_node=src_id,
                    target_node=tgt_id,
                    relation=relation,
                    provenance=prov,
                    confidence=conf
                )
                edges_added += 1

            logger.info(
                "[KnowledgeGraphUpdater] Enriched graph: %d nodes, %d edges added/updated", 
                nodes_added, edges_added
            )
            return {"nodes_added": nodes_added, "edges_added": edges_added}

        except Exception as e:
            logger.warning("[KnowledgeGraphUpdater] Failed to enrich graph: %s", e)
            return {"nodes_added": 0, "edges_added": 0, "error": str(e)}
