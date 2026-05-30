"""
graph_history.py — Study Mode session history: concept-graph tracking.

LangChain upgrade notes
-----------------------
* Messages  : LangChain ChatMessageHistory (HumanMessage / AIMessage)
              in sync with the existing message list.
* KG writes : unchanged — concept nodes + 'led_to' / 'followed_by' / 'mentions' edges.
* format_for_prompt() : returns a LangChain-compatible history string AND
  exposes lc_messages for RunnableWithMessageHistory.

Class name
----------
Original had a mismatch: __init__.py exported 'GraphHistory',
chat_history_manager.py imported 'GraphChatHistory'.
Canonical name is GraphHistory throughout (matching __init__.py).
GraphChatHistory is kept as an alias for backward compat.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class GraphHistory:
    """
    Study Mode session history: concept-node tracking + KG enrichment.

    Parameters
    ----------
    session_id      : str
    knowledge_graph : KnowledgeGraph  (duck-typed — must implement add_chunk,
                                       add_edge, get_neighbors, .graph)
    """

    def __init__(self, session_id: str, knowledge_graph: Any) -> None:
        self.session_id  = session_id
        self.kg          = knowledge_graph
        self.messages:   List[Dict[str, Any]] = []
        self._visited:   Dict[str, Dict[str, Any]] = {}
        self._lc_history = ChatMessageHistory()

    # ── Concept tracking ──────────────────────────────────────────────────────

    def has_visited(self, concept: str) -> bool:
        """Return True if this concept was already covered in the session."""
        return self._concept_key(concept) in self._visited

    def mark_visited(self, concept: str, triggering_query: str = "") -> None:
        """
        Mark concept as visited and add a node in the KnowledgeGraph.
        Links to the previous concept node with a 'led_to' edge.
        """
        key = self._concept_key(concept)
        if key in self._visited:
            return
        record: Dict[str, Any] = {
            "concept":    concept,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "query":      triggering_query,
        }
        self._visited[key] = record

        concept_node_id = f"concept__{self.session_id}__{key}"
        self.kg.add_chunk({
            "id":       concept_node_id,
            "content":  concept,
            "modality": "study_concept",
            "source_id": self.session_id,
            "metadata": record,
        })
        prev = self._last_concept_node_id()
        if prev and prev != concept_node_id:
            self.kg.add_edge(prev, concept_node_id, weight=0.9, relation="led_to")
        logger.debug("[GraphHistory] Marked visited: %s", concept)

    def get_visited_concepts(self) -> List[str]:
        return [v["concept"] for v in self._visited.values()]

    # ── Message tracking ──────────────────────────────────────────────────────

    def add_message(
        self,
        role:         str,
        content:      str,
        concepts:     Optional[List[str]] = None,
        sources_used: Optional[List[str]] = None,
    ) -> None:
        """
        Record a message, update KG, and sync LangChain ChatMessageHistory.
        """
        message_id = f"{self.session_id}_msg_{len(self.messages)}"
        msg: Dict[str, Any] = {
            "id":           message_id,
            "role":         role,
            "content":      content,
            "concepts":     concepts or [],
            "sources_used": sources_used or [],
            "index":        len(self.messages),
        }
        self.messages.append(msg)

        # LangChain message history sync
        if role.lower() in ("user", "human"):
            self._lc_history.add_user_message(content)
        else:
            self._lc_history.add_ai_message(content)

        # KG: message node
        self.kg.add_chunk({
            "id":       message_id,
            "content":  content,
            "modality": "chat_message",
            "source_id": self.session_id,
            "metadata": {"role": role, "concepts": concepts or []},
        })
        if len(self.messages) > 1:
            prev_msg_id = self.messages[-2]["id"]
            self.kg.add_edge(prev_msg_id, message_id, weight=1.0, relation="followed_by")

        # KG: concept → message edges
        for concept in (concepts or []):
            self.mark_visited(concept, triggering_query=content)
            cnode_id = f"concept__{self.session_id}__{self._concept_key(concept)}"
            if cnode_id in self.kg.graph:
                self.kg.add_edge(message_id, cnode_id, weight=0.8, relation="mentions")

    # ── Learning path ─────────────────────────────────────────────────────────

    def get_learning_path(self, concept: str) -> List[Dict[str, Any]]:
        key            = self._concept_key(concept)
        concept_node_id = f"concept__{self.session_id}__{key}"
        if concept_node_id not in self.kg.graph:
            return []
        return self.kg.get_neighbors(concept_node_id, depth=2)

    def get_concept_connections(self) -> List[Dict[str, Any]]:
        """Return visited concept records — used by ChatHistoryManager."""
        return list(self._visited.values())

    def concept_graph_for_ui(self) -> Dict[str, Any]:
        """
        Returns {nodes, edges} JSON for the frontend concept-map component.
        Only surfaces 'led_to' / 'prerequisite_of' / 'causes' / 'is_a_type_of' edges.
        """
        nodes: List[Dict] = []
        edges: List[Dict] = []
        prefix = f"concept__{self.session_id}__"
        for node_id, data in self.kg.graph.nodes(data=True):
            if (
                data.get("modality") == "study_concept"
                and node_id.startswith(prefix)
            ):
                nodes.append({
                    "id":         node_id,
                    "label":      data.get("content", node_id),
                    "type":       "concept",
                    "first_seen": data.get("metadata", {}).get("first_seen"),
                })
        for u, v, edata in self.kg.graph.edges(data=True):
            rel = edata.get("relation", "related")
            if rel in ("led_to", "prerequisite_of", "causes", "is_a_type_of"):
                edges.append({"from": u, "to": v, "relation": rel})
        return {"nodes": nodes, "edges": edges}

    # ── LangChain interop ─────────────────────────────────────────────────────

    @property
    def lc_messages(self) -> List[BaseMessage]:
        """LangChain BaseMessage list — for RunnableWithMessageHistory."""
        return self._lc_history.messages

    def format_for_prompt(self) -> str:
        """
        Compact history string for the LLM prompt.
        Lists previously discussed concepts + last 2 messages.
        """
        parts: List[str] = []
        if self._visited:
            concepts_str = ", ".join(self.get_visited_concepts())
            parts.append(f"Previously discussed concepts: {concepts_str}")
        for msg in self.messages[-2:]:
            parts.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(parts)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self.messages = []
        self._visited = {}
        self._lc_history.clear()
        logger.debug("[GraphHistory] Cleared session=%s", self.session_id)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _concept_key(concept: str) -> str:
        return concept.strip().lower().replace(" ", "_")

    def _last_concept_node_id(self) -> Optional[str]:
        if not self._visited:
            return None
        last_key = list(self._visited.keys())[-1]
        return f"concept__{self.session_id}__{last_key}"


# Backward-compat alias (chat_history_manager.py imported this name)
GraphChatHistory = GraphHistory