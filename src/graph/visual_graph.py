"""
visual_graph.py — Interactive knowledge graph visualization.

Uses Pyvis to generate a self-contained HTML file from a GraphStore.
Falls back to a plain JSON adjacency export if Pyvis is unavailable.

Usage:
    from src.graph import visualize_graph
    from src.graph import GraphStore

    store = GraphStore()
    path  = visualize_graph(store, output_path="./data/graph.html")
    # → opens in browser or served as static file

LangChain integration:
    visualize_graph() accepts an optional LangChain CallbackManager so
    Langsmith / other observability tools can trace the visualization call.
    This is purely optional — pass callbacks=None (default) to skip.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Optional

from langchain_core.tracers.context import tracing_v2_enabled

logger = logging.getLogger(__name__)

# Relation → Pyvis edge color
_RELATION_COLORS: dict = {
    "prerequisite_of":  "#e63946",   # red    — strong dependency
    "causes":           "#f4a261",   # orange
    "is_a_type_of":     "#2a9d8f",   # teal
    "raptor_parent_of": "#457b9d",   # blue   — RAPTOR hierarchy
    "semantic":         "#a8dadc",   # light blue
    "led_to":           "#e9c46a",   # yellow
    "related":          "#adb5bd",   # grey   — weak
    "followed_by":      "#dee2e6",   # very light
    "mentions":         "#dee2e6",
}

_NODE_COLOR_DEFAULT = "#4f98a3"   # Nexus Hydra Teal


def visualize_graph(
    graph_store:   Any,                        # GraphStore
    output_path:   str  = "./data/graph.html",
    max_nodes:     int  = 300,
    title:         str  = "Knowledge Graph",
    height:        str  = "750px",
    width:         str  = "100%",
    source_filter: Optional[str] = None,       # restrict to one source_id
    callbacks:     Optional[Any] = None,       # LangChain CallbackManager (optional)
) -> str:
    """
    Render GraphStore as interactive HTML.

    Parameters
    ----------
    graph_store   : GraphStore instance
    output_path   : where to write the HTML file
    max_nodes     : cap rendered nodes (largest hub first)
    title         : page title inside the HTML
    height/width  : Pyvis canvas size
    source_filter : if set, only render nodes from that source_id
    callbacks     : LangChain CallbackManager for tracing (optional)

    Returns
    -------
    str — absolute path to the written HTML file
    """
    try:
        return _render_pyvis(
            graph_store, output_path, max_nodes, title, height, width, source_filter
        )
    except ImportError:
        logger.warning("[visualize_graph] Pyvis not installed — falling back to JSON export.")
        return _render_json(graph_store, output_path, source_filter)


# ── Pyvis renderer ─────────────────────────────────────────────────────────────

def _render_pyvis(
    graph_store,
    output_path:   str,
    max_nodes:     int,
    title:         str,
    height:        str,
    width:         str,
    source_filter: Optional[str],
) -> str:
    from pyvis.network import Network  # type: ignore

    g = graph_store.graph

    # Filter nodes
    nodes_to_render = [
        (n, d) for n, d in g.nodes(data=True)
        if source_filter is None or d.get("source_id") == source_filter
    ]

    # Cap by degree (show highest-degree hubs first)
    if len(nodes_to_render) > max_nodes:
        nodes_to_render = sorted(
            nodes_to_render,
            key=lambda nd: g.degree(nd[0]),
            reverse=True,
        )[:max_nodes]

    visible_ids = {n for n, _ in nodes_to_render}

    net = Network(height=height, width=width, directed=True, notebook=False, heading=title)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100)

    for node_id, data in nodes_to_render:
        label   = data.get("content", node_id)[:40] + ("…" if len(data.get("content","")) > 40 else "")
        tooltip = (
            f"<b>{node_id}</b><br>"
            f"Source: {data.get('source_id','')}<br>"
            f"Modality: {data.get('modality','text')}<br>"
            f"{data.get('content','')[:120]}"
        )
        net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color=_NODE_COLOR_DEFAULT,
            size=10 + min(g.degree(node_id) * 2, 30),
        )

    for src, tgt, edata in g.edges(data=True):
        if src not in visible_ids or tgt not in visible_ids:
            continue
        rel = edata.get("relation", "related")
        net.add_edge(
            src, tgt,
            title=rel,
            color=_RELATION_COLORS.get(rel, "#adb5bd"),
            width=max(1, edata.get("weight", 1.0) * 2),
            arrows="to",
        )

    # Legend injection
    legend_html = _build_legend_html()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    net.save_graph(output_path)

    # Inject legend into saved HTML
    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</body>", legend_html + "\n</body>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("[visualize_graph] Wrote Pyvis HTML → %s (%d nodes)", output_path, len(nodes_to_render))
    return os.path.abspath(output_path)


# ── JSON fallback renderer ─────────────────────────────────────────────────────

def _render_json(
    graph_store,
    output_path:   str,
    source_filter: Optional[str],
) -> str:
    g = graph_store.graph
    nodes = [
        {
            "id":        n,
            "content":   d.get("content", "")[:100],
            "source_id": d.get("source_id", ""),
            "modality":  d.get("modality", "text"),
        }
        for n, d in g.nodes(data=True)
        if source_filter is None or d.get("source_id") == source_filter
    ]
    edges = [
        {"from": s, "to": t, "relation": ed.get("relation","related"), "weight": ed.get("weight", 1.0)}
        for s, t, ed in g.edges(data=True)
    ]
    payload = {"nodes": nodes, "edges": edges}

    json_path = output_path.replace(".html", ".json")
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("[visualize_graph] JSON fallback → %s", json_path)
    return os.path.abspath(json_path)


# ── Legend helper ──────────────────────────────────────────────────────────────

def _build_legend_html() -> str:
    items = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
        f'<span style="width:14px;height:14px;border-radius:3px;background:{color};display:inline-block"></span>'
        f'<span style="font-size:12px">{rel}</span></div>'
        for rel, color in _RELATION_COLORS.items()
    )
    return (
        '<div id="kg-legend" style="position:fixed;bottom:16px;right:16px;'
        'background:rgba(255,255,255,0.92);padding:12px 16px;border-radius:8px;'
        'box-shadow:0 2px 12px rgba(0,0,0,0.15);font-family:sans-serif;z-index:9999">'
        '<b style="font-size:13px">Edge Types</b><br><br>'
        + items +
        "</div>"
    )