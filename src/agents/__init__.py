"""
src/agents — LangChain/LangGraph agent tools.

Public API:
    WebSearchAgent    — class-based Tavily search + content fetcher
    web_search_tool   — @tool decorated function for LangGraph ToolNode
    build_search_node — returns a LangGraph ToolNode ready to wire into a graph

Usage:
    # 1. As a plain tool in a LangGraph agent
    from src.agents import web_search_tool, build_search_node

    # 2. As a class for direct search calls
    from src.agents import WebSearchAgent
    agent = WebSearchAgent()
    results = agent.search("transformer architecture explained")
"""
from .web_search_agent import WebSearchAgent, web_search_tool, build_search_node  # noqa: F401

__all__ = ["WebSearchAgent", "web_search_tool", "build_search_node"]