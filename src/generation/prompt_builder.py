from typing import List, Dict, Any

class PromptBuilder:
    """Builds mode-specific prompts with citations and grounding."""
    
    GROUNDING_INSTRUCTION = """
CRITICAL INSTRUCTIONS:
- Answer using ONLY the provided sources below
- Never make up information not in the sources
- If the answer is not in the sources, say "I don't have enough information"
- Always cite sources using [SOURCE_X] format
- Be concise but complete
"""
    
    @staticmethod
    def build_chat_prompt(query: str, context: str, history: str = "") -> str:
        """Build prompt for Chat mode."""
        prompt = f"""{PromptBuilder.GROUNDING_INSTRUCTION}

CONVERSATION HISTORY:
{history}

RELEVANT SOURCES:
{context}

USER QUESTION: {query}

ANSWER (with citations):"""
        return prompt
    
    @staticmethod
    def build_deep_research_prompt(query: str, context: str, history: str = "") -> str:
        """Build prompt for Deep Research mode."""
        prompt = f"""{PromptBuilder.GROUNDING_INSTRUCTION}

You are in Deep Research mode. Provide a comprehensive, well-structured answer.

CONVERSATION HISTORY:
{history}

RELEVANT SOURCES (ranked by relevance):
{context}

USER QUESTION: {query}

DETAILED ANSWER (with citations and analysis):"""
        return prompt
    
    @staticmethod
    def build_study_mode_prompt(query: str, context: str, learning_path: List[Dict], history: str = "") -> str:
        """Build prompt for Study Mode with learning path."""
        path_str = "\n".join([
            f"{i+1}. {step['from']} → {step['to']} ({step['relationship']})"
            for i, step in enumerate(learning_path[:3])
        ])
        
        prompt = f"""{PromptBuilder.GROUNDING_INSTRUCTION}

You are in Study Mode. Explain concepts clearly and show relationships.

LEARNING PATH:
{path_str}

CONVERSATION HISTORY:
{history}

RELEVANT SOURCES:
{context}

USER QUESTION: {query}

EDUCATIONAL ANSWER (with citations and concept connections):"""
        return prompt
    
    @staticmethod
    def format_context(chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into context string with source labels."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_label = f"[SOURCE_{i}]"
            content = chunk["content"][:500]  # Limit length
            context_parts.append(f"{source_label} {content}")
        return "\n\n".join(context_parts)