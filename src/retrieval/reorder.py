"""
reorder.py — lost-in-the-middle reordering utility.
Rearranges retrieved chunks to put the highest relevance chunk first,
second-highest last, and the rest in the middle, preserving the original order.
"""
from __future__ import annotations
from typing import Any, List, Tuple


def reorder_chunks(chunks_with_scores: List[Tuple[Any, float]]) -> List[Any]:
    """
    Reorders source chunks to mitigate lost-in-the-middle performance degradation in LLMs.
    
    Args:
        chunks_with_scores: A list of tuples containing (chunk_object, score).
        
    Returns:
        A list of chunk objects (without scores), reordered such that:
        - < 2 chunks: returned in original order.
        - == 2 chunks: highest score first, second-highest second.
        - 3 or 4 chunks: highest first, second-highest last, rest in the middle.
        - >= 5 chunks: highest at index 0, second-highest at the last index,
          and all remaining chunks in their original order in between.
    """
    if len(chunks_with_scores) < 2:
        return [chunk for chunk, _ in chunks_with_scores]

    if len(chunks_with_scores) == 2:
        # Sort by score descending
        sorted_chunks = sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in sorted_chunks]

    n = len(chunks_with_scores)
    
    # Identify the highest and second-highest scores.
    # We do a stable sort of indices by score descending.
    sorted_indices = sorted(range(n), key=lambda i: chunks_with_scores[i][1], reverse=True)
    highest_idx = sorted_indices[0]
    second_highest_idx = sorted_indices[1]

    highest_chunk = chunks_with_scores[highest_idx][0]
    second_highest_chunk = chunks_with_scores[second_highest_idx][0]

    # Filter out the highest and second-highest chunks from the remaining,
    # keeping the others in their original relative order.
    remaining_chunks = [
        chunk for idx, (chunk, _) in enumerate(chunks_with_scores)
        if idx != highest_idx and idx != second_highest_idx
    ]

    return [highest_chunk] + remaining_chunks + [second_highest_chunk]
