"""
parent_child_creator.py

Heuristics to group fine-grained child chunks into larger, semantically coherent parent chunks
for parent-child retrieval.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def format_time(sec: float) -> str:
    """Formats seconds into MM:SS."""
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"

def _token_estimate(text: str) -> int:
    """Estimates the token count of a piece of text."""
    return int(len(text.split()) * 1.33)

def group_chunks_into_parents(
    chunks: List[Document],
    source_id: str,
    source_type: str,
) -> List[Dict[str, Any]]:
    """
    Groups child chunks into parents based on the source type and metadata.

    Returns:
        List of dicts representing parents, matching the SQLite schema:
        {
            "parent_id": str,
            "source_id": str,
            "source_type": str,
            "parent_strategy": str,
            "parent_type": str,
            "parent_text": str,
            "parent_metadata": dict,
            "range_info": str,
            "child_ids": List[str]
        }
    """
    if not chunks:
        return []

    # Sort chunks by chunk_index to ensure sequential grouping
    sorted_chunks = sorted(chunks, key=lambda c: c.metadata.get("chunk_index", 0))
    parents = []
    
    if source_type == "youtube":
        # Group adjacent child chunks in the same approximate time window (120 to 180 seconds)
        current_group = []
        group_start = None
        
        for c in sorted_chunks:
            idx = c.metadata.get("chunk_index", 0)
            c_start = c.metadata.get("start") or c.metadata.get("start_time") or (idx * 45.0)
            c_end = c.metadata.get("end") or c.metadata.get("end_time") or (c_start + 45.0)
            
            try:
                c_start = float(c_start)
                c_end = float(c_end)
            except (ValueError, TypeError):
                c_start = idx * 45.0
                c_end = c_start + 45.0
            
            c.metadata["start"] = c_start
            c.metadata["end"] = c_end
            
            if group_start is None:
                group_start = c_start
            
            # Save group if it exceeds 180 seconds or contains 5 chunks
            if current_group and (c_end - group_start > 180.0 or len(current_group) >= 5):
                child_ids = [gc.metadata.get("chunk_id") for gc in current_group if gc.metadata.get("chunk_id")]
                parent_text = "\n\n".join(gc.page_content for gc in current_group)
                min_t = group_start
                max_t = current_group[-1].metadata["end"]
                range_str = f"{format_time(min_t)}-{format_time(max_t)}"
                
                parent_idx = len(parents)
                parent_id = f"{source_id}_p{parent_idx}"
                
                parents.append({
                    "parent_id": parent_id,
                    "source_id": source_id,
                    "source_type": source_type,
                    "parent_strategy": "YouTube time-window grouping (<=180s)",
                    "parent_type": "youtube_window",
                    "parent_text": parent_text,
                    "parent_metadata": {
                        "start_time": min_t,
                        "end_time": max_t,
                        "child_count": len(current_group)
                    },
                    "range_info": range_str,
                    "child_ids": child_ids
                })
                current_group = []
                group_start = c_start
            
            current_group.append(c)
            
        if current_group:
            child_ids = [gc.metadata.get("chunk_id") for gc in current_group if gc.metadata.get("chunk_id")]
            parent_text = "\n\n".join(gc.page_content for gc in current_group)
            min_t = group_start
            max_t = current_group[-1].metadata["end"]
            range_str = f"{format_time(min_t)}-{format_time(max_t)}"
            
            parent_idx = len(parents)
            parent_id = f"{source_id}_p{parent_idx}"
            
            parents.append({
                "parent_id": parent_id,
                "source_id": source_id,
                "source_type": source_type,
                "parent_strategy": "YouTube time-window grouping (<=180s)",
                "parent_type": "youtube_window",
                "parent_text": parent_text,
                "parent_metadata": {
                    "start_time": min_t,
                    "end_time": max_t,
                    "child_count": len(current_group)
                },
                "range_info": range_str,
                "child_ids": child_ids
            })

    elif source_type == "image":
        # Strategy: Group all image chunks (caption, OCR, etc.) under a single parent record
        child_ids = [c.metadata.get("chunk_id") for c in sorted_chunks if c.metadata.get("chunk_id")]
        parent_text = "\n\n".join(f"[{c.metadata.get('child_type', 'image_part')}]\n{c.page_content}" for c in sorted_chunks)
        parents.append({
            "parent_id": f"{source_id}_p0",
            "source_id": source_id,
            "source_type": source_type,
            "parent_strategy": "Single parent image catalog",
            "parent_type": "image",
            "parent_text": parent_text,
            "parent_metadata": {
                "child_count": len(sorted_chunks)
            },
            "range_info": "Image content catalog",
            "child_ids": child_ids
        })

    elif source_type == "website":
        # Strategy: Group paragraphs under parent heading section. Split large sections (>1500 tokens)
        groups: Dict[str, List[Document]] = {}
        for c in sorted_chunks:
            heading = c.metadata.get("section_heading") or "General Content"
            groups.setdefault(heading, []).append(c)

        for heading, group_chunks in groups.items():
            current_subgroup = []
            current_tokens = 0
            subgroup_index = 1
            
            for c in group_chunks:
                c_tokens = _token_estimate(c.page_content)
                if current_subgroup and (current_tokens + c_tokens > 2500):
                    child_ids = [gc.metadata.get("chunk_id") for gc in current_subgroup if gc.metadata.get("chunk_id")]
                    parent_text = "\n\n".join(gc.page_content for gc in current_subgroup)
                    
                    parent_idx = len(parents)
                    parent_id = f"{source_id}_p{parent_idx}"
                    range_str = f"Section: {heading}"
                    if len(group_chunks) > len(current_subgroup):
                        range_str += f" (Part {subgroup_index})"

                    parents.append({
                        "parent_id": parent_id,
                        "source_id": source_id,
                        "source_type": source_type,
                        "parent_strategy": "Website section-heading grouping",
                        "parent_type": "heading_section",
                        "parent_text": parent_text,
                        "parent_metadata": {
                            "section_heading": heading,
                            "child_count": len(current_subgroup)
                        },
                        "range_info": range_str,
                        "child_ids": child_ids
                    })
                    current_subgroup = []
                    current_tokens = 0
                    subgroup_index += 1
                
                current_subgroup.append(c)
                current_tokens += c_tokens
                
            if current_subgroup:
                child_ids = [gc.metadata.get("chunk_id") for gc in current_subgroup if gc.metadata.get("chunk_id")]
                parent_text = "\n\n".join(gc.page_content for gc in current_subgroup)
                
                parent_idx = len(parents)
                parent_id = f"{source_id}_p{parent_idx}"
                range_str = f"Section: {heading}"
                if subgroup_index > 1:
                    range_str += f" (Part {subgroup_index})"

                parents.append({
                    "parent_id": parent_id,
                    "source_id": source_id,
                    "source_type": source_type,
                    "parent_strategy": "Website section-heading grouping",
                    "parent_type": "heading_section",
                    "parent_text": parent_text,
                    "parent_metadata": {
                        "section_heading": heading,
                        "child_count": len(current_subgroup)
                    },
                    "range_info": range_str,
                    "child_ids": child_ids
                })

    elif source_type == "pdf":
        has_headings = any("chapter" in c.metadata or "section_heading" in c.metadata for c in sorted_chunks)
        has_pages = any("page" in c.metadata or "page_number" in c.metadata for c in sorted_chunks)

        if has_headings:
            groups = {}
            for c in sorted_chunks:
                heading = c.metadata.get("chapter") or c.metadata.get("section_heading") or "General"
                groups.setdefault(heading, []).append(c)

            for heading, group_chunks in groups.items():
                current_subgroup = []
                current_tokens = 0
                subgroup_index = 1
                
                for c in group_chunks:
                    c_tokens = _token_estimate(c.page_content)
                    if current_subgroup and (current_tokens + c_tokens > 2500):
                        child_ids = [gc.metadata.get("chunk_id") for gc in current_subgroup if gc.metadata.get("chunk_id")]
                        parent_text = "\n\n".join(gc.page_content for gc in current_subgroup)
                        pages = sorted(list(set(gc.metadata.get("page") or gc.metadata.get("page_number") for gc in current_subgroup if gc.metadata.get("page") or gc.metadata.get("page_number"))))
                        page_range = f" (Pages {pages[0]}-{pages[-1]})" if len(pages) > 1 else (f" (Page {pages[0]})" if pages else "")
                        
                        parent_idx = len(parents)
                        parent_id = f"{source_id}_p{parent_idx}"
                        range_str = f"Section: {heading}{page_range}"
                        if len(group_chunks) > len(current_subgroup):
                            range_str += f" (Part {subgroup_index})"
                            
                        parents.append({
                            "parent_id": parent_id,
                            "source_id": source_id,
                            "source_type": source_type,
                            "parent_strategy": "Structured PDF chapter/section grouping",
                            "parent_type": "chapter_section",
                            "parent_text": parent_text,
                            "parent_metadata": {
                                "section_heading": heading,
                                "pages": pages,
                                "child_count": len(current_subgroup)
                            },
                            "range_info": range_str,
                            "child_ids": child_ids
                        })
                        current_subgroup = []
                        current_tokens = 0
                        subgroup_index += 1
                    
                    current_subgroup.append(c)
                    current_tokens += c_tokens
                    
                if current_subgroup:
                    child_ids = [gc.metadata.get("chunk_id") for gc in current_subgroup if gc.metadata.get("chunk_id")]
                    parent_text = "\n\n".join(gc.page_content for gc in current_subgroup)
                    pages = sorted(list(set(gc.metadata.get("page") or gc.metadata.get("page_number") for gc in current_subgroup if gc.metadata.get("page") or gc.metadata.get("page_number"))))
                    page_range = f" (Pages {pages[0]}-{pages[-1]})" if len(pages) > 1 else (f" (Page {pages[0]})" if pages else "")
                    
                    parent_idx = len(parents)
                    parent_id = f"{source_id}_p{parent_idx}"
                    range_str = f"Section: {heading}{page_range}"
                    if subgroup_index > 1:
                        range_str += f" (Part {subgroup_index})"
                        
                    parents.append({
                        "parent_id": parent_id,
                        "source_id": source_id,
                        "source_type": source_type,
                        "parent_strategy": "Structured PDF chapter/section grouping",
                        "parent_type": "chapter_section",
                        "parent_text": parent_text,
                        "parent_metadata": {
                            "section_heading": heading,
                            "pages": pages,
                            "child_count": len(current_subgroup)
                        },
                        "range_info": range_str,
                        "child_ids": child_ids
                    })

        elif has_pages:
            page_groups: Dict[int, List[Document]] = {}
            for c in sorted_chunks:
                p = c.metadata.get("page") or c.metadata.get("page_number") or 0
                page_groups.setdefault(p, []).append(c)

            sorted_pages = sorted(page_groups.keys())
            temp_group: List[Document] = []
            temp_tokens = 0
            
            for page in sorted_pages:
                page_chunks = page_groups[page]
                page_tokens = sum(_token_estimate(c.page_content) for c in page_chunks)
                
                if temp_group and (temp_tokens + page_tokens > 2500):
                    child_ids = [c.metadata.get("chunk_id") for c in temp_group if c.metadata.get("chunk_id")]
                    parent_text = "\n\n".join(c.page_content for c in temp_group)
                    pages = sorted(list(set(gc.metadata.get("page") or gc.metadata.get("page_number") for gc in temp_group if gc.metadata.get("page") or gc.metadata.get("page_number"))))
                    range_str = f"Pages {pages[0]}-{pages[-1]}" if len(pages) > 1 else f"Page {pages[0]}"
                    
                    parent_idx = len(parents)
                    parents.append({
                        "parent_id": f"{source_id}_p{parent_idx}",
                        "source_id": source_id,
                        "source_type": source_type,
                        "parent_strategy": "Page-group grouping (adjacent pages up to cap)",
                        "parent_type": "page_group",
                        "parent_text": parent_text,
                        "parent_metadata": {
                            "pages": pages,
                            "child_count": len(temp_group)
                        },
                        "range_info": range_str,
                        "child_ids": child_ids
                    })
                    temp_group = []
                    temp_tokens = 0
                
                temp_group.extend(page_chunks)
                temp_tokens += page_tokens

            if temp_group:
                child_ids = [c.metadata.get("chunk_id") for c in temp_group if c.metadata.get("chunk_id")]
                parent_text = "\n\n".join(c.page_content for c in temp_group)
                pages = sorted(list(set(gc.metadata.get("page") or gc.metadata.get("page_number") for gc in temp_group if gc.metadata.get("page") or gc.metadata.get("page_number"))))
                range_str = f"Pages {pages[0]}-{pages[-1]}" if len(pages) > 1 else f"Page {pages[0]}"
                parent_idx = len(parents)
                parents.append({
                    "parent_id": f"{source_id}_p{parent_idx}",
                    "source_id": source_id,
                    "source_type": source_type,
                    "parent_strategy": "Page-group grouping (adjacent pages up to cap)",
                    "parent_type": "page_group",
                    "parent_text": parent_text,
                    "parent_metadata": {
                        "pages": pages,
                        "child_count": len(temp_group)
                    },
                    "range_info": range_str,
                    "child_ids": child_ids
                })

        else:
            # Fallback to sequential grouping of 4 chunks
            max_sub = 4
            for i in range(0, len(sorted_chunks), max_sub):
                subgroup = sorted_chunks[i : i + max_sub]
                child_ids = [c.metadata.get("chunk_id") for c in subgroup if c.metadata.get("chunk_id")]
                parent_text = "\n\n".join(c.page_content for c in subgroup)
                parent_idx = len(parents)
                parent_id = f"{source_id}_p{parent_idx}"
                parents.append({
                    "parent_id": parent_id,
                    "source_id": source_id,
                    "source_type": source_type,
                    "parent_strategy": "Sequential chunk grouping (fallback)",
                    "parent_type": "sequential_group",
                    "parent_text": parent_text,
                    "parent_metadata": {
                        "child_count": len(subgroup)
                    },
                    "range_info": f"Chunks {i}-{i+len(subgroup)-1}",
                    "child_ids": child_ids
                })

    else:
        # Strategy: Sequential chunk grouping (default fallback)
        max_sub = 4
        for i in range(0, len(sorted_chunks), max_sub):
            subgroup = sorted_chunks[i : i + max_sub]
            child_ids = [c.metadata.get("chunk_id") for c in subgroup if c.metadata.get("chunk_id")]
            parent_text = "\n\n".join(c.page_content for c in subgroup)
            parent_idx = len(parents)
            parent_id = f"{source_id}_p{parent_idx}"
            parents.append({
                "parent_id": parent_id,
                "source_id": source_id,
                "source_type": source_type,
                "parent_strategy": "Sequential chunk grouping (default)",
                "parent_type": "sequential_group",
                "parent_text": parent_text,
                "parent_metadata": {
                    "child_count": len(subgroup)
                },
                "range_info": f"Chunks {i}-{i+len(subgroup)-1}",
                "child_ids": child_ids
            })

    return parents
