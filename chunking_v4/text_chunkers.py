#!/usr/bin/env python3
"""
Text chunking strategies for v4 chunks.

This module provides a v2-compatible text chunker that reuses the existing
CodeChunker logic from src/Chunker.py. It only handles text splitting - all
v4 metadata (symbols, roles, byte ranges, etc.) is preserved unchanged.
"""

import os
import sys
from typing import List, Dict, Any

# Add src/ to path to import CodeChunker
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from Chunker import CodeChunker


def chunk_code_text_v2(
    source_text: str,
    language: str = "",
    file_extension: str = "",
    max_chars: int = 12000,
    max_lines: int = 400,
    token_limit: int = 1200,
    encoding_name: str = "gpt-4",
) -> List[Dict[str, Any]]:
    """
    Returns a list of text chunks using v2's CodeChunker strategy.
    
    This replicates v2's chunking behavior (function/method boundary awareness,
    Tree-sitter based breakpoints) without any metadata - just the text splits.
    
    Args:
        source_text: Full file content
        language: Language name (e.g., "go", "python")
        file_extension: File extension (e.g., "go", "py")
        max_chars: Maximum characters per chunk (not strictly enforced if single function)
        max_lines: Maximum lines per chunk (not strictly enforced)
        token_limit: Token limit for CodeChunker
        encoding_name: Encoding for token counting
    
    Returns:
        List of dicts with keys:
          - start_line (1-based, inclusive)
          - end_line (1-based, inclusive)
          - text (the code snippet)
    
    Note:
        The CodeChunker uses Tree-sitter to find function/class boundaries and
        creates chunks at those breakpoints. This ensures clean semantic breaks.
        Single large functions can exceed limits (acceptable for completeness).
    """
    # Determine file extension
    if not file_extension and language:
        file_extension = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "go": "go",
            "ruby": "rb",
            "php": "php",
            "java": "java",
        }.get(language.lower(), language.lower())
    
    if not file_extension:
        file_extension = "txt"  # Fallback
    
    # Create chunker
    chunker = CodeChunker(file_extension=file_extension, encoding_name=encoding_name)
    
    # Chunk the code
    try:
        chunks_dict = chunker.chunk(source_text, token_limit=token_limit)
    except Exception as e:
        # Fallback: return entire file as one chunk if chunking fails
        lines = source_text.split("\n")
        return [{
            "start_line": 1,
            "end_line": len(lines),
            "text": source_text,
        }]
    
    # Convert to list format
    result = []
    for chunk_idx in sorted(chunks_dict.keys()):
        chunk = chunks_dict[chunk_idx]
        if isinstance(chunk, dict):
            result.append({
                "start_line": chunk.get("start_line", 1),
                "end_line": chunk.get("end_line", chunk.get("start_line", 1)),
                "text": chunk.get("text", ""),
            })
        else:
            # Legacy string format (shouldn't happen with current Chunker)
            result.append({
                "start_line": 1,
                "end_line": len(source_text.split("\n")),
                "text": str(chunk),
            })
    
    return result


def map_v2_chunk_to_v4_span(
    v2_chunks: List[Dict[str, Any]],
    v4_start_line: int,
    v4_end_line: int,
    source_lines: List[str]
) -> str:
    """
    Map a v4 chunk's line range to the best matching v2 chunk text.
    
    Strategy:
    1. Find v2 chunk(s) that overlap with v4's [start_line, end_line]
    2. Prefer the v2 chunk that covers the v4 midpoint
    3. If multiple overlaps, pick the one with most overlap
    4. Clip the text to exactly match v4's line range
    
    Args:
        v2_chunks: List of v2 text chunks from chunk_code_text_v2()
        v4_start_line: V4 chunk start line (1-based, inclusive)
        v4_end_line: V4 chunk end line (1-based, inclusive)
        source_lines: Full file lines (for clipping)
    
    Returns:
        Text clipped to v4's line range
    """
    if not v2_chunks:
        # Fallback: extract directly from source
        return "\n".join(source_lines[v4_start_line - 1:v4_end_line])
    
    # Find v2 chunks that overlap with v4 range
    v4_midpoint = (v4_start_line + v4_end_line) / 2
    overlapping = []
    
    for v2_chunk in v2_chunks:
        v2_start = v2_chunk["start_line"]
        v2_end = v2_chunk["end_line"]
        
        # Check for overlap
        if not (v2_end < v4_start_line or v2_start > v4_end_line):
            # Calculate overlap amount
            overlap_start = max(v2_start, v4_start_line)
            overlap_end = min(v2_end, v4_end_line)
            overlap_lines = overlap_end - overlap_start + 1
            
            # Calculate distance from midpoint
            v2_midpoint = (v2_start + v2_end) / 2
            midpoint_distance = abs(v2_midpoint - v4_midpoint)
            
            overlapping.append({
                "chunk": v2_chunk,
                "overlap_lines": overlap_lines,
                "midpoint_distance": midpoint_distance,
            })
    
    if not overlapping:
        # No v2 chunks overlap - fallback to direct extraction
        return "\n".join(source_lines[v4_start_line - 1:v4_end_line])
    
    # Sort by overlap (descending) and then by midpoint distance (ascending)
    overlapping.sort(key=lambda x: (-x["overlap_lines"], x["midpoint_distance"]))
    best_match = overlapping[0]["chunk"]
    
    # Clip to v4's exact range
    # Extract lines from the best matching v2 chunk, clipped to v4 range
    v2_start = best_match["start_line"]
    v2_end = best_match["end_line"]
    v2_text = best_match["text"]
    v2_lines = v2_text.split("\n")
    
    # Calculate offset within v2 chunk
    if v4_start_line >= v2_start and v4_end_line <= v2_end:
        # V4 range is fully within v2 chunk - extract substring
        offset_start = v4_start_line - v2_start
        offset_end = v4_end_line - v2_start + 1
        return "\n".join(v2_lines[offset_start:offset_end])
    else:
        # V4 range extends beyond v2 chunk - clip from source
        return "\n".join(source_lines[v4_start_line - 1:v4_end_line])

