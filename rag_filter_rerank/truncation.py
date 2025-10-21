"""
Text truncation helpers with markdown fence safety.

Ensures truncated text doesn't break code blocks or contain partial UTF-8.
"""
import re
from typing import Optional
from .schemas import Chunk


def safe_truncate_text(text: str, max_chars: int) -> str:
    """
    Truncate text safely without breaking UTF-8 or words.
    
    Args:
        text: Text to truncate
        max_chars: Maximum characters
    
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    
    # Truncate at character boundary
    truncated = text[:max_chars]
    
    # Try to break at last newline
    last_newline = truncated.rfind('\n')
    if last_newline > max_chars * 0.8:  # If newline is in last 20%
        truncated = truncated[:last_newline]
    else:
        # Try to break at last space
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:
            truncated = truncated[:last_space]
    
    # Ensure valid UTF-8 by encoding/decoding
    try:
        truncated = truncated.encode('utf-8', errors='ignore').decode('utf-8')
    except:
        pass
    
    return truncated.rstrip()


def close_open_fences(text: str) -> str:
    """
    Close any open markdown code fences in text.
    
    Args:
        text: Text that may have open fences
    
    Returns:
        Text with fences closed
    """
    # Count triple backticks
    fence_count = text.count('```')
    
    # If odd number, close the fence
    if fence_count % 2 == 1:
        text = text.rstrip() + '\n```'
    
    return text


def extract_relevant_lines(text: str, max_lines: int = 40) -> str:
    """
    Extract most relevant lines from code (first N lines around primary content).
    
    Args:
        text: Full code text
        max_lines: Maximum lines to keep
    
    Returns:
        Extracted lines
    """
    lines = text.split('\n')
    
    if len(lines) <= max_lines:
        return text
    
    # Keep first max_lines, add ellipsis
    kept_lines = lines[:max_lines]
    return '\n'.join(kept_lines) + '\n// ... (truncated)'


def build_rerank_snippet(chunk: Chunk, max_chars: int = 800) -> str:
    """
    Build a snippet for reranking from chunk metadata and code.
    
    Format:
        {rel_path}:{start_line}-{end_line}
        Summary: {summary_en}
        
        ```{language}
        {code_preview}
        ```
    
    Args:
        chunk: Chunk to build snippet from
        max_chars: Maximum total characters (default 800)
    
    Returns:
        Formatted snippet string
    """
    parts = []
    
    # Header with file location
    header = f"{chunk.rel_path}:{chunk.start_line}-{chunk.end_line}"
    parts.append(header)
    current_len = len(header)
    
    # Summary if available
    if chunk.summary_en:
        summary_line = f"Summary: {chunk.summary_en}"
        summary_trunc = safe_truncate_text(summary_line, 200)
        parts.append(summary_trunc)
        current_len += len(summary_trunc) + 1
    
    # Code preview if available and space remaining
    if chunk.text and current_len < max_chars - 100:
        remaining = max_chars - current_len - 50  # Reserve for fences
        
        # Normalize line endings
        code = chunk.text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Extract relevant lines (first 20-30 lines)
        code = extract_relevant_lines(code, max_lines=30)
        
        # Truncate to fit
        code = safe_truncate_text(code, remaining)
        
        # Close any open fences
        code = close_open_fences(code)
        
        # Wrap in fence
        lang = chunk.language.lower() if chunk.language else ""
        code_block = f"\n```{lang}\n{code}\n```"
        parts.append(code_block)
    
    return "\n".join(parts)


def truncate_for_filter(chunk: Chunk, max_chars: int) -> str:
    """
    Truncate chunk text for filter prompt.
    
    Extracts first max_chars with line-aware truncation and fence safety.
    
    Args:
        chunk: Chunk with text
        max_chars: Maximum characters
    
    Returns:
        Truncated code text
    """
    if not chunk.text:
        return ""
    
    # Normalize line endings
    text = chunk.text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Truncate safely
    truncated = safe_truncate_text(text, max_chars)
    
    # Close fences
    truncated = close_open_fences(truncated)
    
    return truncated

