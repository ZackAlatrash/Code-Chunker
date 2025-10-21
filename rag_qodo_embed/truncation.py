"""
Text truncation utilities for embedding pipeline.

Provides UTF-8 safe truncation to ensure consistent embedding dimensions
and prevent memory issues with very long code chunks.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def truncate_text(text: str, max_chars: int) -> str:
    """
    Truncate text to maximum character count, UTF-8 safe.
    
    Args:
        text: Input text to truncate
        max_chars: Maximum number of characters
    
    Returns:
        Truncated text (may be shorter than max_chars if UTF-8 boundary)
    
    Raises:
        ValueError: If max_chars is less than 1
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    
    if not text:
        return text
    
    if len(text) <= max_chars:
        return text
    
    # Truncate to max_chars
    truncated = text[:max_chars]
    
    # Ensure we don't break UTF-8 sequences
    # Find the last complete character
    while truncated:
        try:
            # Try to encode/decode to check for valid UTF-8
            truncated.encode('utf-8').decode('utf-8')
            break
        except UnicodeDecodeError:
            # Remove the last character and try again
            truncated = truncated[:-1]
    
    # Log truncation if significant
    original_len = len(text)
    truncated_len = len(truncated)
    if truncated_len < original_len * 0.9:  # More than 10% truncated
        logger.warning(
            f"Truncated text from {original_len} to {truncated_len} chars "
            f"({(original_len - truncated_len) / original_len * 100:.1f}% reduction)"
        )
    
    return truncated


def truncate_text_smart(text: str, max_chars: int, preserve_lines: bool = True) -> str:
    """
    Smart truncation that tries to preserve line boundaries.
    
    Args:
        text: Input text to truncate
        max_chars: Maximum number of characters
        preserve_lines: Whether to try to end at a line boundary
    
    Returns:
        Truncated text
    """
    if not preserve_lines or len(text) <= max_chars:
        return truncate_text(text, max_chars)
    
    # Find the last newline before max_chars
    last_newline = text.rfind('\n', 0, max_chars)
    
    if last_newline > max_chars * 0.8:  # If we can preserve a reasonable amount
        truncated = text[:last_newline]
    else:
        truncated = text[:max_chars]
    
    # Apply UTF-8 safe truncation
    return truncate_text(truncated, max_chars)
