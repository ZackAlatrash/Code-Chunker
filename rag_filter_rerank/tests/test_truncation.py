"""
Tests for truncation helpers.
"""
import pytest
from rag_filter_rerank.schemas import Chunk
from rag_filter_rerank.truncation import (
    safe_truncate_text,
    close_open_fences,
    extract_relevant_lines,
    build_rerank_snippet,
    truncate_for_filter
)


def test_safe_truncate_text():
    """Test safe text truncation."""
    # Short text unchanged
    text = "Hello world"
    assert safe_truncate_text(text, 100) == text
    
    # Long text truncated at word boundary
    text = "The quick brown fox jumps over the lazy dog"
    result = safe_truncate_text(text, 20)
    assert len(result) <= 20
    # Should be truncated and not include full word if it would exceed limit
    assert "lazy" not in result  # Long text should be truncated before "lazy"
    
    # Truncate at newline
    text = "Line 1\nLine 2\nLine 3\nLine 4"
    result = safe_truncate_text(text, 15)
    assert len(result) <= 15
    assert result.count('\n') > 0


def test_close_open_fences():
    """Test markdown fence closing."""
    # Even fences (closed)
    text = "```python\ncode\n```"
    assert close_open_fences(text) == text
    
    # Odd fences (open)
    text = "```python\ncode"
    result = close_open_fences(text)
    assert result.endswith('```')
    assert result.count('```') == 2


def test_extract_relevant_lines():
    """Test line extraction."""
    lines = ["Line " + str(i) for i in range(100)]
    text = '\n'.join(lines)
    
    # Extract first 40 lines
    result = extract_relevant_lines(text, max_lines=40)
    result_lines = result.split('\n')
    
    # Should have 40 lines + truncation marker
    assert len(result_lines) <= 41
    assert "truncated" in result.lower()


def test_build_rerank_snippet():
    """Test rerank snippet building."""
    chunk = Chunk(
        id="test-1",
        repo_id="test-repo",
        rel_path="src/main.py",
        start_line=10,
        end_line=20,
        language="python",
        text="def hello():\n    print('Hello world')",
        summary_en="Prints hello world"
    )
    
    snippet = build_rerank_snippet(chunk, max_chars=800)
    
    # Should contain key elements
    assert "src/main.py:10-20" in snippet
    assert "Summary: Prints hello world" in snippet
    assert "```python" in snippet
    assert "def hello()" in snippet


def test_build_rerank_snippet_long_code():
    """Test snippet building with long code."""
    long_code = '\n'.join(["line " + str(i) for i in range(200)])
    
    chunk = Chunk(
        id="test-2",
        repo_id="test-repo",
        rel_path="src/long.py",
        start_line=1,
        end_line=200,
        language="python",
        text=long_code,
        summary_en="Long code file"
    )
    
    snippet = build_rerank_snippet(chunk, max_chars=800)
    
    # Should be truncated
    assert len(snippet) <= 850  # Small buffer for formatting
    assert "```python" in snippet
    assert "```" in snippet[snippet.index("```python") + 10:]  # Closes fence


def test_truncate_for_filter():
    """Test filter truncation."""
    chunk = Chunk(
        id="test-3",
        repo_id="test-repo",
        rel_path="src/code.go",
        start_line=1,
        end_line=50,
        language="go",
        text="package main\n" + ("// comment\n" * 100) + "func main() {}"
    )
    
    result = truncate_for_filter(chunk, max_chars=200)
    
    # Should be truncated
    assert len(result) <= 250  # Buffer for fence closing
    
    # Should close fences if any
    fence_count = result.count('```')
    assert fence_count % 2 == 0 or fence_count == 0


def test_truncate_with_open_fence():
    """Test truncation handles open code fences."""
    chunk = Chunk(
        id="test-4",
        repo_id="test-repo",
        rel_path="src/test.py",
        start_line=1,
        end_line=10,
        language="python",
        text="```python\ndef test():\n    pass\n# This is a very long comment that will get cut off"
    )
    
    result = truncate_for_filter(chunk, max_chars=50)
    
    # Should close the fence
    assert result.count('```') % 2 == 0

