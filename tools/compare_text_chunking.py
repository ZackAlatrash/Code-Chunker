#!/usr/bin/env python3
"""
Dry-run comparison tool for v4 vs v2 text chunking.

This script compares the text output from v4's current chunking strategy
against v2's CodeChunker-based strategy without modifying any indexes or
persisted data.

Usage:
  python tools/compare_text_chunking.py <file_path> <language> --chunk-range <start>:<end>
  python tools/compare_text_chunking.py service.go go --chunk-range 45:87

The script will:
1. Load the file
2. Chunk it with both v4 (current) and v2 strategies
3. Show a unified diff between the text outputs for the specified line range
"""

import os
import sys
import argparse
import difflib
import json
from typing import List, Dict, Any

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "chunking_v4"))

from Chunker import CodeChunker
from chunking_v4.text_chunkers import chunk_code_text_v2, map_v2_chunk_to_v4_span


def read_file_safely(path: str) -> str:
    """Read file with multiple encoding attempts"""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")


def get_v4_chunk_text(file_path: str, language: str, start_line: int, end_line: int, token_limit: int = 300) -> str:
    """Get chunk text using v4's current strategy"""
    code = read_file_safely(file_path)
    
    # Determine file extension
    ext = language.lower()
    if ext == "python":
        ext = "py"
    elif ext == "javascript":
        ext = "js"
    elif ext == "typescript":
        ext = "ts"
    
    # Create chunker
    chunker = CodeChunker(file_extension=ext)
    
    # Chunk the code
    chunks = chunker.chunk(code, token_limit=token_limit)
    
    # Find chunk that covers the requested range
    for chunk_idx in sorted(chunks.keys()):
        chunk = chunks[chunk_idx]
        if isinstance(chunk, dict):
            chunk_start = chunk.get("start_line")
            chunk_end = chunk.get("end_line")
            
            # Check if this chunk overlaps with our range
            if chunk_start and chunk_end:
                if not (chunk_end < start_line or chunk_start > end_line):
                    # Found overlapping chunk - extract the text for our range
                    chunk_text = chunk.get("text", "")
                    chunk_lines = chunk_text.split("\n")
                    
                    # Calculate offset within chunk
                    if start_line >= chunk_start and end_line <= chunk_end:
                        offset_start = start_line - chunk_start
                        offset_end = end_line - chunk_start + 1
                        return "\n".join(chunk_lines[offset_start:offset_end])
                    else:
                        # Range extends beyond this chunk - return what we have
                        return chunk_text
    
    # Fallback: extract directly from source
    lines = code.split("\n")
    return "\n".join(lines[start_line - 1:end_line])


def get_v2_chunk_text(file_path: str, language: str, start_line: int, end_line: int, token_limit: int = 300) -> str:
    """Get chunk text using v2's CodeChunker strategy"""
    code = read_file_safely(file_path)
    
    # Determine file extension
    ext = language.lower()
    if ext == "python":
        ext = "py"
    elif ext == "javascript":
        ext = "js"
    elif ext == "typescript":
        ext = "ts"
    
    # Get v2 chunks
    v2_chunks = chunk_code_text_v2(
        source_text=code,
        file_extension=ext,
        token_limit=token_limit
    )
    
    # Map to our line range
    source_lines = code.split("\n")
    return map_v2_chunk_to_v4_span(
        v2_chunks=v2_chunks,
        v4_start_line=start_line,
        v4_end_line=end_line,
        source_lines=source_lines
    )


def print_diff(v4_text: str, v2_text: str, file_path: str, start_line: int, end_line: int):
    """Print a unified diff between v4 and v2 text"""
    v4_lines = v4_text.splitlines(keepends=True)
    v2_lines = v2_text.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        v4_lines,
        v2_lines,
        fromfile=f"v4/{os.path.basename(file_path)} (L{start_line}-{end_line})",
        tofile=f"v2/{os.path.basename(file_path)} (L{start_line}-{end_line})",
        lineterm=""
    )
    
    print("=" * 80)
    print(f"TEXT CHUNKING COMPARISON: {file_path}")
    print(f"Line range: {start_line}-{end_line}")
    print("=" * 80)
    print()
    
    diff_lines = list(diff)
    if not diff_lines:
        print("✅ NO DIFFERENCES - v4 and v2 produce identical text for this range")
    else:
        print("📊 DIFF (v4 → v2):")
        print()
        for line in diff_lines:
            # Color code the diff
            if line.startswith("+++") or line.startswith("---"):
                print(f"\033[1m{line}\033[0m")  # Bold
            elif line.startswith("+"):
                print(f"\033[32m{line}\033[0m")  # Green
            elif line.startswith("-"):
                print(f"\033[31m{line}\033[0m")  # Red
            elif line.startswith("@@"):
                print(f"\033[36m{line}\033[0m")  # Cyan
            else:
                print(line)
    
    print()
    print("=" * 80)
    print(f"v4 text: {len(v4_text)} chars, {len(v4_lines)} lines")
    print(f"v2 text: {len(v2_text)} chars, {len(v2_lines)} lines")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare v4 vs v2 text chunking strategies")
    parser.add_argument("file_path", help="Path to source file")
    parser.add_argument("language", help="Language (e.g., go, python, typescript)")
    parser.add_argument("--chunk-range", required=True, 
                        help="Line range to compare (format: start:end, e.g., 45:87)")
    parser.add_argument("--token-limit", type=int, default=300,
                        help="Token limit for chunking (default: 300)")
    parser.add_argument("--json", action="store_true",
                        help="Output comparison as JSON instead of diff")
    
    args = parser.parse_args()
    
    # Parse chunk range
    try:
        start_line, end_line = map(int, args.chunk_range.split(":"))
    except ValueError:
        print("Error: --chunk-range must be in format start:end (e.g., 45:87)")
        sys.exit(1)
    
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)
    
    # Get texts
    try:
        v4_text = get_v4_chunk_text(args.file_path, args.language, start_line, end_line, args.token_limit)
    except Exception as e:
        print(f"Error getting v4 text: {e}")
        sys.exit(1)
    
    try:
        v2_text = get_v2_chunk_text(args.file_path, args.language, start_line, end_line, args.token_limit)
    except Exception as e:
        print(f"Error getting v2 text: {e}")
        sys.exit(1)
    
    # Output
    if args.json:
        result = {
            "file": args.file_path,
            "language": args.language,
            "start_line": start_line,
            "end_line": end_line,
            "v4_text": v4_text,
            "v2_text": v2_text,
            "identical": v4_text == v2_text,
            "v4_chars": len(v4_text),
            "v2_chars": len(v2_text),
            "v4_lines": len(v4_text.splitlines()),
            "v2_lines": len(v2_text.splitlines())
        }
        print(json.dumps(result, indent=2))
    else:
        print_diff(v4_text, v2_text, args.file_path, start_line, end_line)


if __name__ == "__main__":
    main()

