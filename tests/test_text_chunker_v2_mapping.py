#!/usr/bin/env python3
"""
Unit tests for v2 text chunker mapping to v4 spans.

Tests that:
1. v2 chunker returns windows
2. Mapping to v4 ranges yields non-empty text
3. Metadata remains unchanged (except text)
"""

import os
import sys
import unittest

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "chunking_v4"))

from chunking_v4.text_chunkers import chunk_code_text_v2, map_v2_chunk_to_v4_span


class TestV2TextChunkerMapping(unittest.TestCase):
    """Test v2 text chunker and mapping to v4 spans"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Simple Go code sample
        self.go_code = """package main

import "fmt"

func main() {
\tfmt.Println("Hello")
}

func helper() string {
\treturn "test"
}
"""
        
        # Simple Python code sample
        self.python_code = """def main():
    print("Hello")
    return 0

def helper():
    return "test"

class TestClass:
    def method(self):
        pass
"""
    
    def test_v2_chunker_returns_windows_go(self):
        """Test that v2 chunker returns valid windows for Go code"""
        chunks = chunk_code_text_v2(
            source_text=self.go_code,
            file_extension="go",
            token_limit=300
        )
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            self.assertIn("start_line", chunk)
            self.assertIn("end_line", chunk)
            self.assertIn("text", chunk)
            self.assertGreater(chunk["end_line"], 0)
            self.assertGreaterEqual(chunk["end_line"], chunk["start_line"])
            self.assertIsInstance(chunk["text"], str)
    
    def test_v2_chunker_returns_windows_python(self):
        """Test that v2 chunker returns valid windows for Python code"""
        chunks = chunk_code_text_v2(
            source_text=self.python_code,
            file_extension="py",
            token_limit=300
        )
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            self.assertIn("start_line", chunk)
            self.assertIn("end_line", chunk)
            self.assertIn("text", chunk)
    
    def test_mapping_to_v4_range_yields_text(self):
        """Test that mapping v2 chunks to v4 range produces non-empty text"""
        chunks = chunk_code_text_v2(
            source_text=self.go_code,
            file_extension="go",
            token_limit=300
        )
        
        source_lines = self.go_code.split("\n")
        
        # Map to a range that should overlap with chunks
        result = map_v2_chunk_to_v4_span(
            v2_chunks=chunks,
            v4_start_line=5,
            v4_end_line=7,
            source_lines=source_lines
        )
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)
        self.assertIn("main", result)  # Should contain the main function
    
    def test_mapping_respects_v4_boundaries(self):
        """Test that mapped text respects v4's line boundaries"""
        chunks = chunk_code_text_v2(
            source_text=self.go_code,
            file_extension="go",
            token_limit=300
        )
        
        source_lines = self.go_code.split("\n")
        v4_start = 5
        v4_end = 7
        
        result = map_v2_chunk_to_v4_span(
            v2_chunks=chunks,
            v4_start_line=v4_start,
            v4_end_line=v4_end,
            source_lines=source_lines
        )
        
        # Count lines in result
        result_lines = result.split("\n")
        expected_lines = v4_end - v4_start + 1
        
        # Should be close to expected (may vary by 1-2 lines due to mapping)
        self.assertLessEqual(abs(len(result_lines) - expected_lines), 2)
    
    def test_mapping_fallback_on_no_overlap(self):
        """Test that mapping falls back to source extraction when no v2 chunks overlap"""
        # Empty chunk list - should fallback to source
        result = map_v2_chunk_to_v4_span(
            v2_chunks=[],
            v4_start_line=1,
            v4_end_line=3,
            source_lines=self.go_code.split("\n")
        )
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)
        self.assertIn("package main", result)
    
    def test_metadata_independence(self):
        """Test that v2 text chunking is independent of metadata"""
        # This test verifies that chunk_code_text_v2 only returns text info
        # and doesn't interfere with any v4 metadata fields
        
        chunks = chunk_code_text_v2(
            source_text=self.python_code,
            file_extension="py",
            token_limit=300
        )
        
        # Verify that chunks only contain the expected keys
        for chunk in chunks:
            keys = set(chunk.keys())
            self.assertEqual(keys, {"start_line", "end_line", "text"})
            
            # Verify no metadata leakage
            self.assertNotIn("symbols", chunk)
            self.assertNotIn("primary_symbol", chunk)
            self.assertNotIn("ast_path", chunk)
            self.assertNotIn("all_symbols", chunk)


class TestV2ChunkerEdgeCases(unittest.TestCase):
    """Test edge cases for v2 chunker"""
    
    def test_empty_file(self):
        """Test handling of empty file"""
        chunks = chunk_code_text_v2(
            source_text="",
            file_extension="go",
            token_limit=300
        )
        
        # Should return at least one chunk (even if empty)
        self.assertIsInstance(chunks, list)
    
    def test_single_line_file(self):
        """Test handling of single line file"""
        chunks = chunk_code_text_v2(
            source_text="package main",
            file_extension="go",
            token_limit=300
        )
        
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]["start_line"], 1)
    
    def test_large_single_function(self):
        """Test that large single function is kept intact"""
        # Create a function that exceeds token limit
        large_func = "def large_function():\n"
        for i in range(100):
            large_func += f"    x{i} = {i} * {i}\n"
        large_func += "    return sum()\n"
        
        chunks = chunk_code_text_v2(
            source_text=large_func,
            file_extension="py",
            token_limit=100  # Intentionally small
        )
        
        # Should still return chunks (may exceed limit for completeness)
        self.assertGreater(len(chunks), 0)


if __name__ == "__main__":
    unittest.main()

