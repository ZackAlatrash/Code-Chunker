#!/usr/bin/env python3
"""
Test that Go chunking produces no empty chunks or invalid spans.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_go_file, create_go_chunk


class TestGoNoEmptyChunks:
    """Test that Go chunking produces no empty chunks or invalid spans."""
    
    def test_no_empty_core_chunks(self):
        """Test that no chunks have empty core content."""
        go_code = '''
package main

import "fmt"

type Service struct {
    name string
}

func (s *Service) GetName() string {
    return s.name
}

func main() {
    fmt.Println("Hello, World!")
}
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
            f.write(go_code)
            temp_path = Path(f.name)
        
        try:
            # Mock parser for testing
            class MockParser:
                def parse_code(self, content, ext):
                    class MockNode:
                        def __init__(self, node_type, start_point, end_point):
                            self.type = node_type
                            self.start_point = start_point
                            self.end_point = end_point
                            self.children = []
                    
                    root = MockNode('source_file', (0, 0), (15, 0))
                    root.children = [
                        MockNode('package_declaration', (0, 0), (0, 10)),
                        MockNode('import_declaration', (2, 0), (2, 20)),
                        MockNode('type_declaration', (4, 0), (6, 0)),
                        MockNode('method_declaration', (8, 0), (10, 0)),
                        MockNode('function_declaration', (12, 0), (14, 0)),
                    ]
                    return root
            
            parser = MockParser()
            chunks = chunk_go_file(
                go_code, temp_path, "test_repo", "sha123", "test.go",
                "gpt-4", 150, 380, None, parser
            )
            
            # All chunks should have non-empty core content
            for chunk in chunks:
                assert chunk.core.strip() != "", f"Chunk {chunk.chunk_id} has empty core"
                assert len(chunk.core.strip()) > 0, f"Chunk {chunk.chunk_id} has empty core"
                
        finally:
            temp_path.unlink()
    
    def test_no_reversed_spans(self):
        """Test that no chunks have end_line < start_line."""
        go_code = '''
package test

type User struct {
    ID   string
    Name string
}

func (u *User) GetID() string {
    return u.ID
}
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
            f.write(go_code)
            temp_path = Path(f.name)
        
        try:
            # Mock parser for testing
            class MockParser:
                def parse_code(self, content, ext):
                    class MockNode:
                        def __init__(self, node_type, start_point, end_point):
                            self.type = node_type
                            self.start_point = start_point
                            self.end_point = end_point
                            self.children = []
                    
                    root = MockNode('source_file', (0, 0), (10, 0))
                    root.children = [
                        MockNode('package_declaration', (0, 0), (0, 10)),
                        MockNode('type_declaration', (2, 0), (5, 0)),
                        MockNode('method_declaration', (7, 0), (9, 0)),
                    ]
                    return root
            
            parser = MockParser()
            chunks = chunk_go_file(
                go_code, temp_path, "test_repo", "sha123", "test.go",
                "gpt-4", 150, 380, None, parser
            )
            
            # All chunks should have valid line spans
            for chunk in chunks:
                assert chunk.end_line >= chunk.start_line, \
                    f"Chunk {chunk.chunk_id} has invalid span: {chunk.start_line}-{chunk.end_line}"
                assert chunk.start_line > 0, \
                    f"Chunk {chunk.chunk_id} has invalid start_line: {chunk.start_line}"
                assert chunk.end_line > 0, \
                    f"Chunk {chunk.chunk_id} has invalid end_line: {chunk.end_line}"
                
        finally:
            temp_path.unlink()
    
    def test_no_chunks_with_empty_text(self):
        """Test that no chunks have empty text field."""
        go_code = '''
package main

import "fmt"

func main() {
    fmt.Println("Hello")
}
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
            f.write(go_code)
            temp_path = Path(f.name)
        
        try:
            # Mock parser for testing
            class MockParser:
                def parse_code(self, content, ext):
                    class MockNode:
                        def __init__(self, node_type, start_point, end_point):
                            self.type = node_type
                            self.start_point = start_point
                            self.end_point = end_point
                            self.children = []
                    
                    root = MockNode('source_file', (0, 0), (6, 0))
                    root.children = [
                        MockNode('package_declaration', (0, 0), (0, 10)),
                        MockNode('import_declaration', (2, 0), (2, 20)),
                        MockNode('function_declaration', (4, 0), (6, 0)),
                    ]
                    return root
            
            parser = MockParser()
            chunks = chunk_go_file(
                go_code, temp_path, "test_repo", "sha123", "test.go",
                "gpt-4", 150, 380, None, parser
            )
            
            # All chunks should have non-empty text
            for chunk in chunks:
                assert chunk.text.strip() != "", f"Chunk {chunk.chunk_id} has empty text"
                assert len(chunk.text.strip()) > 0, f"Chunk {chunk.chunk_id} has empty text"
                
        finally:
            temp_path.unlink()
    
    def test_create_go_chunk_validation(self):
        """Test that create_go_chunk validates input properly."""
        # Test with empty chunk text
        chunk = create_go_chunk(
            "", "package test", 1, 5, "go:test", "repo", Path("test.go"), 
            "sha123", "test.go", "type_declaration", {}, [], "gpt-4", None
        )
        assert chunk is None, "Should return None for empty chunk text"
        
        # Test with reversed line numbers
        chunk = create_go_chunk(
            "type Test struct {}", "package test", 10, 5, "go:test", 
            "repo", Path("test.go"), "sha123", "test.go", "type_declaration", 
            {}, [], "gpt-4", None
        )
        assert chunk is None, "Should return None for reversed line numbers"
        
        # Test with valid input
        chunk = create_go_chunk(
            "type Test struct {\n    field string\n}", 
            "package test", 1, 3, "go:type:Test", "repo", Path("test.go"), 
            "sha123", "test.go", "type_declaration", {"type_name": "Test"}, 
            [], "gpt-4", None
        )
        assert chunk is not None, "Should create chunk for valid input"
        assert chunk.core.strip() != "", "Should have non-empty core"
        assert chunk.end_line >= chunk.start_line, "Should have valid line span"
    
    def test_no_whitespace_only_chunks(self):
        """Test that chunks with only whitespace are rejected."""
        go_code = '''
package main

type Service struct {
    // This is a comment
    field string
}

func (s *Service) Method() {
    // Another comment
    return
}
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
            f.write(go_code)
            temp_path = Path(f.name)
        
        try:
            # Mock parser for testing
            class MockParser:
                def parse_code(self, content, ext):
                    class MockNode:
                        def __init__(self, node_type, start_point, end_point):
                            self.type = node_type
                            self.start_point = start_point
                            self.end_point = end_point
                            self.children = []
                    
                    root = MockNode('source_file', (0, 0), (12, 0))
                    root.children = [
                        MockNode('package_declaration', (0, 0), (0, 10)),
                        MockNode('type_declaration', (2, 0), (6, 0)),
                        MockNode('method_declaration', (8, 0), (12, 0)),
                    ]
                    return root
            
            parser = MockParser()
            chunks = chunk_go_file(
                go_code, temp_path, "test_repo", "sha123", "test.go",
                "gpt-4", 150, 380, None, parser
            )
            
            # All chunks should have meaningful content (not just whitespace)
            for chunk in chunks:
                # Remove whitespace and check for meaningful content
                meaningful_content = ''.join(chunk.core.split())
                assert len(meaningful_content) > 0, \
                    f"Chunk {chunk.chunk_id} has only whitespace: '{chunk.core}'"
                
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__])
