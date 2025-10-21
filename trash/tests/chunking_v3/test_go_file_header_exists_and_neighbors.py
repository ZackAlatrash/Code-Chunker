"""
Test Go file header chunk exists and proper neighbor linking.
"""

import pytest
import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from build_chunks_v3 import chunk_go_file
from src.CodeParser import CodeParser


class TestGoFileHeaderExistsAndNeighbors:
    """Test Go file header chunk creation and neighbor linking."""
    
    def test_file_header_exists_and_neighbors(self):
        """Test that file header chunk exists and first non-header chunk points to it."""
        # Skip if Go parser not available
        try:
            parser = CodeParser()
            tree = parser.parse_code("package main\nimport \"fmt\"\nfunc main() {}", 'go')
            if not tree:
                pytest.skip("Go parser not available, skipping test")
        except Exception:
            pytest.skip("Go parser not available, skipping test")
        
        # Test Go file content
        go_content = '''package main

import (
    "fmt"
    "os"
)

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
        
        # Create temporary file
        test_file = Path("test.go")
        test_file.write_text(go_content)
        
        try:
            # Chunk the file
            chunks = chunk_go_file(
                go_content, test_file, "test-repo", "test-sha", "test.go", 
                "gpt-4", 150, 380, logging.getLogger(__name__), parser
            )
            
            # Should have at least 2 chunks (header + at least one other)
            assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
            
            # Find file header chunk
            header_chunk = None
            non_header_chunks = []
            
            for chunk in chunks:
                if chunk.ast_path == "go:file_header":
                    header_chunk = chunk
                else:
                    non_header_chunks.append(chunk)
            
            # File header chunk should exist
            assert header_chunk is not None, "File header chunk not found"
            assert header_chunk.neighbors.prev is None, "File header chunk should have prev = null"
            
            # Sort non-header chunks by start_line
            non_header_chunks.sort(key=lambda x: x.start_line)
            
            if non_header_chunks:
                first_non_header = non_header_chunks[0]
                # First non-header chunk should point to header chunk
                assert first_non_header.neighbors.prev == header_chunk.chunk_id, \
                    f"First non-header chunk should point to header chunk, got prev: {first_non_header.neighbors.prev}"
            
            # Verify header chunk content
            assert "package main" in header_chunk.text, "Header chunk should contain package declaration"
            assert "import" in header_chunk.text, "Header chunk should contain imports"
            
        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()
    
    def test_file_header_only_package_no_imports(self):
        """Test file header chunk with only package declaration, no imports."""
        # Skip if Go parser not available
        try:
            parser = CodeParser()
            tree = parser.parse_code("package main\nfunc main() {}", 'go')
            if not tree:
                pytest.skip("Go parser not available, skipping test")
        except Exception:
            pytest.skip("Go parser not available, skipping test")
        
        # Test Go file content with no imports
        go_content = '''package main

func main() {
    println("Hello, World!")
}
'''
        
        # Create temporary file
        test_file = Path("test_no_imports.go")
        test_file.write_text(go_content)
        
        try:
            # Chunk the file
            chunks = chunk_go_file(
                go_content, test_file, "test-repo", "test-sha", "test_no_imports.go", 
                "gpt-4", 150, 380, logging.getLogger(__name__), parser
            )
            
            # Should have at least 2 chunks (header + main function)
            assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
            
            # Find file header chunk
            header_chunk = None
            for chunk in chunks:
                if chunk.ast_path == "go:file_header":
                    header_chunk = chunk
                    break
            
            # File header chunk should exist
            assert header_chunk is not None, "File header chunk not found"
            assert header_chunk.neighbors.prev is None, "File header chunk should have prev = null"
            
            # Verify header chunk content
            assert "package main" in header_chunk.text, "Header chunk should contain package declaration"
            
        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()
    
    def test_multiple_chunks_neighbor_chain(self):
        """Test that all chunks are properly linked in a chain."""
        # Skip if Go parser not available
        try:
            parser = CodeParser()
            tree = parser.parse_code("package main\nimport \"fmt\"\nfunc main() {}", 'go')
            if not tree:
                pytest.skip("Go parser not available, skipping test")
        except Exception:
            pytest.skip("Go parser not available, skipping test")
        
        # Test Go file content with multiple chunks
        go_content = '''package main

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
        
        # Create temporary file
        test_file = Path("test_multiple.go")
        test_file.write_text(go_content)
        
        try:
            # Chunk the file
            chunks = chunk_go_file(
                go_content, test_file, "test-repo", "test-sha", "test_multiple.go", 
                "gpt-4", 150, 380, logging.getLogger(__name__), parser
            )
            
            # Should have multiple chunks
            assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"
            
            # Sort chunks by start_line
            chunks.sort(key=lambda x: x.start_line)
            
            # First chunk should be file header with prev = null
            first_chunk = chunks[0]
            assert first_chunk.ast_path == "go:file_header", "First chunk should be file header"
            assert first_chunk.neighbors.prev is None, "First chunk should have prev = null"
            
            # Each subsequent chunk should point to the previous one
            for i in range(1, len(chunks)):
                current_chunk = chunks[i]
                previous_chunk = chunks[i-1]
                
                assert current_chunk.neighbors.prev == previous_chunk.chunk_id, \
                    f"Chunk {i} should point to previous chunk {i-1}"
            
            # Last chunk should have next = null
            last_chunk = chunks[-1]
            assert last_chunk.neighbors.next is None, "Last chunk should have next = null"
            
        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()


if __name__ == '__main__':
    pytest.main([__file__])