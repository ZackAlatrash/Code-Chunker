#!/usr/bin/env python3
"""
Test Go chunk neighbors and ordering.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_go_file


class TestGoNeighbors:
    """Test that Go chunks have correct neighbors and ordering."""
    
    def test_chunks_ordered_by_start_line(self):
        """Test that chunks are ordered by start_line in ascending order."""
        go_code = '''
package main

import "fmt"

type Service struct {
    name string
}

func (s *Service) GetName() string {
    return s.name
}

func NewService(name string) *Service {
    return &Service{name: name}
}

func main() {
    service := NewService("test")
    fmt.Println(service.GetName())
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
                    
                    root = MockNode('source_file', (0, 0), (20, 0))
                    root.children = [
                        MockNode('package_declaration', (0, 0), (0, 10)),
                        MockNode('import_declaration', (2, 0), (2, 20)),
                        MockNode('type_declaration', (4, 0), (6, 0)),
                        MockNode('method_declaration', (8, 0), (10, 0)),
                        MockNode('function_declaration', (12, 0), (14, 0)),
                        MockNode('function_declaration', (16, 0), (20, 0)),
                    ]
                    return root
            
            parser = MockParser()
            chunks = chunk_go_file(
                go_code, temp_path, "test_repo", "sha123", "test.go",
                "gpt-4", 150, 380, None, parser
            )
            
            # Chunks should be ordered by start_line
            for i in range(1, len(chunks)):
                assert chunks[i].start_line >= chunks[i-1].start_line, \
                    f"Chunks not ordered: {chunks[i-1].start_line} vs {chunks[i].start_line}"
                
        finally:
            temp_path.unlink()
    
    def test_neighbors_chain_correctly(self):
        """Test that neighbors.prev/next chain correctly."""
        go_code = '''
package test

type User struct {
    ID   string
    Name string
}

func (u *User) GetID() string {
    return u.ID
}

func (u *User) GetName() string {
    return u.Name
}

func NewUser(id, name string) *User {
    return &User{ID: id, Name: name}
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
                    
                    root = MockNode('source_file', (0, 0), (20, 0))
                    root.children = [
                        MockNode('package_declaration', (0, 0), (0, 10)),
                        MockNode('type_declaration', (2, 0), (5, 0)),
                        MockNode('method_declaration', (7, 0), (9, 0)),
                        MockNode('method_declaration', (11, 0), (13, 0)),
                        MockNode('function_declaration', (15, 0), (17, 0)),
                    ]
                    return root
            
            parser = MockParser()
            chunks = chunk_go_file(
                go_code, temp_path, "test_repo", "sha123", "test.go",
                "gpt-4", 150, 380, None, parser
            )
            
            # Test neighbor chaining
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # Current chunk should have prev pointing to previous chunk
                    assert chunk.neighbors.prev == chunks[i-1].chunk_id, \
                        f"Chunk {i} prev neighbor incorrect"
                else:
                    # First chunk should have no prev neighbor
                    assert chunk.neighbors.prev is None, \
                        f"First chunk should have no prev neighbor"
                
                if i < len(chunks) - 1:
                    # Current chunk should have next pointing to next chunk
                    assert chunk.neighbors.next == chunks[i+1].chunk_id, \
                        f"Chunk {i} next neighbor incorrect"
                else:
                    # Last chunk should have no next neighbor
                    assert chunk.neighbors.next is None, \
                        f"Last chunk should have no next neighbor"
                
        finally:
            temp_path.unlink()
    
    def test_neighbors_with_header_type_methods(self):
        """Test neighbors across header → type → methods sequence."""
        go_code = '''
package foreca

import (
    "context"
    "time"
)

type WeatherService struct {
    client Client
    cache  Cache
}

func (ws *WeatherService) GetForecast(ctx context.Context, location string) (*Forecast, error) {
    // Implementation here
    return nil, nil
}

func (ws *WeatherService) GetCurrentWeather(ctx context.Context, location string) (*Weather, error) {
    // Implementation here
    return nil, nil
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
                    
                    root = MockNode('source_file', (0, 0), (20, 0))
                    root.children = [
                        MockNode('package_declaration', (0, 0), (0, 10)),
                        MockNode('import_declaration', (2, 0), (5, 0)),
                        MockNode('type_declaration', (7, 0), (10, 0)),
                        MockNode('method_declaration', (12, 0), (15, 0)),
                        MockNode('method_declaration', (17, 0), (20, 0)),
                    ]
                    return root
            
            parser = MockParser()
            chunks = chunk_go_file(
                go_code, temp_path, "test_repo", "sha123", "test.go",
                "gpt-4", 150, 380, None, parser
            )
            
            # Should have chunks for package/imports, type, and methods
            assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"
            
            # Verify the chain: header → type → method1 → method2
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                
                # Current chunk's next should point to next chunk
                assert current_chunk.neighbors.next == next_chunk.chunk_id, \
                    f"Chunk {i} next neighbor incorrect"
                
                # Next chunk's prev should point to current chunk
                assert next_chunk.neighbors.prev == current_chunk.chunk_id, \
                    f"Chunk {i+1} prev neighbor incorrect"
            
            # First chunk should have no prev
            assert chunks[0].neighbors.prev is None, "First chunk should have no prev"
            
            # Last chunk should have no next
            assert chunks[-1].neighbors.next is None, "Last chunk should have no next"
                
        finally:
            temp_path.unlink()
    
    def test_unique_chunk_ids(self):
        """Test that all chunks have unique IDs."""
        go_code = '''
package test

type Service struct {
    field string
}

func (s *Service) Method1() {
    return
}

func (s *Service) Method2() {
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
                    
                    root = MockNode('source_file', (0, 0), (15, 0))
                    root.children = [
                        MockNode('package_declaration', (0, 0), (0, 10)),
                        MockNode('type_declaration', (2, 0), (4, 0)),
                        MockNode('method_declaration', (6, 0), (8, 0)),
                        MockNode('method_declaration', (10, 0), (12, 0)),
                    ]
                    return root
            
            parser = MockParser()
            chunks = chunk_go_file(
                go_code, temp_path, "test_repo", "sha123", "test.go",
                "gpt-4", 150, 380, None, parser
            )
            
            # All chunk IDs should be unique
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            assert len(chunk_ids) == len(set(chunk_ids)), \
                f"Duplicate chunk IDs found: {chunk_ids}"
                
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__])
