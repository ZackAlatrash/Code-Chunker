#!/usr/bin/env python3
"""
Test that Go chunks have proper neighbor linking from file header.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_go_file
from CodeParser import CodeParser
import logging


class TestGoNeighborsFromFileHeader:
    """Test that neighbors chain correctly from file header."""
    
    def test_file_header_has_no_prev_neighbor(self):
        """Test that file header chunk has prev = null."""
        go_code = '''package main

import (
    "fmt"
    "time"
)

type Service struct {
    Name string
}

func (s *Service) Method() {
    fmt.Println("Hello")
}

func main() {
    s := &Service{Name: "test"}
    s.Method()
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Find file header chunk
        header_chunk = None
        for chunk in chunks:
            if chunk.ast_path == "go:file_header":
                header_chunk = chunk
                break
        
        assert header_chunk is not None, "Should have a file header chunk"
        assert header_chunk.neighbors.prev is None, f"File header should have prev = null: {header_chunk.neighbors.prev}"
    
    def test_chunks_sorted_by_start_line(self):
        """Test that chunks are sorted by start_line."""
        go_code = '''package main

import "fmt"

type Service struct {
    Name string
}

func (s *Service) Method() {
    fmt.Println("Hello")
}

func main() {
    s := &Service{Name: "test"}
    s.Method()
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Check that chunks are sorted by start_line
        for i in range(1, len(chunks)):
            assert chunks[i].start_line >= chunks[i-1].start_line, \
                f"Chunks should be sorted by start_line: {chunks[i-1].start_line} vs {chunks[i].start_line}"
    
    def test_neighbor_chain_continuity(self):
        """Test that neighbor chain is continuous across the entire file."""
        go_code = '''package main

import "fmt"

type Service struct {
    Name string
}

func (s *Service) Method() {
    fmt.Println("Hello")
}

func main() {
    s := &Service{Name: "test"}
    s.Method()
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Sort chunks by start_line to ensure proper order
        chunks.sort(key=lambda c: c.start_line)
        
        # Check neighbor chain continuity
        for i in range(len(chunks)):
            chunk = chunks[i]
            
            # First chunk should have prev = null
            if i == 0:
                assert chunk.neighbors.prev is None, f"First chunk should have prev = null: {chunk.neighbors.prev}"
            else:
                # Other chunks should point to previous chunk
                prev_chunk = chunks[i-1]
                assert chunk.neighbors.prev == prev_chunk.chunk_id, \
                    f"Chunk {i} should point to previous chunk: {chunk.neighbors.prev} vs {prev_chunk.chunk_id}"
            
            # Last chunk should have next = null
            if i == len(chunks) - 1:
                assert chunk.neighbors.next is None, f"Last chunk should have next = null: {chunk.neighbors.next}"
            else:
                # Other chunks should point to next chunk
                next_chunk = chunks[i+1]
                assert chunk.neighbors.next == next_chunk.chunk_id, \
                    f"Chunk {i} should point to next chunk: {chunk.neighbors.next} vs {next_chunk.chunk_id}"
    
    def test_file_header_is_first_chunk(self):
        """Test that file header chunk is the first chunk by start_line."""
        go_code = '''package main

import (
    "fmt"
    "time"
)

type Service struct {
    Name string
}

func (s *Service) Method() {
    fmt.Println("Hello")
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Sort chunks by start_line
        chunks.sort(key=lambda c: c.start_line)
        
        # First chunk should be file header
        first_chunk = chunks[0]
        assert first_chunk.ast_path == "go:file_header", \
            f"First chunk should be file header: {first_chunk.ast_path}"
        
        # File header should have no previous neighbor
        assert first_chunk.neighbors.prev is None, \
            f"File header should have prev = null: {first_chunk.neighbors.prev}"
        
        # File header should point to next chunk
        if len(chunks) > 1:
            second_chunk = chunks[1]
            assert first_chunk.neighbors.next == second_chunk.chunk_id, \
                f"File header should point to second chunk: {first_chunk.neighbors.next} vs {second_chunk.chunk_id}"
    
    def test_exactly_one_file_header_per_file(self):
        """Test that there is exactly one file header chunk per file."""
        go_code = '''package main

import "fmt"

type Service struct {
    Name string
}

func (s *Service) Method() {
    fmt.Println("Hello")
}

func main() {
    s := &Service{Name: "test"}
    s.Method()
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Count file header chunks
        header_chunks = [c for c in chunks if c.ast_path == "go:file_header"]
        
        assert len(header_chunks) == 1, f"Should have exactly one file header chunk: {len(header_chunks)}"
        
        # The file header should cover package and imports
        header_chunk = header_chunks[0]
        assert header_chunk.start_line == 1, f"File header should start at line 1: {header_chunk.start_line}"
        assert "package main" in header_chunk.text, f"File header should contain package: {header_chunk.text}"
        assert "import" in header_chunk.text, f"File header should contain imports: {header_chunk.text}"
    
    def test_neighbor_chain_with_split_methods(self):
        """Test neighbor chain when methods are split into multiple parts."""
        go_code = '''package main

import (
    "context"
    "fmt"
    "time"
)

type Service struct {
    cache cacheClient
}

func (s *Service) LongMethod(ctx context.Context) error {
    // Part 1: Setup
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()
    
    // Part 2: Cache lookup
    item, err := s.cache.Get("key")
    if err == nil {
        return nil
    }
    
    // Part 3: Processing
    time.Sleep(100 * time.Millisecond)
    
    // Part 4: Cache set
    s.cache.Set("key", "value")
    
    return nil
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 30, 60, logging.getLogger(__name__), parser  # Small limits to force splitting
        )
        
        # Sort chunks by start_line
        chunks.sort(key=lambda c: c.start_line)
        
        # Check that all chunks are properly linked
        for i in range(len(chunks)):
            chunk = chunks[i]
            
            if i == 0:
                assert chunk.neighbors.prev is None, f"First chunk should have prev = null"
            else:
                prev_chunk = chunks[i-1]
                assert chunk.neighbors.prev == prev_chunk.chunk_id, \
                    f"Chunk {i} should link to previous chunk"
            
            if i == len(chunks) - 1:
                assert chunk.neighbors.next is None, f"Last chunk should have next = null"
            else:
                next_chunk = chunks[i+1]
                assert chunk.neighbors.next == next_chunk.chunk_id, \
                    f"Chunk {i} should link to next chunk"
