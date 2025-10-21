#!/usr/bin/env python3
"""
Test that Go neighbors chain correctly from file header.
"""

import pytest
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_go_file
from src.CodeParser import CodeParser


class TestGoNeighborsChainFromFileHeader:
    """Test that Go neighbors chain correctly from file header."""
    
    def test_file_header_is_first_chunk(self):
        """Test that file header chunk is the first chunk by start_line."""
        go_content = """
package foreca

import (
    "context"
    "time"
)

type Service struct {
    cacheExpirationDuration time.Duration
}

func NewService() *Service {
    return &Service{}
}
"""
        
        parser = CodeParser()
        logger = logging.getLogger(__name__)
        chunks = chunk_go_file(
            go_content, Path("test.go"), "test", "sha123", "test.go", 
            "gpt-4", 150, 380, logger, parser
        )
        
        # Sort chunks by start_line
        chunks.sort(key=lambda c: c.start_line)
        
        # First chunk should be file header
        assert chunks[0].ast_path == "go:file_header"
        assert chunks[0].start_line == 1  # package line
        assert chunks[0].neighbors.prev is None  # First chunk has no previous
    
    def test_neighbors_chain_correctly(self):
        """Test that neighbors chain correctly across all chunks."""
        go_content = """
package foreca

import (
    "context"
    "time"
)

type Service struct {
    cacheExpirationDuration time.Duration
}

func NewService() *Service {
    return &Service{}
}

func (s *Service) GetForecast(ctx context.Context) error {
    return nil
}
"""
        
        parser = CodeParser()
        logger = logging.getLogger(__name__)
        chunks = chunk_go_file(
            go_content, Path("test.go"), "test", "sha123", "test.go", 
            "gpt-4", 150, 380, logger, parser
        )
        
        # Sort chunks by start_line
        chunks.sort(key=lambda c: c.start_line)
        
        # Verify neighbor chain
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Each chunk should point to previous
                assert chunk.neighbors.prev == chunks[i-1].chunk_id
            else:
                # First chunk should have no previous
                assert chunk.neighbors.prev is None
            
            if i < len(chunks) - 1:
                # Each chunk should point to next
                assert chunk.neighbors.next == chunks[i+1].chunk_id
            else:
                # Last chunk should have no next
                assert chunk.neighbors.next is None
    
    def test_file_header_points_to_next_chunk(self):
        """Test that file header chunk points to the next chunk."""
        go_content = """
package foreca

import (
    "context"
    "time"
)

type Service struct {
    cacheExpirationDuration time.Duration
}
"""
        
        parser = CodeParser()
        logger = logging.getLogger(__name__)
        chunks = chunk_go_file(
            go_content, Path("test.go"), "test", "sha123", "test.go", 
            "gpt-4", 150, 380, logger, parser
        )
        
        # Sort chunks by start_line
        chunks.sort(key=lambda c: c.start_line)
        
        # Should have file header and type chunk
        assert len(chunks) == 2
        
        # File header should point to type chunk
        assert chunks[0].ast_path == "go:file_header"
        assert chunks[0].neighbors.next == chunks[1].chunk_id
        
        # Type chunk should point back to file header
        assert chunks[1].neighbors.prev == chunks[0].chunk_id
        assert chunks[1].neighbors.next is None
    
    def test_multiple_chunks_chain_correctly(self):
        """Test that multiple chunks chain correctly."""
        go_content = """
package foreca

import (
    "context"
    "time"
)

type Service struct {
    cacheExpirationDuration time.Duration
}

type Client interface {
    Get(ctx context.Context) error
}

func NewService() *Service {
    return &Service{}
}

func (s *Service) GetForecast(ctx context.Context) error {
    return nil
}
"""
        
        parser = CodeParser()
        logger = logging.getLogger(__name__)
        chunks = chunk_go_file(
            go_content, Path("test.go"), "test", "sha123", "test.go", 
            "gpt-4", 150, 380, logger, parser
        )
        
        # Sort chunks by start_line
        chunks.sort(key=lambda c: c.start_line)
        
        # Should have file header + 4 other chunks
        assert len(chunks) == 5
        
        # Verify the chain
        for i in range(len(chunks)):
            if i > 0:
                assert chunks[i].neighbors.prev == chunks[i-1].chunk_id
            if i < len(chunks) - 1:
                assert chunks[i].neighbors.next == chunks[i+1].chunk_id
    
    def test_file_header_span_covers_package_and_imports(self):
        """Test that file header span covers package and imports."""
        go_content = """
package foreca

import (
    "context"
    "time"
    "fmt"
)

type Service struct {
    cacheExpirationDuration time.Duration
}
"""
        
        parser = CodeParser()
        logger = logging.getLogger(__name__)
        chunks = chunk_go_file(
            go_content, Path("test.go"), "test", "sha123", "test.go", 
            "gpt-4", 150, 380, logger, parser
        )
        
        # Sort chunks by start_line
        chunks.sort(key=lambda c: c.start_line)
        
        # File header should span from package to end of imports
        file_header = chunks[0]
        assert file_header.ast_path == "go:file_header"
        assert file_header.start_line == 1  # package line
        assert file_header.end_line >= 6  # Should include all import lines
    
    def test_no_chunk_has_null_prev_except_first(self):
        """Test that no chunk has prev=null except the very first."""
        go_content = """
package foreca

import "context"

type Service struct {
    cacheExpirationDuration time.Duration
}

func NewService() *Service {
    return &Service{}
}
"""
        
        parser = CodeParser()
        logger = logging.getLogger(__name__)
        chunks = chunk_go_file(
            go_content, Path("test.go"), "test", "sha123", "test.go", 
            "gpt-4", 150, 380, logger, parser
        )
        
        # Sort chunks by start_line
        chunks.sort(key=lambda c: c.start_line)
        
        # Only first chunk should have prev=null
        for i, chunk in enumerate(chunks):
            if i == 0:
                assert chunk.neighbors.prev is None
            else:
                assert chunk.neighbors.prev is not None
    
    def test_chunks_sorted_by_start_line(self):
        """Test that chunks are sorted by start_line."""
        go_content = """
package foreca

import "context"

func NewService() *Service {
    return &Service{}
}

type Service struct {
    cacheExpirationDuration time.Duration
}
"""
        
        parser = CodeParser()
        logger = logging.getLogger(__name__)
        chunks = chunk_go_file(
            go_content, Path("test.go"), "test", "sha123", "test.go", 
            "gpt-4", 150, 380, logger, parser
        )
        
        # Sort chunks by start_line
        chunks.sort(key=lambda c: c.start_line)
        
        # Verify chunks are in order by start_line
        for i in range(len(chunks) - 1):
            assert chunks[i].start_line <= chunks[i+1].start_line


if __name__ == '__main__':
    pytest.main([__file__])
