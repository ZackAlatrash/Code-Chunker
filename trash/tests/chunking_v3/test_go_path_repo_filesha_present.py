#!/usr/bin/env python3
"""
Test that Go chunks have proper path, repo, and file_sha metadata.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_go_file, normalize_go_path
from CodeParser import CodeParser
import logging


class TestGoPathRepoFileShaPresent:
    """Test that chunks have proper path, repo, and file_sha metadata."""
    
    def test_no_rel_path_field(self):
        """Test that rel_path field is not emitted."""
        go_code = '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Check that no chunk has rel_path field
        for chunk in chunks:
            # The ChunkRecord model should not have rel_path field
            assert not hasattr(chunk, 'rel_path'), f"Chunk should not have rel_path field: {chunk}"
    
    def test_repo_relative_paths(self):
        """Test that paths are repo-relative POSIX paths."""
        go_code = '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("internal/service/main.go"), "test-repo", "test-sha", "internal/service/main.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Check that all chunks have repo-relative paths
        for chunk in chunks:
            assert chunk.path == "internal/service/main.go", f"Path should be repo-relative: {chunk.path}"
            assert not chunk.path.startswith('/'), f"Path should not start with /: {chunk.path}"
            assert '\\' not in chunk.path, f"Path should use POSIX separators: {chunk.path}"
    
    def test_repo_field_present(self):
        """Test that repo field is present and non-empty."""
        go_code = '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Check that all chunks have repo field
        for chunk in chunks:
            assert chunk.repo == "test-repo", f"Repo should be set: {chunk.repo}"
            assert chunk.repo, f"Repo should be non-empty: {chunk.repo}"
    
    def test_file_sha_field_present(self):
        """Test that file_sha field is present and non-empty."""
        go_code = '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Check that all chunks have file_sha field
        for chunk in chunks:
            assert chunk.file_sha == "test-sha", f"File SHA should be set: {chunk.file_sha}"
            assert chunk.file_sha, f"File SHA should be non-empty: {chunk.file_sha}"
    
    def test_missing_repo_skips_chunks(self):
        """Test that missing repo causes chunks to be skipped."""
        go_code = '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Should return empty list when repo is missing
        assert len(chunks) == 0, f"Should skip chunks when repo is missing: {len(chunks)}"
    
    def test_missing_file_sha_skips_chunks(self):
        """Test that missing file_sha causes chunks to be skipped."""
        go_code = '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Should return empty list when file_sha is missing
        assert len(chunks) == 0, f"Should skip chunks when file_sha is missing: {len(chunks)}"
    
    def test_normalize_go_path_function(self):
        """Test the normalize_go_path function."""
        # Test repo-relative path
        path = normalize_go_path(Path("/path/to/repo/internal/service/main.go"), "repo")
        assert path == "internal/service/main.go", f"Should extract repo-relative path: {path}"
        
        # Test path without repo
        path = normalize_go_path(Path("internal/service/main.go"), "")
        assert path == "internal/service/main.go", f"Should handle path without repo: {path}"
        
        # Test Windows-style path
        path = normalize_go_path(Path("internal\\service\\main.go"), "repo")
        assert path == "internal/service/main.go", f"Should convert to POSIX: {path}"
        
        # Test absolute path
        path = normalize_go_path(Path("/absolute/path/to/file.go"), "repo")
        assert path == "absolute/path/to/file.go", f"Should remove leading slash: {path}"
    
    def test_path_consistency_across_chunks(self):
        """Test that all chunks from the same file have the same path."""
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
            go_code, Path("internal/service/main.go"), "test-repo", "test-sha", "internal/service/main.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # All chunks should have the same path
        expected_path = "internal/service/main.go"
        for chunk in chunks:
            assert chunk.path == expected_path, f"All chunks should have same path: {chunk.path} vs {expected_path}"
    
    def test_chunk_id_includes_path_and_sha(self):
        """Test that chunk_id includes path and file_sha for uniqueness."""
        go_code = '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Check that chunk_id is unique and includes relevant info
        chunk_ids = set()
        for chunk in chunks:
            assert chunk.chunk_id not in chunk_ids, f"Chunk IDs should be unique: {chunk.chunk_id}"
            chunk_ids.add(chunk.chunk_id)
            
            # Chunk ID should be a reasonable length (SHA256 hex = 64 chars)
            assert len(chunk.chunk_id) == 64, f"Chunk ID should be 64 chars (SHA256): {len(chunk.chunk_id)}"
    
    def test_metadata_consistency_with_split_chunks(self):
        """Test that split chunks maintain consistent metadata."""
        go_code = '''package main

import (
    "context"
    "fmt"
    "time"
)

func (s *Service) LongMethod(ctx context.Context) error {
    // Part 1: Setup
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()
    
    // Part 2: Processing
    time.Sleep(100 * time.Millisecond)
    
    // Part 3: Cleanup
    fmt.Println("Done")
    
    return nil
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("internal/service/main.go"), "test-repo", "test-sha", "internal/service/main.go", 
            "gpt-4", 30, 60, logging.getLogger(__name__), parser  # Small limits to force splitting
        )
        
        # All chunks should have consistent metadata
        expected_repo = "test-repo"
        expected_sha = "test-sha"
        expected_path = "internal/service/main.go"
        
        for chunk in chunks:
            assert chunk.repo == expected_repo, f"All chunks should have same repo: {chunk.repo}"
            assert chunk.file_sha == expected_sha, f"All chunks should have same file_sha: {chunk.file_sha}"
            assert chunk.path == expected_path, f"All chunks should have same path: {chunk.path}"
