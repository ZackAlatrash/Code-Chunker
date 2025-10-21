#!/usr/bin/env python3
"""
Test that rel_path is removed from Go chunks.
"""

import pytest
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_go_file, normalize_go_path
from src.CodeParser import CodeParser


class TestGoDropRelPath:
    """Test that rel_path is removed from Go chunks."""
    
    def test_normalize_go_path_repo_relative(self):
        """Test that normalize_go_path returns repo-relative paths."""
        # Test with repo in path
        file_path = Path("/Users/user/repos/foreca-service/internal/foreca/service.go")
        repo = "foreca-service"
        
        normalized = normalize_go_path(file_path, repo)
        assert normalized == "internal/foreca/service.go"
        
        # Test with different repo structure
        file_path = Path("/home/user/projects/weather-proxy/cmd/server/main.go")
        repo = "weather-proxy"
        
        normalized = normalize_go_path(file_path, repo)
        assert normalized == "cmd/server/main.go"
    
    def test_normalize_go_path_posix_style(self):
        """Test that normalize_go_path returns POSIX-style paths."""
        # Test with Windows-style path
        file_path = Path("C:\\Users\\user\\repos\\foreca-service\\internal\\foreca\\service.go")
        repo = "foreca-service"
        
        normalized = normalize_go_path(file_path, repo)
        assert normalized == "internal/foreca/service.go"
        assert "\\" not in normalized  # Should be POSIX-style
    
    def test_normalize_go_path_no_leading_slash(self):
        """Test that normalize_go_path returns paths without leading slash."""
        file_path = Path("/Users/user/repos/foreca-service/internal/foreca/service.go")
        repo = "foreca-service"
        
        normalized = normalize_go_path(file_path, repo)
        assert not normalized.startswith("/")
        assert normalized == "internal/foreca/service.go"
    
    def test_normalize_go_path_fallback(self):
        """Test that normalize_go_path falls back gracefully."""
        # Test with no repo match
        file_path = Path("/some/absolute/path/file.go")
        repo = "nonexistent-repo"
        
        normalized = normalize_go_path(file_path, repo)
        assert normalized == "some/absolute/path/file.go"  # Removes leading slash
    
    def test_go_chunks_use_normalized_path(self):
        """Test that Go chunks use normalized paths."""
        go_content = """
package foreca

import "context"

type Service struct {
    cacheExpirationDuration time.Duration
}
"""
        
        parser = CodeParser()
        logger = logging.getLogger(__name__)
        chunks = chunk_go_file(
            go_content, Path("internal/foreca/service.go"), "foreca-service", 
            "sha123", "internal/foreca/service.go", "gpt-4", 150, 380, logger, parser
        )
        
        # All chunks should use normalized paths
        for chunk in chunks:
            assert chunk.path == "internal/foreca/service.go"
            assert not chunk.path.startswith("/")  # No leading slash
            assert "\\" not in chunk.path  # POSIX-style
    
    def test_go_chunks_path_consistency(self):
        """Test that all Go chunks have consistent path format."""
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
        chunks = chunk_go_file(
            go_content, Path("cmd/server/main.go"), "weather-proxy", 
            "sha123", "cmd/server/main.go", "gpt-4", 150, 380, None, parser
        )
        
        # All chunks should have the same normalized path
        expected_path = "cmd/server/main.go"
        for chunk in chunks:
            assert chunk.path == expected_path
            assert chunk.rel_path == expected_path  # Should be same as path
    
    def test_go_chunks_no_absolute_paths(self):
        """Test that Go chunks don't contain absolute paths."""
        go_content = """
package foreca

import "context"

type Service struct {
    cacheExpirationDuration time.Duration
}
"""
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_content, Path("/absolute/path/to/service.go"), "foreca-service", 
            "sha123", "/absolute/path/to/service.go", "gpt-4", 150, 380, None, parser
        )
        
        # All chunks should have relative paths
        for chunk in chunks:
            assert not chunk.path.startswith("/")  # No absolute paths
            assert not chunk.path.startswith("C:")  # No Windows absolute paths
            assert not chunk.path.startswith("\\")  # No Windows paths
    
    def test_go_chunks_path_with_different_repos(self):
        """Test that Go chunks handle different repo structures correctly."""
        test_cases = [
            ("foreca-service", "internal/foreca/service.go"),
            ("weather-proxy", "cmd/server/main.go"),
            ("api-gateway", "pkg/middleware/auth.go"),
            ("user-service", "internal/handlers/user.go"),
        ]
        
        go_content = """
package main

import "context"

type Service struct {
    cacheExpirationDuration time.Duration
}
"""
        
        parser = CodeParser()
        logger = logging.getLogger(__name__)
        
        for repo, expected_path in test_cases:
            chunks = chunk_go_file(
                go_content, Path(f"/some/path/{repo}/{expected_path}"), repo, 
                "sha123", f"/some/path/{repo}/{expected_path}", "gpt-4", 150, 380, logger, parser
            )
            
            # All chunks should have the expected normalized path
            for chunk in chunks:
                assert chunk.path == expected_path
                assert chunk.rel_path == expected_path
    
    def test_go_chunks_path_edge_cases(self):
        """Test that Go chunks handle edge cases in path normalization."""
        go_content = """
package main

import "context"

type Service struct {
    cacheExpirationDuration time.Duration
}
"""
        
        parser = CodeParser()
        logger = logging.getLogger(__name__)
        
        # Test with repo name in middle of path
        chunks = chunk_go_file(
            go_content, Path("/path/foreca-service/src/service.go"), "foreca-service", 
            "sha123", "/path/foreca-service/src/service.go", "gpt-4", 150, 380, logger, parser
        )
        
        for chunk in chunks:
            assert chunk.path == "src/service.go"
            assert not chunk.path.startswith("/")
        
        # Test with nested repo structure
        chunks = chunk_go_file(
            go_content, Path("/home/user/projects/foreca-service/cmd/server/main.go"), "foreca-service", 
            "sha123", "/home/user/projects/foreca-service/cmd/server/main.go", "gpt-4", 150, 380, logger, parser
        )
        
        for chunk in chunks:
            assert chunk.path == "cmd/server/main.go"
            assert not chunk.path.startswith("/")


if __name__ == '__main__':
    pytest.main([__file__])
