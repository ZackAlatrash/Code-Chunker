"""
Tests for LLM enrichment functions.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from nwe_v3_enrich.llm_qwen import (
    digest_for_cache,
    summarize_chunk_qwen,
    LLMCache,
)


class TestCacheDigest:
    """Test cache digest generation."""
    
    def test_digest_deterministic(self):
        """Test that digest is deterministic for same input."""
        chunk = {
            "ast_path": "go:function:TestFunc",
            "text": "func TestFunc() {}",
            "imports_used_minimal": ["fmt"],
            "symbols_referenced_strict": ["TestFunc"]
        }
        
        digest1 = digest_for_cache(chunk, "v1")
        digest2 = digest_for_cache(chunk, "v1")
        assert digest1 == digest2
    
    def test_digest_different_versions(self):
        """Test that different prompt versions produce different digests."""
        chunk = {
            "ast_path": "go:function:TestFunc",
            "text": "func TestFunc() {}",
            "imports_used_minimal": ["fmt"],
            "symbols_referenced_strict": ["TestFunc"]
        }
        
        digest1 = digest_for_cache(chunk, "v1")
        digest2 = digest_for_cache(chunk, "v2")
        assert digest1 != digest2


class TestLLMEnrichment:
    """Test LLM enrichment functionality."""
    
    @patch('urllib.request.urlopen')
    def test_successful_llm_enrichment(self, mock_urlopen):
        """Test successful LLM enrichment."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "response": json.dumps({
                "summary_llm": "This function processes weather data and returns a forecast response.",
                "keywords_llm": ["weather", "forecast", "data", "process", "response", "function", "service", "api", "temperature", "humidity"]
            })
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        chunk = {
            "language": "go",
            "rel_path": "service.go",
            "ast_path": "go:function:ProcessWeather",
            "package": "main",
            "imports_used_minimal": ["fmt", "time"],
            "symbols_referenced_strict": ["WeatherData", "ForecastResponse"],
            "text": "func ProcessWeather(data WeatherData) ForecastResponse { return ForecastResponse{} }"
        }
        
        result = summarize_chunk_qwen(chunk, "qwen2.5-coder:7b-instruct", "v1")
        
        assert "summary_llm" in result
        assert "keywords_llm" in result
        assert isinstance(result["keywords_llm"], list)
        assert 8 <= len(result["keywords_llm"]) <= 14
        assert all(isinstance(kw, str) for kw in result["keywords_llm"])
    
    @patch('urllib.request.urlopen')
    def test_llm_enrichment_failure(self, mock_urlopen):
        """Test LLM enrichment failure handling."""
        # Mock failure
        mock_urlopen.side_effect = Exception("Connection failed")
        
        chunk = {
            "language": "go",
            "rel_path": "service.go",
            "ast_path": "go:function:TestFunc",
            "package": "main",
            "imports_used_minimal": [],
            "symbols_referenced_strict": [],
            "text": "func TestFunc() {}"
        }
        
        result = summarize_chunk_qwen(chunk, "qwen2.5-coder:7b-instruct", "v1")
        
        assert "summary_llm" in result
        assert "keywords_llm" in result
        assert "unavailable" in result["summary_llm"].lower()
    
    @patch('urllib.request.urlopen')
    def test_invalid_json_response(self, mock_urlopen):
        """Test handling of invalid JSON response."""
        # Mock invalid JSON response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "response": "This is not valid JSON"
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        chunk = {
            "language": "go",
            "rel_path": "service.go",
            "ast_path": "go:function:TestFunc",
            "package": "main",
            "imports_used_minimal": [],
            "symbols_referenced_strict": [],
            "text": "func TestFunc() {}"
        }
        
        result = summarize_chunk_qwen(chunk, "qwen2.5-coder:7b-instruct", "v1")
        
        # Should retry and eventually return fallback
        assert "summary_llm" in result
        assert "keywords_llm" in result


class TestLLMCache:
    """Test LLM cache functionality."""
    
    def test_cache_operations(self):
        """Test cache get/set operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "test_cache.sqlite")
            cache = LLMCache(cache_path)
            
            # Test cache miss
            result = cache.get("nonexistent_key")
            assert result is None
            
            # Test cache set and get
            test_result = {
                "summary_llm": "Test summary",
                "keywords_llm": ["test", "keywords", "cache"]
            }
            
            cache.set("test_key", "chunk_1", "file_sha_123", "v1", "go:function:Test", test_result)
            
            cached_result = cache.get("test_key")
            assert cached_result is not None
            assert cached_result["summary_llm"] == "Test summary"
            assert cached_result["keywords_llm"] == ["test", "keywords", "cache"]
    
    def test_cache_key_uniqueness(self):
        """Test that different cache keys produce different results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "test_cache.sqlite")
            cache = LLMCache(cache_path)
            
            result1 = {"summary_llm": "Summary 1", "keywords_llm": ["key1"]}
            result2 = {"summary_llm": "Summary 2", "keywords_llm": ["key2"]}
            
            cache.set("key1", "chunk_1", "sha1", "v1", "ast1", result1)
            cache.set("key2", "chunk_2", "sha2", "v1", "ast2", result2)
            
            cached1 = cache.get("key1")
            cached2 = cache.get("key2")
            
            assert cached1["summary_llm"] == "Summary 1"
            assert cached2["summary_llm"] == "Summary 2"
            assert cached1 != cached2
