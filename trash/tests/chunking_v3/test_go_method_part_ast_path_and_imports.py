#!/usr/bin/env python3
"""
Test that Go method parts have proper ast_path naming and per-part imports.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_go_file, detect_block_label, find_logical_split_point
from CodeParser import CodeParser
import logging


class TestGoMethodPartAstPathAndImports:
    """Test method splitting with proper ast_path naming and imports."""
    
    def test_method_part_ast_path_naming(self):
        """Test that method parts have proper ast_path with part numbers and labels."""
        go_code = '''package foreca

import (
    "context"
    "time"
    "go.uber.org/zap"
    "go.opentelemetry.io/otel/trace"
)

type Service struct {
    cache cacheClient
}

func (s *Service) GetForecastForLocation(ctx context.Context, id int) (*Forecast, error) {
    // Cache lookup
    item, err := s.cache.Get(key)
    if err == nil {
        return item.Forecast, nil
    }
    
    // Provider call
    forecast, err := s.provider.GetForecast(ctx, id)
    if err != nil {
        return nil, err
    }
    
    // Cache the result
    s.cache.Set(key, forecast)
    return forecast, nil
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("service.go"), "test-repo", "test-sha", "service.go", 
            "gpt-4", 50, 100, logging.getLogger(__name__), parser  # Small limits to force splitting
        )
        
        # Find method chunks
        method_chunks = [c for c in chunks if "GetForecastForLocation" in c.ast_path]
        
        if len(method_chunks) > 1:  # If splitting occurred
            for i, chunk in enumerate(method_chunks):
                assert "#part" in chunk.ast_path, f"Method part should have #part in ast_path: {chunk.ast_path}"
                assert f"part{i+1}" in chunk.ast_path, f"Should have part{i+1} in ast_path: {chunk.ast_path}"
                assert "Service" in chunk.ast_path, f"Should include receiver type: {chunk.ast_path}"
    
    def test_block_label_detection(self):
        """Test block label detection for different code patterns."""
        # Test cache lookup detection
        cache_text = "item, err := s.cache.Get(key)\nif err == nil {\n    return item.Forecast, nil\n}"
        label = detect_block_label(cache_text, 0)
        assert label == "cache_lookup"
        
        # Test singleflight detection
        sf_text = "result, err, _ := s.sf.Do(key, func() (interface{}, error) {\n    return s.provider.GetForecast(ctx, id)\n})"
        label = detect_block_label(sf_text, 0)
        assert label == "singleflight_cache"
        
        # Test JSON operations
        json_text = "var item expirableCacheItem\n_ = json.Unmarshal(data, &item)"
        label = detect_block_label(json_text, 0)
        assert label == "json_ops"
        
        # Test timezone loading
        tz_text = "loc, err := time.LoadLocation(mapping.Timezone)\nif err != nil {\n    return nil, err\n}"
        label = detect_block_label(tz_text, 0)
        assert label == "timezone_load"
        
        # Test provider calls
        provider_text = "forecast, err := s.provider.GetForecastForLocation(ctx, id, loc)"
        label = detect_block_label(provider_text, 0)
        assert label == "provider_call"
        
        # Test rate limiting
        rate_text = "if errors.Is(err, ErrRequestThrottled) {\n    return cached, nil\n}"
        label = detect_block_label(rate_text, 0)
        assert label == "rate_limit"
        
        # Test error handling
        error_text = "if err != nil {\n    span.RecordError(err)\n    return nil, errors.Wrap(err, \"failed\")\n}"
        label = detect_block_label(error_text, 0)
        assert label == "error_handling"
        
        # Test OTEL spans
        span_text = "span.SetAttributes(attribute.Bool(\"cache_hit\", true))\nspan.RecordError(err)"
        label = detect_block_label(span_text, 0)
        assert label == "otel_span"
        
        # Test logging
        log_text = "log.Error(ctx, \"Failed to get forecast\", zap.Error(err))"
        label = detect_block_label(log_text, 0)
        assert label == "logging"
    
    def test_logical_split_point_detection(self):
        """Test logical split point detection."""
        lines = [
            "func (s *Service) GetForecast(ctx context.Context) {",
            "    // Cache lookup",
            "    item, err := s.cache.Get(key)",
            "    if err == nil {",
            "        return item.Forecast, nil",
            "    }",
            "    ",
            "    // Provider call",
            "    forecast, err := s.provider.GetForecast(ctx)",
            "    if err != nil {",
            "        return nil, err",
            "    }",
            "    ",
            "    return forecast, nil",
            "}"
        ]
        
        # Test splitting at logical boundaries
        split_point = find_logical_split_point(lines, 0, 50)
        assert split_point > 0, "Should find a split point"
        assert split_point < len(lines), "Split point should be within bounds"
    
    def test_per_part_imports_recomputation(self):
        """Test that imports are recomputed for each part."""
        go_code = '''package foreca

import (
    "context"
    "time"
    "go.uber.org/zap"
    "go.opentelemetry.io/otel/trace"
    "encoding/json"
)

func (s *Service) GetForecast(ctx context.Context) (*Forecast, error) {
    // Part 1: Cache with context
    ctx, span := trace.Start(ctx, "cache")
    defer span.End()
    
    // Part 2: JSON operations
    var item expirableCacheItem
    _ = json.Unmarshal(data, &item)
    
    // Part 3: Logging
    log.Info(ctx, "Got forecast", zap.String("id", "123"))
    
    return item.Forecast, nil
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("service.go"), "test-repo", "test-sha", "service.go", 
            "gpt-4", 30, 60, logging.getLogger(__name__), parser  # Very small limits to force splitting
        )
        
        # Find method chunks
        method_chunks = [c for c in chunks if "GetForecast" in c.ast_path and "#part" in c.ast_path]
        
        if len(method_chunks) > 1:
            # Check that each part has appropriate imports
            for chunk in method_chunks:
                if "otel_span" in chunk.ast_path or "cache_lookup" in chunk.ast_path:
                    # Should have trace import
                    assert any("otel" in imp for imp in chunk.imports_used), f"OTEL part should have trace import: {chunk.imports_used}"
                
                if "json_ops" in chunk.ast_path:
                    # Should have json import
                    assert any("json" in imp for imp in chunk.imports_used), f"JSON part should have json import: {chunk.imports_used}"
                
                if "logging" in chunk.ast_path:
                    # Should have zap import
                    assert any("zap" in imp for imp in chunk.imports_used), f"Logging part should have zap import: {chunk.imports_used}"
    
    def test_ast_path_receiver_formatting(self):
        """Test that ast_path properly formats receiver types."""
        go_code = '''package main

type Service struct{}

func (s *Service) Method1() {}
func (s Service) Method2() {}
func (*Service) Method3() {}
func (Service) Method4() {}
'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # Check method ast_paths
        method_chunks = [c for c in chunks if c.ast_path.startswith("go:method:")]
        
        for chunk in method_chunks:
            if "Method1" in chunk.ast_path:
                assert "(*Service)" in chunk.ast_path, f"Pointer receiver should be (*Service): {chunk.ast_path}"
            elif "Method2" in chunk.ast_path:
                assert "(Service)" in chunk.ast_path, f"Value receiver should be (Service): {chunk.ast_path}"
