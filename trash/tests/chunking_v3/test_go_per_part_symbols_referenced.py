#!/usr/bin/env python3
"""
Test that Go chunks have proper symbols_referenced populated per part.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_go_file, extract_go_symbols_referenced_strict
from CodeParser import CodeParser
import logging


class TestGoPerPartSymbolsReferenced:
    """Test that symbols_referenced is properly populated for each part."""
    
    def test_qualified_identifiers_detection(self):
        """Test detection of qualified identifiers like package.Symbol."""
        chunk_text = '''
        ctx, span := xotel.Tracer.Start(ctx, "service:forecast")
        span.SetAttributes(attribute.Int("location_id", id))
        log.Error(ctx, "Error", zap.Int("id", id), zap.Error(err))
        '''
        
        symbol_map = {
            'xotel': 'go.impalastudios.com/otel',
            'attribute': 'go.opentelemetry.io/otel/attribute',
            'zap': 'go.uber.org/zap'
        }
        
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should detect qualified identifiers
        assert any('xotel.Tracer' in s for s in symbols), f"Should detect xotel.Tracer: {symbols}"
        assert any('attribute.Int' in s for s in symbols), f"Should detect attribute.Int: {symbols}"
        assert any('zap.Int' in s for s in symbols), f"Should detect zap.Int: {symbols}"
        assert any('zap.Error' in s for s in symbols), f"Should detect zap.Error: {symbols}"
    
    def test_local_type_references(self):
        """Test detection of local type references."""
        chunk_text = '''
        var item expirableCacheItem
        var forecast *Forecast
        var mapping *Mapping
        var service Service
        '''
        
        symbol_map = {}
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should detect local types
        assert 'expirableCacheItem' in symbols, f"Should detect expirableCacheItem: {symbols}"
        assert 'Forecast' in symbols, f"Should detect Forecast: {symbols}"
        assert 'Mapping' in symbols, f"Should detect Mapping: {symbols}"
        assert 'Service' in symbols, f"Should detect Service: {symbols}"
    
    def test_builtin_types_excluded(self):
        """Test that builtin types are excluded unless they appear as selectors."""
        chunk_text = '''
        var name string
        var age int
        var active bool
        var data []byte
        var err error
        var ctx context.Context
        '''
        
        symbol_map = {'context': 'context'}
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Builtin types should be excluded
        assert 'string' not in symbols, f"Builtin string should be excluded: {symbols}"
        assert 'int' not in symbols, f"Builtin int should be excluded: {symbols}"
        assert 'bool' not in symbols, f"Builtin bool should be excluded: {symbols}"
        assert 'byte' not in symbols, f"Builtin byte should be excluded: {symbols}"
        
        # But context.Context should be included as it's a selector
        assert 'context.Context' in symbols, f"context.Context should be included: {symbols}"
        assert 'error' in symbols, f"error should be included: {symbols}"
    
    def test_method_calls_and_selectors(self):
        """Test detection of method calls and selectors."""
        chunk_text = '''
        result, err, _ := s.sf.Do(key, func() (interface{}, error) {
            i, err := s.cache.Get(key)
            if err == nil {
                var item expirableCacheItem
                _ = json.Unmarshal(i.Value, &item)
                return item.Forecast, nil
            }
            return nil, err
        })
        '''
        
        symbol_map = {
            'json': 'encoding/json',
            'singleflight': 'golang.org/x/sync/singleflight'
        }
        
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should detect method calls and selectors
        assert 'json.Unmarshal' in symbols, f"Should detect json.Unmarshal: {symbols}"
        assert 'singleflight.Group' in symbols, f"Should detect singleflight.Group: {symbols}"
        assert 'expirableCacheItem' in symbols, f"Should detect expirableCacheItem: {symbols}"
        assert 'Forecast' in symbols, f"Should detect Forecast: {symbols}"
    
    def test_per_part_symbols_in_chunks(self):
        """Test that each chunk part has appropriate symbols_referenced."""
        go_code = '''package foreca

import (
    "context"
    "time"
    "go.uber.org/zap"
    "go.opentelemetry.io/otel/trace"
    "encoding/json"
    "golang.org/x/sync/singleflight"
)

type Service struct {
    sf    singleflight.Group
    cache cacheClient
}

func (s *Service) GetForecast(ctx context.Context, id int) (*Forecast, error) {
    // Part 1: OTEL span setup
    ctx, span := xotel.Tracer.Start(ctx, "service:forecast")
    defer span.End()
    span.SetAttributes(attribute.Int("location_id", id))
    
    // Part 2: Cache lookup with singleflight
    result, err, _ := s.sf.Do(key, func() (interface{}, error) {
        i, err := s.cache.Get(key)
        if err == nil {
            var item expirableCacheItem
            _ = json.Unmarshal(i.Value, &item)
            return item.Forecast, nil
        }
        return nil, err
    })
    
    // Part 3: Error handling and logging
    if err != nil {
        span.RecordError(err)
        log.Error(ctx, "Failed to get forecast", zap.Int("id", id), zap.Error(err))
        return nil, err
    }
    
    return result.(*Forecast), nil
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("service.go"), "test-repo", "test-sha", "service.go", 
            "gpt-4", 40, 80, logging.getLogger(__name__), parser  # Small limits to force splitting
        )
        
        # Find method chunks
        method_chunks = [c for c in chunks if "GetForecast" in c.ast_path and "#part" in c.ast_path]
        
        if len(method_chunks) > 1:
            for chunk in method_chunks:
                symbols = chunk.symbols_referenced
                
                if "otel_span" in chunk.ast_path:
                    # Should have OTEL-related symbols
                    assert any("xotel" in s or "attribute" in s or "trace" in s for s in symbols), \
                        f"OTEL part should have OTEL symbols: {symbols}"
                
                if "singleflight_cache" in chunk.ast_path:
                    # Should have singleflight and JSON symbols
                    assert any("singleflight" in s or "json" in s for s in symbols), \
                        f"Singleflight part should have singleflight/JSON symbols: {symbols}"
                
                if "error_handling" in chunk.ast_path or "logging" in chunk.ast_path:
                    # Should have logging symbols
                    assert any("zap" in s for s in symbols), \
                        f"Error/logging part should have zap symbols: {symbols}"
    
    def test_interface_and_struct_types(self):
        """Test detection of interface and struct type references."""
        chunk_text = '''
        type Service struct {
            provider providerClient
            cache    cacheClient
        }
        
        type providerClient interface {
            GetForecast(ctx context.Context, id int) (*Forecast, error)
        }
        
        var s Service
        var p providerClient
        '''
        
        symbol_map = {'context': 'context'}
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should detect type references
        assert 'Service' in symbols, f"Should detect Service: {symbols}"
        assert 'providerClient' in symbols, f"Should detect providerClient: {symbols}"
        assert 'cacheClient' in symbols, f"Should detect cacheClient: {symbols}"
        assert 'Forecast' in symbols, f"Should detect Forecast: {symbols}"
        assert 'context.Context' in symbols, f"Should detect context.Context: {symbols}"
    
    def test_function_calls_and_returns(self):
        """Test detection of function calls and return types."""
        chunk_text = '''
        func NewService(provider providerClient, cache cacheClient) *Service {
            return &Service{
                provider: provider,
                cache:    cache,
            }
        }
        
        func (s *Service) GetForecast(id int) (*Forecast, error) {
            forecast, err := s.provider.GetForecast(ctx, id)
            if err != nil {
                return nil, errors.Wrap(err, "failed")
            }
            return forecast, nil
        }
        '''
        
        symbol_map = {'errors': 'github.com/pkg/errors'}
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should detect function calls and types
        assert 'NewService' in symbols, f"Should detect NewService: {symbols}"
        assert 'Service' in symbols, f"Should detect Service: {symbols}"
        assert 'providerClient' in symbols, f"Should detect providerClient: {symbols}"
        assert 'cacheClient' in symbols, f"Should detect cacheClient: {symbols}"
        assert 'Forecast' in symbols, f"Should detect Forecast: {symbols}"
        assert 'errors.Wrap' in symbols, f"Should detect errors.Wrap: {symbols}"
