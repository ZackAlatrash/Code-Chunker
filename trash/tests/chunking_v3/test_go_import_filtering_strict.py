#!/usr/bin/env python3
"""
Test strict import filtering for Go chunks.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import extract_chunk_imports_strict, build_symbol_to_import_map


class TestGoImportFilteringStrict:
    """Test strict import filtering for Go chunks."""
    
    def test_strict_import_filtering_no_unused_imports(self):
        """Test that unused imports are not included in imports_used."""
        chunk_text = """
        ctx := context.Background()
        now := time.Now()
        fmt.Println("Hello")
        """
        
        symbol_map = build_symbol_to_import_map(
            ['context', 'time', 'fmt', 'os', 'strings'], {}
        )
        
        imports_used = extract_chunk_imports_strict(chunk_text, symbol_map)
        
        # Should only include context, time, and fmt
        assert 'context' in imports_used
        assert 'time' in imports_used
        assert 'fmt' in imports_used
        assert 'os' not in imports_used  # Not used in chunk
        assert 'strings' not in imports_used  # Not used in chunk
    
    def test_strict_import_filtering_with_aliases(self):
        """Test strict filtering respects import aliases."""
        chunk_text = """
        ctx, span := xotel.Tracer.Start(ctx, "service:forecast-location")
        defer span.End()
        
        log.Error(ctx, "Cannot get mapping", zap.Error(err))
        """
        
        symbol_map = build_symbol_to_import_map(
            ['go.opentelemetry.io/otel/trace', 'go.uber.org/zap', 'go.impalastudios.com/log'],
            {'xotel': 'go.impalastudios.com/otel'}
        )
        
        imports_used = extract_chunk_imports_strict(chunk_text, symbol_map)
        
        # Should include aliased import
        assert 'go.impalastudios.com/otel' in imports_used  # xotel alias
        assert 'go.uber.org/zap' in imports_used  # zap used
        assert 'go.impalastudios.com/log' in imports_used  # log used
        assert 'go.opentelemetry.io/otel/trace' not in imports_used  # Not directly used
    
    def test_strict_import_filtering_qualified_identifiers(self):
        """Test strict filtering with qualified identifiers."""
        chunk_text = """
        ctx := context.Background()
        loc, err := time.LoadLocation("UTC")
        duration := time.Duration(30) * time.Minute
        """
        
        symbol_map = build_symbol_to_import_map(
            ['context', 'time'], {}
        )
        
        imports_used = extract_chunk_imports_strict(chunk_text, symbol_map)
        
        # Should include both context and time
        assert 'context' in imports_used
        assert 'time' in imports_used
    
    def test_strict_import_filtering_interface_chunk(self):
        """Test that interface chunks only include actually used imports."""
        chunk_text = """
        providerClient interface {
            GetForecastForLocation(ctx context.Context, id int, loc *time.Location) (*Forecast, error)
        }
        """
        
        symbol_map = build_symbol_to_import_map(
            ['context', 'time', 'golang.org/x/sync/singleflight'], {}
        )
        
        imports_used = extract_chunk_imports_strict(chunk_text, symbol_map)
        
        # Should only include context and time, not singleflight
        assert 'context' in imports_used
        assert 'time' in imports_used
        assert 'golang.org/x/sync/singleflight' not in imports_used  # Not used in this chunk
    
    def test_strict_import_filtering_struct_chunk(self):
        """Test that struct chunks only include actually used imports."""
        chunk_text = """
        type Service struct {
            sf                      singleflight.Group
            provider                providerClient
            cacheExpirationDuration time.Duration
        }
        """
        
        symbol_map = build_symbol_to_import_map(
            ['golang.org/x/sync/singleflight', 'time', 'context'], {}
        )
        
        imports_used = extract_chunk_imports_strict(chunk_text, symbol_map)
        
        # Should include singleflight and time, not context
        assert 'golang.org/x/sync/singleflight' in imports_used
        assert 'time' in imports_used
        assert 'context' not in imports_used  # Not used in this chunk
    
    def test_strict_import_filtering_method_chunk(self):
        """Test that method chunks only include actually used imports."""
        chunk_text = """
        func (s *Service) GetForecastForLocation(ctx context.Context, id int) (*Forecast, error) {
            ctx, span := xotel.Tracer.Start(ctx, "service:forecast-location")
            defer span.End()
            
            mapping, err := s.mappings.Get(ctx, id)
            if err != nil {
                log.Error(ctx, "Cannot get mapping", zap.Error(err))
                return nil, err
            }
        }
        """
        
        symbol_map = build_symbol_to_import_map(
            ['context', 'go.opentelemetry.io/otel/trace', 'go.uber.org/zap', 'go.impalastudios.com/log'],
            {'xotel': 'go.impalastudios.com/otel'}
        )
        
        imports_used = extract_chunk_imports_strict(chunk_text, symbol_map)
        
        # Should include all used imports
        assert 'context' in imports_used
        assert 'go.impalastudios.com/otel' in imports_used  # xotel alias
        assert 'go.uber.org/zap' in imports_used
        assert 'go.impalastudios.com/log' in imports_used
        assert 'go.opentelemetry.io/otel/trace' not in imports_used  # Not directly used
    
    def test_build_symbol_to_import_map_common_packages(self):
        """Test that symbol mapping includes common package symbols."""
        symbol_map = build_symbol_to_import_map(
            ['context', 'time', 'errors', 'fmt', 'strings'], {}
        )
        
        # Test context symbols
        assert symbol_map['context'] == 'context'
        assert symbol_map['Context'] == 'context'
        
        # Test time symbols
        assert symbol_map['time'] == 'time'
        assert symbol_map['Time'] == 'time'
        assert symbol_map['Location'] == 'time'
        assert symbol_map['Duration'] == 'time'
        
        # Test error symbols
        assert symbol_map['error'] == 'errors'
        
        # Test fmt symbols
        assert symbol_map['fmt'] == 'fmt'
        assert symbol_map['Printf'] == 'fmt'
        assert symbol_map['Sprintf'] == 'fmt'
        
        # Test strings symbols
        assert symbol_map['strings'] == 'strings'
        assert symbol_map['Trim'] == 'strings'
        assert symbol_map['Split'] == 'strings'


if __name__ == '__main__':
    pytest.main([__file__])
