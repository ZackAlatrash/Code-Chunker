#!/usr/bin/env python3
"""
Test that Go symbols_referenced is properly populated.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import extract_go_symbols_referenced_strict, build_symbol_to_import_map


class TestGoSymbolsReferencedPopulated:
    """Test that Go symbols_referenced is properly populated."""
    
    def test_qualified_identifiers_extracted(self):
        """Test that qualified identifiers are extracted as symbols_referenced."""
        chunk_text = """
        ctx := context.Background()
        now := time.Now()
        loc, err := time.LoadLocation("UTC")
        duration := time.Duration(30) * time.Minute
        """
        
        symbol_map = build_symbol_to_import_map(
            ['context', 'time'], {}
        )
        
        symbols_referenced = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should include qualified identifiers
        assert 'context.Background' in symbols_referenced
        assert 'time.Now' in symbols_referenced
        assert 'time.LoadLocation' in symbols_referenced
        assert 'time.Duration' in symbols_referenced
        assert 'time.Minute' in symbols_referenced
    
    def test_type_references_extracted(self):
        """Test that type references are extracted."""
        chunk_text = """
        type Service struct {
            sf                      singleflight.Group
            provider                providerClient
            cacheExpirationDuration time.Duration
        }
        """
        
        symbol_map = build_symbol_to_import_map(
            ['golang.org/x/sync/singleflight', 'time'], {}
        )
        
        symbols_referenced = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should include type references
        assert 'singleflight.Group' in symbols_referenced
        assert 'time.Duration' in symbols_referenced
        assert 'providerClient' in symbols_referenced  # Local interface
    
    def test_interface_method_signatures_extracted(self):
        """Test that interface method signatures are extracted."""
        chunk_text = """
        providerClient interface {
            GetForecastForLocation(ctx context.Context, id int, loc *time.Location) (*Forecast, error)
        }
        """
        
        symbol_map = build_symbol_to_import_map(
            ['context', 'time'], {}
        )
        
        symbols_referenced = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should include type references from method signatures
        assert 'context.Context' in symbols_referenced
        assert 'time.Location' in symbols_referenced
        assert 'Forecast' in symbols_referenced  # Local type
        assert 'error' in symbols_referenced  # Built-in type
    
    def test_function_calls_extracted(self):
        """Test that function calls are extracted."""
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
            ['context', 'go.uber.org/zap', 'go.impalastudios.com/log'],
            {'xotel': 'go.impalastudios.com/otel'}
        )
        
        symbols_referenced = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should include function calls
        assert 'context.Context' in symbols_referenced
        assert 'xotel.Tracer' in symbols_referenced
        assert 'zap.Error' in symbols_referenced
        assert 'log.Error' in symbols_referenced
        assert 'Forecast' in symbols_referenced  # Return type
        assert 'error' in symbols_referenced  # Return type
    
    def test_pointer_types_extracted(self):
        """Test that pointer types are extracted."""
        chunk_text = """
        func NewService(provider *providerClient, mappings *mappingsRepository) *Service {
            return &Service{
                provider: provider,
                mappings: mappings,
            }
        }
        """
        
        symbol_map = build_symbol_to_import_map([], {})
        
        symbols_referenced = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should include pointer types
        assert 'providerClient' in symbols_referenced
        assert 'mappingsRepository' in symbols_referenced
        assert 'Service' in symbols_referenced
    
    def test_array_slice_types_extracted(self):
        """Test that array/slice types are extracted."""
        chunk_text = """
        func ProcessItems(items []Item, configs []Config) []Result {
            results := make([]Result, 0, len(items))
            for _, item := range items {
                result := processItem(item, configs[0])
                results = append(results, result)
            }
            return results
        }
        """
        
        symbol_map = build_symbol_to_import_map([], {})
        
        symbols_referenced = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should include slice types
        assert 'Item' in symbols_referenced
        assert 'Config' in symbols_referenced
        assert 'Result' in symbols_referenced
    
    def test_struct_literals_extracted(self):
        """Test that struct literals are extracted."""
        chunk_text = """
        service := &Service{
            sf:                      singleflight.Group{},
            provider:                provider,
            cacheExpirationDuration: time.Duration(30) * time.Minute,
        }
        """
        
        symbol_map = build_symbol_to_import_map(
            ['golang.org/x/sync/singleflight', 'time'], {}
        )
        
        symbols_referenced = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should include struct types
        assert 'Service' in symbols_referenced
        assert 'singleflight.Group' in symbols_referenced
        assert 'time.Duration' in symbols_referenced
        assert 'time.Minute' in symbols_referenced
    
    def test_error_type_special_case(self):
        """Test that error type is handled as special case."""
        chunk_text = """
        func DoSomething() error {
            if err := someFunction(); err != nil {
                return err
            }
            return nil
        }
        """
        
        symbol_map = build_symbol_to_import_map([], {})
        
        symbols_referenced = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should include error as special case
        assert 'error' in symbols_referenced
    
    def test_builtin_types_excluded_unless_selectors(self):
        """Test that builtin types are excluded unless they appear as selectors."""
        chunk_text = """
        func ProcessData(data string, count int, flag bool) (string, int, error) {
            if flag {
                return data, count, nil
            }
            return "", 0, errors.New("failed")
        }
        """
        
        symbol_map = build_symbol_to_import_map(['errors'], {})
        
        symbols_referenced = extract_go_symbols_referenced_strict(chunk_text, symbol_map)
        
        # Should include error (special case) but not other builtins
        assert 'error' in symbols_referenced
        # string, int, bool should not be included as they're builtins
        assert 'string' not in symbols_referenced
        assert 'int' not in symbols_referenced
        assert 'bool' not in symbols_referenced


if __name__ == '__main__':
    pytest.main([__file__])
