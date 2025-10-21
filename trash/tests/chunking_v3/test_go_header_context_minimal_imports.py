#!/usr/bin/env python3
"""
Test Go header context with minimal imports.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import build_go_minimal_header_context, extract_chunk_imports_with_aliases


class TestGoHeaderContextMinimalImports:
    """Test Go header context with minimal imports."""
    
    def test_single_import_minimal_header(self):
        """Test minimal header context with single import."""
        header = build_go_minimal_header_context(
            'foreca', ['context'], 'function_declaration', {}
        )
        
        expected = 'package foreca\nimport "context"'
        assert header == expected
    
    def test_multiple_imports_minimal_header(self):
        """Test minimal header context with multiple imports."""
        imports = ['context', 'time', 'fmt']
        header = build_go_minimal_header_context(
            'foreca', imports, 'function_declaration', {}
        )
        
        expected = 'package foreca\nimport (\n\t"context"\n\t"time"\n\t"fmt"\n)'
        assert header == expected
    
    def test_method_with_receiver_comment(self):
        """Test minimal header context for method with receiver comment."""
        extra = {'receiver': '*Service'}
        header = build_go_minimal_header_context(
            'foreca', ['context'], 'method_declaration', extra
        )
        
        expected = 'package foreca\nimport "context"\n// receiver: *Service'
        assert header == expected
    
    def test_no_imports_minimal_header(self):
        """Test minimal header context with no imports."""
        header = build_go_minimal_header_context(
            'main', [], 'function_declaration', {}
        )
        
        expected = 'package main'
        assert header == expected
    
    def test_import_limit_in_header(self):
        """Test that header context limits imports to 5."""
        imports = ['context', 'time', 'fmt', 'os', 'net', 'http', 'json']
        header = build_go_minimal_header_context(
            'foreca', imports, 'function_declaration', {}
        )
        
        # Should only include first 5 imports
        assert 'context' in header
        assert 'time' in header
        assert 'fmt' in header
        assert 'os' in header
        assert 'net' in header
        assert 'http' not in header
        assert 'json' not in header
    
    def test_extract_imports_with_aliases(self):
        """Test import extraction respecting aliases."""
        chunk_text = """
        ctx, span := xotel.Tracer.Start(ctx, "service:forecast-location")
        defer span.End()
        
        mapping, err := s.mappings.Get(ctx, id)
        if err != nil {
            log.Error(ctx, "Cannot get mapping", zap.Error(err))
        }
        """
        
        all_imports = [
            'context', 'time', 'go.opentelemetry.io/otel/trace', 
            'go.uber.org/zap', 'go.impalastudios.com/log'
        ]
        
        import_aliases = {
            'xotel': 'go.impalastudios.com/otel',
            'log': 'go.impalastudios.com/log'
        }
        
        imports_used = extract_chunk_imports_with_aliases(
            chunk_text, all_imports, import_aliases
        )
        
        # Should include aliased imports
        assert 'go.impalastudios.com/otel' in imports_used
        assert 'go.impalastudios.com/log' in imports_used
        assert 'go.opentelemetry.io/otel/trace' in imports_used
        assert 'go.uber.org/zap' in imports_used
    
    def test_import_extraction_without_aliases(self):
        """Test import extraction without aliases."""
        chunk_text = """
        ctx := context.Background()
        now := time.Now()
        fmt.Println("Hello")
        """
        
        all_imports = ['context', 'time', 'fmt', 'os']
        import_aliases = {}
        
        imports_used = extract_chunk_imports_with_aliases(
            chunk_text, all_imports, import_aliases
        )
        
        assert 'context' in imports_used
        assert 'time' in imports_used
        assert 'fmt' in imports_used
        assert 'os' not in imports_used  # Not used in chunk
    
    def test_value_receiver_comment(self):
        """Test receiver comment for value receiver."""
        extra = {'receiver': 'User'}
        header = build_go_minimal_header_context(
            'models', ['fmt'], 'method_declaration', extra
        )
        
        expected = 'package models\nimport "fmt"\n// receiver: User'
        assert header == expected
    
    def test_complex_receiver_comment(self):
        """Test receiver comment for complex receiver types."""
        extra = {'receiver': '*WeatherService'}
        header = build_go_minimal_header_context(
            'foreca', ['context'], 'method_declaration', extra
        )
        
        expected = 'package foreca\nimport "context"\n// receiver: *WeatherService'
        assert header == expected


if __name__ == '__main__':
    pytest.main([__file__])
