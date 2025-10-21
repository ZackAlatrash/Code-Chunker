"""
Test Go interface chunks have minimal imports (no unused imports).
"""

import pytest
import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from build_chunks_v3 import extract_chunk_imports_strict, build_symbol_to_import_map


class TestGoInterfaceMinimalImports:
    """Test Go interface chunks have minimal imports."""
    
    def test_interface_chunk_no_unused_imports(self):
        """Test that interface chunks don't include unused imports like github.com/pkg/errors."""
        # Interface chunk that doesn't use errors package
        interface_chunk = '''
        type providerClient interface {
            GetForecastForLocation(ctx context.Context, location time.Location) (*Forecast, error)
        }
        '''
        
        # Symbol to import mapping (including unused errors import)
        symbol_to_import_map = {
            'context': 'context',
            'time': 'time',
            'Forecast': 'go.impalastudios.com/weather/foreca_proxy/internal/foreca',
            'error': 'builtin',
            'errors': 'github.com/pkg/errors'  # This should NOT be included
        }
        
        # Extract imports used in the chunk
        imports_used = extract_chunk_imports_strict(interface_chunk, symbol_to_import_map)
        
        # Should not include github.com/pkg/errors since it's not used
        assert 'github.com/pkg/errors' not in imports_used, \
            f"Interface chunk should not include unused errors import, got: {imports_used}"
        
        # Should include context and time since they're used
        assert 'context' in imports_used, "Should include context import"
        assert 'time' in imports_used, "Should include time import"
    
    def test_struct_chunk_minimal_imports(self):
        """Test that struct chunks only include actually used imports."""
        # Struct chunk that uses specific imports
        struct_chunk = '''
        type Service struct {
            provider providerClient
            cache    cacheClient
            mappings mappingsRepository
            ttl      time.Duration
        }
        '''
        
        # Symbol to import mapping
        symbol_to_import_map = {
            'providerClient': 'go.impalastudios.com/weather/foreca_proxy/internal/foreca',
            'cacheClient': 'go.impalastudios.com/weather/foreca_proxy/internal/foreca',
            'mappingsRepository': 'go.impalastudios.com/weather/foreca_proxy/internal/foreca',
            'time': 'time',
            'Duration': 'time',
            'errors': 'github.com/pkg/errors',  # Unused
            'fmt': 'fmt'  # Unused
        }
        
        # Extract imports used in the chunk
        imports_used = extract_chunk_imports_strict(struct_chunk, symbol_to_import_map)
        
        # Should include time since time.Duration is used
        assert 'time' in imports_used, "Should include time import for time.Duration"
        
        # Should not include unused imports
        assert 'github.com/pkg/errors' not in imports_used, \
            f"Struct chunk should not include unused errors import, got: {imports_used}"
        assert 'fmt' not in imports_used, \
            f"Struct chunk should not include unused fmt import, got: {imports_used}"
    
    def test_function_chunk_minimal_imports(self):
        """Test that function chunks only include actually used imports."""
        # Function chunk that uses specific imports
        function_chunk = '''
        func NewService(provider providerClient, cache cacheClient) *Service {
            return &Service{
                provider: provider,
                cache:    cache,
                ttl:      time.Hour,
            }
        }
        '''
        
        # Symbol to import mapping
        symbol_to_import_map = {
            'providerClient': 'go.impalastudios.com/weather/foreca_proxy/internal/foreca',
            'cacheClient': 'go.impalastudios.com/weather/foreca_proxy/internal/foreca',
            'Service': 'go.impalastudios.com/weather/foreca_proxy/internal/foreca',
            'time': 'time',
            'Hour': 'time',
            'errors': 'github.com/pkg/errors',  # Unused
            'log': 'log'  # Unused
        }
        
        # Extract imports used in the chunk
        imports_used = extract_chunk_imports_strict(function_chunk, symbol_to_import_map)
        
        # Should include time since time.Hour is used
        assert 'time' in imports_used, "Should include time import for time.Hour"
        
        # Should not include unused imports
        assert 'github.com/pkg/errors' not in imports_used, \
            f"Function chunk should not include unused errors import, got: {imports_used}"
        assert 'log' not in imports_used, \
            f"Function chunk should not include unused log import, got: {imports_used}"
    
    def test_qualified_identifiers_import_extraction(self):
        """Test that qualified identifiers like context.Context are properly mapped to imports."""
        # Chunk with qualified identifiers
        chunk_with_qualified = '''
        func ProcessRequest(ctx context.Context, req *Request) (*Response, error) {
            if ctx.Err() != nil {
                return nil, ctx.Err()
            }
            return &Response{Data: req.Data}, nil
        }
        '''
        
        # Symbol to import mapping
        symbol_to_import_map = {
            'context': 'context',
            'Context': 'context',
            'Request': 'go.impalastudios.com/weather/foreca_proxy/internal/foreca',
            'Response': 'go.impalastudios.com/weather/foreca_proxy/internal/foreca',
            'error': 'builtin',
            'Err': 'context',
            'Data': 'go.impalastudios.com/weather/foreca_proxy/internal/foreca'
        }
        
        # Extract imports used in the chunk
        imports_used = extract_chunk_imports_strict(chunk_with_qualified, symbol_to_import_map)
        
        # Should include context since context.Context and ctx.Err() are used
        assert 'context' in imports_used, "Should include context import for qualified identifiers"
        
        # Should not include unused imports
        assert 'github.com/pkg/errors' not in imports_used, \
            f"Should not include unused errors import, got: {imports_used}"
    
    def test_build_symbol_to_import_map(self):
        """Test that symbol to import mapping is built correctly."""
        # Mock imports and aliases
        all_imports = [
            'context',
            'time',
            'github.com/pkg/errors',
            'go.impalastudios.com/weather/foreca_proxy/internal/foreca'
        ]
        import_aliases = {
            'foreca': 'go.impalastudios.com/weather/foreca_proxy/internal/foreca',
            'errors': 'github.com/pkg/errors'
        }
        
        # Build symbol to import mapping
        symbol_map = build_symbol_to_import_map(all_imports, import_aliases)
        
        # Should map package names to full import paths
        assert symbol_map['context'] == 'context'
        assert symbol_map['time'] == 'time'
        assert symbol_map['foreca'] == 'go.impalastudios.com/weather/foreca_proxy/internal/foreca'
        assert symbol_map['errors'] == 'github.com/pkg/errors'
        
        # Should also map last path component
        assert symbol_map['foreca'] == 'go.impalastudios.com/weather/foreca_proxy/internal/foreca'


if __name__ == '__main__':
    pytest.main([__file__])
