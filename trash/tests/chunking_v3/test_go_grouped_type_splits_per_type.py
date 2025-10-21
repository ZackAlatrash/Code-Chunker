#!/usr/bin/env python3
"""
Test that grouped type declarations are split into individual type chunks.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import create_go_type_chunks, parse_grouped_types


class TestGoGroupedTypeSplitsPerType:
    """Test that grouped type declarations are split into individual type chunks."""
    
    def test_grouped_types_split_into_individual_chunks(self):
        """Test that grouped type declaration creates separate chunks for each type."""
        lines = [
            "type (",
            "    providerClient interface {",
            "        GetForecastForLocation(ctx context.Context, id int, loc *time.Location) (*Forecast, error)",
            "    }",
            "",
            "    mappingsRepository interface {",
            "        Get(ctx context.Context, id int) (*Mapping, error)",
            "    }",
            "",
            "    cacheClient interface {",
            "        Get(key string) (*cache.Item, error)",
            "        Set(key string, value []byte) error",
            "    }",
            ")"
        ]
        
        span = (1, 13)
        extra = {'type_name': '', 'type_kind': ''}
        
        chunks = create_go_type_chunks(
            lines, span, extra, 'foreca', [], {}, 'test_repo',
            Path('internal/foreca/service.go'), 'sha123', 'internal/foreca/service.go',
            'gpt-4', None
        )
        
        # Should create 3 separate chunks for each interface
        assert len(chunks) == 3
        
        # Check first chunk (providerClient)
        provider_chunk = chunks[0]
        assert provider_chunk.ast_path == "go:type:providerClient (interface)"
        assert 'providerClient interface' in provider_chunk.core
        assert 'providerClient' in provider_chunk.symbols_defined
        assert provider_chunk.summary_1l == "Go interface providerClient for weather forecasting"
        
        # Check second chunk (mappingsRepository)
        mappings_chunk = chunks[1]
        assert mappings_chunk.ast_path == "go:type:mappingsRepository (interface)"
        assert 'mappingsRepository interface' in mappings_chunk.core
        assert 'mappingsRepository' in mappings_chunk.symbols_defined
        
        # Check third chunk (cacheClient)
        cache_chunk = chunks[2]
        assert cache_chunk.ast_path == "go:type:cacheClient (interface)"
        assert 'cacheClient interface' in cache_chunk.core
        assert 'cacheClient' in cache_chunk.symbols_defined
    
    def test_parse_grouped_types_extracts_individual_types(self):
        """Test that parse_grouped_types correctly extracts individual types."""
        type_text = """type (
    providerClient interface {
        GetForecastForLocation(ctx context.Context, id int, loc *time.Location) (*Forecast, error)
    }

    mappingsRepository interface {
        Get(ctx context.Context, id int) (*Mapping, error)
    }

    cacheClient interface {
        Get(key string) (*cache.Item, error)
        Set(key string, value []byte) error
    }
)"""
        
        types = parse_grouped_types(type_text, 1)
        
        assert len(types) == 3
        
        # Check first type
        assert types[0]['name'] == 'providerClient'
        assert types[0]['kind'] == 'interface'
        assert 'providerClient interface' in types[0]['content']
        
        # Check second type
        assert types[1]['name'] == 'mappingsRepository'
        assert types[1]['kind'] == 'interface'
        assert 'mappingsRepository interface' in types[1]['content']
        
        # Check third type
        assert types[2]['name'] == 'cacheClient'
        assert types[2]['kind'] == 'interface'
        assert 'cacheClient interface' in types[2]['content']
    
    def test_mixed_type_kinds_in_grouped_declaration(self):
        """Test grouped declaration with struct, interface, and alias types."""
        lines = [
            "type (",
            "    User struct {",
            "        ID   string",
            "        Name string",
            "    }",
            "",
            "    UserService interface {",
            "        GetUser(id string) (*User, error)",
            "    }",
            "",
            "    UserID = string",
            ")"
        ]
        
        span = (1, 11)
        extra = {'type_name': '', 'type_kind': ''}
        
        chunks = create_go_type_chunks(
            lines, span, extra, 'models', [], {}, 'test_repo',
            Path('internal/models/user.go'), 'sha123', 'internal/models/user.go',
            'gpt-4', None
        )
        
        assert len(chunks) == 3
        
        # Check struct chunk
        struct_chunk = chunks[0]
        assert struct_chunk.ast_path == "go:type:User (struct)"
        assert 'User struct' in struct_chunk.core
        assert 'User' in struct_chunk.symbols_defined
        
        # Check interface chunk
        interface_chunk = chunks[1]
        assert interface_chunk.ast_path == "go:type:UserService (interface)"
        assert 'UserService interface' in interface_chunk.core
        assert 'UserService' in interface_chunk.symbols_defined
        
        # Check alias chunk
        alias_chunk = chunks[2]
        assert alias_chunk.ast_path == "go:type:UserID (alias)"
        assert 'UserID = string' in alias_chunk.core
        assert 'UserID' in alias_chunk.symbols_defined
    
    def test_single_type_declaration_not_grouped(self):
        """Test that single type declaration is handled correctly (not grouped)."""
        lines = [
            "type Service struct {",
            "    client Client",
            "    cache  Cache",
            "}"
        ]
        
        span = (1, 4)
        extra = {'type_name': 'Service', 'type_kind': 'struct'}
        
        chunks = create_go_type_chunks(
            lines, span, extra, 'foreca', [], {}, 'test_repo',
            Path('internal/foreca/service.go'), 'sha123', 'internal/foreca/service.go',
            'gpt-4', None
        )
        
        # Should create 1 chunk for the single type
        assert len(chunks) == 1
        
        service_chunk = chunks[0]
        assert service_chunk.ast_path == "go:type:Service (struct)"
        assert 'Service struct' in service_chunk.core
        assert 'Service' in service_chunk.symbols_defined
        assert service_chunk.summary_1l == "Go struct Service for weather forecasting"


if __name__ == '__main__':
    pytest.main([__file__])
