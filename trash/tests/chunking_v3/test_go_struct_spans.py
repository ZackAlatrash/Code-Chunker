#!/usr/bin/env python3
"""
Test Go struct chunking with proper spans.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_go_file, is_go_file


class TestGoStructSpans:
    """Test that Go struct chunks capture full spans, not just starting lines."""
    
    def test_struct_full_span_chunking(self):
        """Test that struct chunks include the full struct block."""
        go_code = '''
package foreca

import (
    "context"
    "time"
)

type Service struct {
    client    Client
    cache     Cache
    timeout   time.Duration
    retries   int
}

type Client interface {
    GetForecast(ctx context.Context, location string) (*Forecast, error)
}

type Cache interface {
    Get(key string) (*Forecast, error)
    Set(key string, forecast *Forecast, ttl time.Duration) error
}
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
            f.write(go_code)
            temp_path = Path(f.name)
        
        try:
            # Mock parser for testing
            class MockParser:
                def parse_code(self, content, ext):
                    # Return a mock tree structure
                    class MockNode:
                        def __init__(self, node_type, start_point, end_point):
                            self.type = node_type
                            self.start_point = start_point
                            self.end_point = end_point
                            self.children = []
                    
                    # Mock tree with package, imports, and types
                    root = MockNode('source_file', (0, 0), (20, 0))
                    root.children = [
                        MockNode('package_declaration', (0, 0), (0, 20)),
                        MockNode('import_declaration', (2, 0), (5, 0)),
                        MockNode('type_declaration', (7, 0), (12, 0)),  # Service struct
                        MockNode('type_declaration', (14, 0), (16, 0)), # Client interface
                        MockNode('type_declaration', (18, 0), (20, 0)), # Cache interface
                    ]
                    return root
            
            parser = MockParser()
            chunks = chunk_go_file(
                go_code, temp_path, "test_repo", "sha123", "test.go",
                "gpt-4", 150, 380, None, parser
            )
            
            # Should create chunks for each type declaration
            assert len(chunks) >= 3
            
            # Find the Service struct chunk
            service_chunk = None
            for chunk in chunks:
                if 'Service' in chunk.text and 'struct' in chunk.text:
                    service_chunk = chunk
                    break
            
            assert service_chunk is not None
            assert service_chunk.language == "go"
            assert service_chunk.ast_path == "go:type:Service"
            
            # The chunk should include the full struct definition
            assert 'type Service struct {' in service_chunk.text
            assert 'client    Client' in service_chunk.text
            assert 'cache     Cache' in service_chunk.text
            assert 'timeout   time.Duration' in service_chunk.text
            assert 'retries   int' in service_chunk.text
            assert '}' in service_chunk.text
            
            # Should not be just a single line
            assert service_chunk.end_line > service_chunk.start_line
            
        finally:
            temp_path.unlink()
    
    def test_is_go_file_detection(self):
        """Test Go file detection."""
        assert is_go_file(Path("test.go"))
        assert is_go_file(Path("main.go"))
        assert is_go_file(Path("service_test.go"))
        assert not is_go_file(Path("test.py"))
        assert not is_go_file(Path("test.js"))
        assert not is_go_file(Path("test.java"))
    
    def test_struct_with_methods_full_span(self):
        """Test that struct with methods captures full spans."""
        go_code = '''
package main

type UserService struct {
    db     Database
    cache  Cache
    logger Logger
}

func (s *UserService) GetUser(id string) (*User, error) {
    // Check cache first
    if user, found := s.cache.Get(id); found {
        return user, nil
    }
    
    // Query database
    user, err := s.db.FindByID(id)
    if err != nil {
        return nil, err
    }
    
    // Cache the result
    s.cache.Set(id, user, time.Hour)
    return user, nil
}

func (s *UserService) CreateUser(user *User) error {
    return s.db.Save(user)
}
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as f:
            f.write(go_code)
            temp_path = Path(f.name)
        
        try:
            # Mock parser for testing
            class MockParser:
                def parse_code(self, content, ext):
                    class MockNode:
                        def __init__(self, node_type, start_point, end_point):
                            self.type = node_type
                            self.start_point = start_point
                            self.end_point = end_point
                            self.children = []
                    
                    root = MockNode('source_file', (0, 0), (30, 0))
                    root.children = [
                        MockNode('package_declaration', (0, 0), (0, 10)),
                        MockNode('type_declaration', (2, 0), (6, 0)),  # UserService struct
                        MockNode('method_declaration', (8, 0), (20, 0)), # GetUser method
                        MockNode('method_declaration', (22, 0), (24, 0)), # CreateUser method
                    ]
                    return root
            
            parser = MockParser()
            chunks = chunk_go_file(
                go_code, temp_path, "test_repo", "sha123", "test.go",
                "gpt-4", 150, 380, None, parser
            )
            
            # Should create chunks for struct and methods
            assert len(chunks) >= 3
            
            # Find the UserService struct chunk
            struct_chunk = None
            for chunk in chunks:
                if 'UserService' in chunk.text and 'struct' in chunk.text:
                    struct_chunk = chunk
                    break
            
            assert struct_chunk is not None
            assert struct_chunk.ast_path == "go:type:UserService"
            
            # The struct chunk should include the full struct definition
            assert 'type UserService struct {' in struct_chunk.text
            assert 'db     Database' in struct_chunk.text
            assert 'cache  Cache' in struct_chunk.text
            assert 'logger Logger' in struct_chunk.text
            assert '}' in struct_chunk.text
            
            # Find the GetUser method chunk
            method_chunk = None
            for chunk in chunks:
                if 'GetUser' in chunk.text and 'func' in chunk.text:
                    method_chunk = chunk
                    break
            
            assert method_chunk is not None
            assert method_chunk.ast_path == "go:method:(*UserService).GetUser"
            
            # The method chunk should include the full method body
            assert 'func (s *UserService) GetUser(id string) (*User, error) {' in method_chunk.text
            assert '// Check cache first' in method_chunk.text
            assert 'return user, nil' in method_chunk.text
            assert '}' in method_chunk.text
            
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__])
