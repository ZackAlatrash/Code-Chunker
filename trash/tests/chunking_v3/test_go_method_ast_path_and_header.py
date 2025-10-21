#!/usr/bin/env python3
"""
Test Go method AST paths and header context.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import (
    build_go_ast_path, build_go_header_context, 
    extract_method_name, extract_receiver
)


class TestGoMethodASTPathAndHeader:
    """Test Go method AST paths and header context generation."""
    
    def test_method_ast_path_generation(self):
        """Test AST path generation for methods."""
        # Test pointer receiver
        extra = {
            'method_name': 'GetUser',
            'receiver': '*UserService'
        }
        ast_path = build_go_ast_path('method_declaration', extra)
        assert ast_path == 'go:method:(*UserService).GetUser'
        
        # Test value receiver
        extra = {
            'method_name': 'String',
            'receiver': 'User'
        }
        ast_path = build_go_ast_path('method_declaration', extra)
        assert ast_path == 'go:method:(User).String'
        
        # Test type declaration
        extra = {
            'type_name': 'Service'
        }
        ast_path = build_go_ast_path('type_declaration', extra)
        assert ast_path == 'go:type:Service'
        
        # Test function declaration
        extra = {
            'function_name': 'NewService'
        }
        ast_path = build_go_ast_path('function_declaration', extra)
        assert ast_path == 'go:function:NewService'
        
        # Test package declaration
        ast_path = build_go_ast_path('package_declaration', {})
        assert ast_path == 'go:file_header'
    
    def test_method_header_context(self):
        """Test header context generation for methods."""
        package_name = "foreca"
        all_imports = [
            "context",
            "time",
            "go.opentelemetry.io/otel/attribute",
            "go.uber.org/zap"
        ]
        
        # Test method with receiver
        extra = {
            'method_name': 'GetForecast',
            'receiver': '*Service'
        }
        
        header = build_go_header_context(
            package_name, all_imports, 'method_declaration', extra, [], 10, 20
        )
        
        # Should include package declaration
        assert 'package foreca' in header
        
        # Should include imports
        assert 'import (' in header
        assert 'context' in header
        assert 'time' in header
        
        # Should include receiver type
        assert 'type Service struct { ... }' in header
    
    def test_type_header_context(self):
        """Test header context generation for types."""
        package_name = "models"
        all_imports = ["time", "encoding/json"]
        
        extra = {
            'type_name': 'User',
            'type_kind': 'struct'
        }
        
        header = build_go_header_context(
            package_name, all_imports, 'type_declaration', extra, [], 5, 10
        )
        
        # Should include package declaration
        assert 'package models' in header
        
        # Should include imports
        assert 'import (' in header
        assert 'time' in header
        assert 'encoding/json' in header
        
        # Should not include receiver type for type declarations
        assert 'type User struct { ... }' not in header
    
    def test_function_header_context(self):
        """Test header context generation for functions."""
        package_name = "utils"
        all_imports = ["fmt", "strings"]
        
        extra = {
            'function_name': 'FormatName'
        }
        
        header = build_go_header_context(
            package_name, all_imports, 'function_declaration', extra, [], 15, 25
        )
        
        # Should include package declaration
        assert 'package utils' in header
        
        # Should include imports
        assert 'import (' in header
        assert 'fmt' in header
        assert 'strings' in header
        
        # Should not include receiver type for functions
        assert 'struct { ... }' not in header
    
    def test_method_name_extraction(self):
        """Test method name extraction from code."""
        # Mock node for testing
        class MockNode:
            def __init__(self, start_point):
                self.start_point = start_point
        
        node = MockNode((10, 0))
        
        # Test method with receiver
        code = '''
func (s *UserService) GetUser(id string) (*User, error) {
    return s.db.FindByID(id)
}
'''
        method_name = extract_method_name(node, code)
        assert method_name == 'GetUser'
        
        # Test method without receiver (function)
        code = '''
func NewUserService(db Database) *UserService {
    return &UserService{db: db}
}
'''
        method_name = extract_method_name(node, code)
        assert method_name == 'NewUserService'
    
    def test_receiver_extraction(self):
        """Test receiver extraction from code."""
        # Mock node for testing
        class MockNode:
            def __init__(self, start_point):
                self.start_point = start_point
        
        node = MockNode((10, 0))
        
        # Test pointer receiver
        code = '''
func (s *UserService) GetUser(id string) (*User, error) {
    return s.db.FindByID(id)
}
'''
        receiver = extract_receiver(node, code)
        assert receiver == 's *UserService'
        
        # Test value receiver
        code = '''
func (u User) String() string {
    return fmt.Sprintf("User{ID: %s}", u.ID)
}
'''
        receiver = extract_receiver(node, code)
        assert receiver == 'u User'
        
        # Test function without receiver
        code = '''
func NewUser(id string) *User {
    return &User{ID: id}
}
'''
        receiver = extract_receiver(node, code)
        assert receiver == ''
    
    def test_header_context_import_limiting(self):
        """Test that header context limits imports to avoid bloat."""
        package_name = "test"
        all_imports = [
            "fmt", "strings", "time", "context", "encoding/json",
            "go.opentelemetry.io/otel/attribute", "go.uber.org/zap",
            "github.com/gin-gonic/gin", "gorm.io/gorm"
        ]
        
        extra = {'method_name': 'Test', 'receiver': '*Service'}
        
        header = build_go_header_context(
            package_name, all_imports, 'method_declaration', extra, [], 1, 10
        )
        
        # Should limit imports to first 5
        import_lines = [line for line in header.split('\n') if line.strip().startswith('"')]
        assert len(import_lines) <= 5
        
        # Should include package and receiver type
        assert 'package test' in header
        assert 'type Service struct { ... }' in header


if __name__ == '__main__':
    pytest.main([__file__])
