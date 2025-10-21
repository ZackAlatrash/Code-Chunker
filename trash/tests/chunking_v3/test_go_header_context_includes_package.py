#!/usr/bin/env python3
"""
Test that Go header context includes package declaration as first line.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import build_go_minimal_header_context


class TestGoHeaderContextIncludesPackage:
    """Test that Go header context includes package declaration as first line."""
    
    def test_header_context_starts_with_package(self):
        """Test that header context starts with package declaration."""
        header = build_go_minimal_header_context(
            'foreca', ['context', 'time'], 'function_declaration', {}
        )
        
        lines = header.split('\n')
        assert lines[0] == 'package foreca'
    
    def test_header_context_with_single_import(self):
        """Test header context with single import includes package first."""
        header = build_go_minimal_header_context(
            'main', ['fmt'], 'function_declaration', {}
        )
        
        lines = header.split('\n')
        assert lines[0] == 'package main'
        assert lines[1] == 'import "fmt"'
    
    def test_header_context_with_multiple_imports(self):
        """Test header context with multiple imports includes package first."""
        header = build_go_minimal_header_context(
            'foreca', ['context', 'time', 'fmt'], 'function_declaration', {}
        )
        
        lines = header.split('\n')
        assert lines[0] == 'package foreca'
        assert lines[1] == 'import ('
        assert lines[2] == '\t"context"'
        assert lines[3] == '\t"time"'
        assert lines[4] == '\t"fmt"'
        assert lines[5] == ')'
    
    def test_header_context_with_method_receiver(self):
        """Test header context with method receiver includes package first."""
        extra = {'receiver': '*Service'}
        header = build_go_minimal_header_context(
            'foreca', ['context'], 'method_declaration', extra
        )
        
        lines = header.split('\n')
        assert lines[0] == 'package foreca'
        assert lines[1] == 'import "context"'
        assert lines[2] == '// receiver: *Service'
    
    def test_header_context_no_imports(self):
        """Test header context with no imports still includes package."""
        header = build_go_minimal_header_context(
            'main', [], 'function_declaration', {}
        )
        
        assert header == 'package main'
    
    def test_header_context_package_name_preserved(self):
        """Test that package name is preserved exactly as provided."""
        test_cases = [
            ('foreca', 'package foreca'),
            ('main', 'package main'),
            ('internal_foreca', 'package internal_foreca'),
            ('pkg_utils', 'package pkg_utils'),
        ]
        
        for package_name, expected in test_cases:
            header = build_go_minimal_header_context(
                package_name, [], 'function_declaration', {}
            )
            assert header == expected
    
    def test_header_context_import_limit_respected(self):
        """Test that import limit is respected while package comes first."""
        imports = ['context', 'time', 'fmt', 'os', 'net', 'http', 'json', 'strings']
        header = build_go_minimal_header_context(
            'foreca', imports, 'function_declaration', {}
        )
        
        lines = header.split('\n')
        assert lines[0] == 'package foreca'
        assert lines[1] == 'import ('
        # Should only include first 5 imports
        assert len([line for line in lines if line.startswith('\t"')]) == 5
        assert '\t"context"' in lines
        assert '\t"time"' in lines
        assert '\t"fmt"' in lines
        assert '\t"os"' in lines
        assert '\t"net"' in lines
        assert '\t"http"' not in lines  # Should be excluded due to limit


if __name__ == '__main__':
    pytest.main([__file__])
