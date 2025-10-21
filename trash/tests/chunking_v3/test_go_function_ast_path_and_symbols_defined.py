"""
Test Go function ast_path and symbols_defined for package-level functions.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from build_chunks_v3 import build_go_ast_path, extract_go_symbols_defined


class TestGoFunctionAstPathAndSymbolsDefined:
    """Test Go function ast_path and symbols_defined generation."""
    
    def test_function_ast_path_newmockhttpclient(self):
        """Test ast_path for NewMockhttpClient constructor."""
        extra = {'function_name': 'NewMockhttpClient'}
        ast_path = build_go_ast_path('function_declaration', extra)
        assert ast_path == "go:function:NewMockhttpClient"
    
    def test_function_ast_path_generic_function(self):
        """Test ast_path for generic function."""
        extra = {'function_name': 'ProcessData'}
        ast_path = build_go_ast_path('function_declaration', extra)
        assert ast_path == "go:function:ProcessData"
    
    def test_function_ast_path_missing_name(self):
        """Test ast_path for function with missing name."""
        extra = {}
        ast_path = build_go_ast_path('function_declaration', extra)
        assert ast_path == "go:function:"
    
    def test_symbols_defined_newmockhttpclient(self):
        """Test symbols_defined for NewMockhttpClient constructor."""
        extra = {'function_name': 'NewMockhttpClient'}
        symbols = extract_go_symbols_defined('function_declaration', extra)
        assert symbols == ['NewMockhttpClient']
    
    def test_symbols_defined_generic_function(self):
        """Test symbols_defined for generic function."""
        extra = {'function_name': 'ProcessData'}
        symbols = extract_go_symbols_defined('function_declaration', extra)
        assert symbols == ['ProcessData']
    
    def test_symbols_defined_missing_name(self):
        """Test symbols_defined for function with missing name."""
        extra = {}
        symbols = extract_go_symbols_defined('function_declaration', extra)
        assert symbols == []
    
    def test_constructor_patterns(self):
        """Test various constructor patterns."""
        constructors = [
            'NewMockhttpClient',
            'NewService',
            'NewClient',
            'NewMockDatabase'
        ]
        
        for constructor in constructors:
            extra = {'function_name': constructor}
            ast_path = build_go_ast_path('function_declaration', extra)
            symbols = extract_go_symbols_defined('function_declaration', extra)
            
            assert ast_path == f"go:function:{constructor}"
            assert symbols == [constructor]
    
    def test_non_function_node_types(self):
        """Test that non-function node types don't get function ast_path."""
        extra = {'function_name': 'SomeFunction'}
        
        # Type declaration should not use function ast_path
        ast_path = build_go_ast_path('type_declaration', extra)
        assert ast_path != "go:function:SomeFunction"
        
        # Method declaration should not use function ast_path
        ast_path = build_go_ast_path('method_declaration', extra)
        assert ast_path != "go:function:SomeFunction"
    
    def test_symbols_defined_non_function_node_types(self):
        """Test that non-function node types don't get function symbols."""
        extra = {'function_name': 'SomeFunction'}
        
        # Type declaration should not return function symbols
        symbols = extract_go_symbols_defined('type_declaration', extra)
        assert 'SomeFunction' not in symbols
        
        # Method declaration should not return function symbols
        symbols = extract_go_symbols_defined('method_declaration', extra)
        assert 'SomeFunction' not in symbols


if __name__ == '__main__':
    pytest.main([__file__])
