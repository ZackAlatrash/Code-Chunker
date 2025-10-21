#!/usr/bin/env python3
"""
Test Go function and method AST paths and symbols.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import build_go_ast_path, extract_go_symbols_defined


class TestGoFunctionASTPathAndSymbols:
    """Test Go function and method AST paths and symbols."""
    
    def test_package_level_function_ast_path(self):
        """Test AST path for package-level functions."""
        extra = {'function_name': 'NewService'}
        ast_path = build_go_ast_path('function_declaration', extra)
        assert ast_path == 'go:function:NewService'
        
        extra = {'function_name': 'RunRootCmd'}
        ast_path = build_go_ast_path('function_declaration', extra)
        assert ast_path == 'go:function:RunRootCmd'
    
    def test_method_with_pointer_receiver_ast_path(self):
        """Test AST path for methods with pointer receivers."""
        extra = {
            'method_name': 'GetForecastForLocation',
            'receiver': '*Service'
        }
        ast_path = build_go_ast_path('method_declaration', extra)
        assert ast_path == 'go:method:(*Service).GetForecastForLocation'
        
        extra = {
            'method_name': 'getCacheKeyForLocation',
            'receiver': '*Service'
        }
        ast_path = build_go_ast_path('method_declaration', extra)
        assert ast_path == 'go:method:(*Service).getCacheKeyForLocation'
    
    def test_method_with_value_receiver_ast_path(self):
        """Test AST path for methods with value receivers."""
        extra = {
            'method_name': 'String',
            'receiver': 'User'
        }
        ast_path = build_go_ast_path('method_declaration', extra)
        assert ast_path == 'go:method:(User).String'
        
        extra = {
            'method_name': 'IsValid',
            'receiver': 'Config'
        }
        ast_path = build_go_ast_path('method_declaration', extra)
        assert ast_path == 'go:method:(Config).IsValid'
    
    def test_function_symbols_defined(self):
        """Test that function names are correctly extracted as symbols defined."""
        extra = {'function_name': 'NewService'}
        symbols = extract_go_symbols_defined('function_declaration', extra)
        assert symbols == ['NewService']
        
        extra = {'function_name': 'RunRootCmd'}
        symbols = extract_go_symbols_defined('function_declaration', extra)
        assert symbols == ['RunRootCmd']
    
    def test_method_symbols_defined(self):
        """Test that method names are correctly extracted as symbols defined."""
        extra = {
            'method_name': 'GetForecastForLocation',
            'receiver': '*Service'
        }
        symbols = extract_go_symbols_defined('method_declaration', extra)
        assert symbols == ['GetForecastForLocation']
        
        extra = {
            'method_name': 'getCacheKeyForLocation',
            'receiver': '*Service'
        }
        symbols = extract_go_symbols_defined('method_declaration', extra)
        assert symbols == ['getCacheKeyForLocation']
    
    def test_empty_extra_dict_handling(self):
        """Test handling of empty extra dictionary."""
        extra = {}
        
        # Function with empty extra
        ast_path = build_go_ast_path('function_declaration', extra)
        assert ast_path == 'go:function:'
        
        symbols = extract_go_symbols_defined('function_declaration', extra)
        assert symbols == []
        
        # Method with empty extra
        ast_path = build_go_ast_path('method_declaration', extra)
        assert ast_path == 'go:method:'
        
        symbols = extract_go_symbols_defined('method_declaration', extra)
        assert symbols == []
    
    def test_type_declaration_ast_path_and_symbols(self):
        """Test type declaration AST paths and symbols."""
        extra = {'type_name': 'Service', 'type_kind': 'struct'}
        ast_path = build_go_ast_path('type_declaration', extra)
        assert ast_path == 'go:type:Service (struct)'
        
        symbols = extract_go_symbols_defined('type_declaration', extra)
        assert symbols == ['Service']
        
        extra = {'type_name': 'Client', 'type_kind': 'interface'}
        ast_path = build_go_ast_path('type_declaration', extra)
        assert ast_path == 'go:type:Client (interface)'
        
        symbols = extract_go_symbols_defined('type_declaration', extra)
        assert symbols == ['Client']
    
    def test_complex_receiver_names(self):
        """Test AST paths with complex receiver names."""
        extra = {
            'method_name': 'Process',
            'receiver': '*WeatherService'
        }
        ast_path = build_go_ast_path('method_declaration', extra)
        assert ast_path == 'go:method:(*WeatherService).Process'
        
        extra = {
            'method_name': 'Validate',
            'receiver': '*foreca.Client'
        }
        ast_path = build_go_ast_path('method_declaration', extra)
        assert ast_path == 'go:method:(*foreca.Client).Validate'


if __name__ == '__main__':
    pytest.main([__file__])
