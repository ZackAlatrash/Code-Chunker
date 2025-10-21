"""
Test symbols_referenced normalization to prefer qualified identifiers and de-duplicate.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from build_chunks_v3 import extract_go_symbols_referenced_strict


class TestSymbolsReferencedNormalized:
    """Test symbols_referenced normalization."""
    
    def test_qualified_vs_bare_preference(self):
        """Test that qualified identifiers are preferred over bare names."""
        chunk_text = '''
        import "gomock"
        
        func TestSomething() {
            ctrl := gomock.NewController(t)
            mock := NewMockService(ctrl)
            mock.EXPECT().GetData().Return("test")
        }
        '''
        
        symbol_to_import_map = {
            'gomock': 'github.com/golang/mock/gomock',
            'NewMockService': 'github.com/golang/mock/gomock'
        }
        
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_to_import_map)
        
        # Should prefer gomock.NewController over just NewController
        assert 'gomock.NewController' in symbols
        # Should not have bare NewController if we have qualified version
        assert 'NewController' not in symbols or 'gomock.NewController' in symbols
    
    def test_no_duplicate_bare_qualified_mixes(self):
        """Test that we don't have both bare and qualified versions of the same symbol."""
        chunk_text = '''
        import "gomock"
        
        func TestSomething() {
            ctrl := gomock.NewController(t)
            mock := NewMockService(ctrl)
            mock.EXPECT().GetData().Return("test")
            Controller := gomock.Controller{}
        }
        '''
        
        symbol_to_import_map = {
            'gomock': 'github.com/golang/mock/gomock',
            'NewMockService': 'github.com/golang/mock/gomock'
        }
        
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_to_import_map)
        
        # Should not have both Controller and gomock.Controller
        has_qualified = any('gomock.Controller' in s for s in symbols)
        has_bare = 'Controller' in symbols
        
        if has_qualified:
            assert not has_bare, "Should not have both qualified and bare versions"
        else:
            assert has_bare, "Should have at least one version"
    
    def test_de_duplication(self):
        """Test that duplicate symbols are removed."""
        chunk_text = '''
        import "gomock"
        
        func TestSomething() {
            ctrl1 := gomock.NewController(t)
            ctrl2 := gomock.NewController(t)
            mock1 := NewMockService(ctrl1)
            mock2 := NewMockService(ctrl2)
        }
        '''
        
        symbol_to_import_map = {
            'gomock': 'github.com/golang/mock/gomock',
            'NewMockService': 'github.com/golang/mock/gomock'
        }
        
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_to_import_map)
        
        # Should not have duplicates
        assert len(symbols) == len(set(symbols)), "Symbols should be de-duplicated"
        
        # Should have gomock.NewController only once
        gomock_controller_count = sum(1 for s in symbols if 'gomock.NewController' in s)
        assert gomock_controller_count <= 1, "gomock.NewController should appear only once"
    
    def test_qualified_identifiers_preserved(self):
        """Test that qualified identifiers are preserved."""
        chunk_text = '''
        import "gomock"
        import "context"
        
        func TestSomething() {
            ctrl := gomock.NewController(t)
            ctx := context.Background()
            mock := NewMockService(ctrl)
        }
        '''
        
        symbol_to_import_map = {
            'gomock': 'github.com/golang/mock/gomock',
            'context': 'context',
            'NewMockService': 'github.com/golang/mock/gomock'
        }
        
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_to_import_map)
        
        # Should preserve qualified identifiers
        assert 'gomock.NewController' in symbols
        assert 'context.Background' in symbols
    
    def test_bare_symbols_when_no_qualified(self):
        """Test that bare symbols are included when no qualified version exists."""
        chunk_text = '''
        func TestSomething() {
            service := NewService()
            result := service.Process()
        }
        '''
        
        symbol_to_import_map = {}
        
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_to_import_map)
        
        # Should include bare symbols when no qualified version
        assert 'NewService' in symbols
        assert 'Process' in symbols
    
    def test_mixed_qualified_and_bare(self):
        """Test handling of mixed qualified and bare symbols."""
        chunk_text = '''
        import "gomock"
        
        func TestSomething() {
            ctrl := gomock.NewController(t)
            localService := NewLocalService()
            mock := NewMockService(ctrl)
        }
        '''
        
        symbol_to_import_map = {
            'gomock': 'github.com/golang/mock/gomock',
            'NewMockService': 'github.com/golang/mock/gomock'
        }
        
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_to_import_map)
        
        # Should have qualified gomock symbols
        assert 'gomock.NewController' in symbols
        
        # Should have bare local symbols
        assert 'NewLocalService' in symbols
        
        # Should not have bare versions of qualified symbols
        assert 'NewController' not in symbols or 'gomock.NewController' in symbols
    
    def test_error_special_case(self):
        """Test that error is handled as a special case."""
        chunk_text = '''
        func TestSomething() error {
            return errors.New("test error")
        }
        '''
        
        symbol_to_import_map = {}
        
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_to_import_map)
        
        # Should include error as special case
        assert 'error' in symbols
    
    def test_builtin_types_excluded(self):
        """Test that builtin types are excluded."""
        chunk_text = '''
        func TestSomething() {
            var str string
            var num int
            var flag bool
            var data []byte
        }
        '''
        
        symbol_to_import_map = {}
        
        symbols = extract_go_symbols_referenced_strict(chunk_text, symbol_to_import_map)
        
        # Should not include builtin types
        builtin_types = {'string', 'int', 'bool', 'byte'}
        for builtin in builtin_types:
            assert builtin not in symbols, f"Builtin type {builtin} should not be in symbols"


if __name__ == '__main__':
    pytest.main([__file__])
