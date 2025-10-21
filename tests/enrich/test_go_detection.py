#!/usr/bin/env python3
"""
Tree-sitter based Go detection tests.

Tests the AST-based detection system for accurate classification of:
- Methods (with/without body chunks)
- Functions
- Types (struct/interface)
- Headers
- Multi-declaration chunks
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from nwe_v3_enrich.treesitter_go import GoTSIndexer, HAS_TREESITTER
from nwe_v3_enrich.adapter import FileContext, infer_go_structure_ts

# Skip tests if tree-sitter not available
if not HAS_TREESITTER:
    print("Warning: tree-sitter not available, skipping tests")
    sys.exit(0)


def test_method_pointer_receiver_body_chunk():
    """Test: Method body chunk correctly identified via AST overlap."""
    source = '''package foreca

type Service struct {
    client *http.Client
}

func (s *Service) GetForecastForLocation(ctx context.Context, id int) (*Forecast, error) {
    ctx, span := xotel.Tracer.Start(ctx, "service:forecast-location")
    defer span.End()
    
    result, err := s.cache.Get(id)
    return result, err
}
'''
    
    indexer = GoTSIndexer()
    idx = indexer.parse_file(source)
    
    print("Test: Method pointer receiver body chunk")
    
    # Find the body chunk (lines after the signature)
    body_chunk = '''    result, err := s.cache.Get(id)
    return result, err'''
    
    start_byte = source.encode('utf-8').find(body_chunk.encode('utf-8'))
    end_byte = start_byte + len(body_chunk.encode('utf-8'))
    
    mapping = indexer.locate_for_chunk(idx, start_byte, end_byte)
    
    print(f"  node_kind: {mapping['node_kind']}")
    print(f"  primary_symbol: {mapping['primary_symbol']}")
    print(f"  ast_path: {mapping['ast_path']}")
    print(f"  receiver: {mapping['receiver']}")
    
    assert mapping["node_kind"] == "method", f"Expected 'method', got '{mapping['node_kind']}'"
    assert mapping["primary_symbol"] == "GetForecastForLocation", f"Expected 'GetForecastForLocation', got '{mapping['primary_symbol']}'"
    assert "(*Service).GetForecastForLocation" in mapping["ast_path"], f"Expected '(*Service).GetForecastForLocation' in ast_path, got '{mapping['ast_path']}'"
    
    print("  ✓ Passed\n")


def test_function_free_function():
    """Test: Free function correctly identified."""
    source = '''package foreca

func NewService(provider providerClient, mappings mappingsRepository) *Service {
    return &Service{
        provider: provider,
        mappings: mappings,
    }
}
'''
    
    indexer = GoTSIndexer()
    idx = indexer.parse_file(source)
    
    print("Test: Free function")
    
    # Chunk covering the function
    chunk_start = source.encode('utf-8').find(b'func NewService')
    chunk_end = source.encode('utf-8').find(b'}') + 1
    
    mapping = indexer.locate_for_chunk(idx, chunk_start, chunk_end)
    
    print(f"  node_kind: {mapping['node_kind']}")
    print(f"  primary_symbol: {mapping['primary_symbol']}")
    print(f"  ast_path: {mapping['ast_path']}")
    
    assert mapping["node_kind"] == "function", f"Expected 'function', got '{mapping['node_kind']}'"
    assert mapping["primary_symbol"] == "NewService", f"Expected 'NewService', got '{mapping['primary_symbol']}'"
    assert mapping["ast_path"] == "go:function:NewService", f"Expected 'go:function:NewService', got '{mapping['ast_path']}'"
    
    print("  ✓ Passed\n")


def test_type_struct_and_interface():
    """Test: Struct and interface types correctly identified."""
    source = '''package foreca

type Service struct {
    provider providerClient
    mappings mappingsRepository
}

type Provider interface {
    GetForecast(id int) (*Forecast, error)
}
'''
    
    indexer = GoTSIndexer()
    idx = indexer.parse_file(source)
    
    print("Test: Struct and interface types")
    
    # Test struct
    struct_start = source.encode('utf-8').find(b'type Service struct')
    struct_end = source.encode('utf-8').find(b'}', struct_start) + 1
    
    mapping = indexer.locate_for_chunk(idx, struct_start, struct_end)
    
    print(f"  Struct - node_kind: {mapping['node_kind']}")
    print(f"  Struct - primary_symbol: {mapping['primary_symbol']}")
    print(f"  Struct - type_kind: {mapping['type_kind']}")
    print(f"  Struct - ast_path: {mapping['ast_path']}")
    
    assert mapping["node_kind"] == "type", f"Expected 'type', got '{mapping['node_kind']}'"
    assert mapping["primary_symbol"] == "Service", f"Expected 'Service', got '{mapping['primary_symbol']}'"
    assert mapping["type_kind"] == "struct", f"Expected 'struct', got '{mapping['type_kind']}'"
    assert mapping["ast_path"] == "go:type:Service (struct)", f"Expected 'go:type:Service (struct)', got '{mapping['ast_path']}'"
    
    # Test interface
    interface_start = source.encode('utf-8').find(b'type Provider interface')
    interface_end = len(source.encode('utf-8'))
    
    mapping = indexer.locate_for_chunk(idx, interface_start, interface_end)
    
    print(f"  Interface - node_kind: {mapping['node_kind']}")
    print(f"  Interface - primary_symbol: {mapping['primary_symbol']}")
    print(f"  Interface - type_kind: {mapping['type_kind']}")
    
    assert mapping["node_kind"] == "type", f"Expected 'type', got '{mapping['node_kind']}'"
    assert mapping["primary_symbol"] == "Provider", f"Expected 'Provider', got '{mapping['primary_symbol']}'"
    assert mapping["type_kind"] == "interface", f"Expected 'interface', got '{mapping['type_kind']}'"
    
    print("  ✓ Passed\n")


def test_multidecl_chunk_primary_and_defs():
    """Test: Multi-declaration chunk with primary and def_symbols."""
    source = '''package foreca

type Service struct {
    client *http.Client
}

func (s *Service) GetForecast(id int) (*Forecast, error) {
    return nil, nil
}

func (s *Service) GetMapping(id int) (*Mapping, error) {
    return nil, nil
}
'''
    
    indexer = GoTSIndexer()
    idx = indexer.parse_file(source)
    
    print("Test: Multi-declaration chunk")
    
    # Chunk covering both methods
    chunk_start = source.encode('utf-8').find(b'func (s *Service) GetForecast')
    chunk_end = len(source.encode('utf-8'))
    
    mapping = indexer.locate_for_chunk(idx, chunk_start, chunk_end)
    
    print(f"  node_kind: {mapping['node_kind']}")
    print(f"  primary_symbol: {mapping['primary_symbol']}")
    print(f"  def_symbols: {mapping.get('def_symbols', [])}")
    
    assert mapping["node_kind"] == "method", f"Expected 'method', got '{mapping['node_kind']}'"
    assert mapping["primary_symbol"] == "GetForecast", f"Expected 'GetForecast', got '{mapping['primary_symbol']}'"
    assert len(mapping.get("def_symbols", [])) >= 1, "Expected at least 1 def_symbol"
    
    # Check second method is in def_symbols
    def_names = [d.get("name", "") for d in mapping.get("def_symbols", [])]
    assert "GetMapping" in def_names, f"Expected 'GetMapping' in def_symbols, got {def_names}"
    
    print("  ✓ Passed\n")


def test_header_chunk():
    """Test: Header chunk (package/imports) correctly identified."""
    source = '''package foreca

import (
    "context"
    "time"
)

type Service struct {}
'''
    
    indexer = GoTSIndexer()
    idx = indexer.parse_file(source)
    
    print("Test: Header chunk")
    
    # Chunk covering only package and imports
    chunk_start = 0
    chunk_end = source.encode('utf-8').find(b'type Service') - 1
    
    mapping = indexer.locate_for_chunk(idx, chunk_start, chunk_end)
    
    print(f"  node_kind: {mapping['node_kind']}")
    print(f"  is_header: {mapping['is_header']}")
    print(f"  ast_path: {mapping['ast_path']}")
    
    assert mapping["node_kind"] == "header", f"Expected 'header', got '{mapping['node_kind']}'"
    assert mapping["is_header"] == True, "Expected is_header=True"
    assert mapping["ast_path"] == "go:file_header", f"Expected 'go:file_header', got '{mapping['ast_path']}'"
    
    print("  ✓ Passed\n")


def test_unknown_comment_only():
    """Test: Comment-only chunk returns unknown."""
    source = '''package foreca

// This is a comment block
// with multiple lines
// but no code declarations

type Service struct {}
'''
    
    indexer = GoTSIndexer()
    idx = indexer.parse_file(source)
    
    print("Test: Unknown comment-only chunk")
    
    # Chunk covering only comments
    chunk_start = source.encode('utf-8').find(b'// This is a comment')
    chunk_end = source.encode('utf-8').find(b'type Service') - 1
    
    mapping = indexer.locate_for_chunk(idx, chunk_start, chunk_end)
    
    print(f"  node_kind: {mapping['node_kind']}")
    print(f"  primary_symbol: {mapping['primary_symbol']}")
    print(f"  ast_path: {mapping['ast_path']}")
    
    # This might be "header" or "unknown" depending on implementation
    # Accept either as long as it's not misclassified as method/function/type
    assert mapping["node_kind"] in ("unknown", "header"), f"Expected 'unknown' or 'header', got '{mapping['node_kind']}'"
    
    print("  ✓ Passed\n")


def test_package_qualified_receiver():
    """Test: Package-qualified receiver handled correctly."""
    source = '''package foreca

import "some/pkg"

func (s *pkg.Service) Do(ctx context.Context) error {
    return nil
}
'''
    
    indexer = GoTSIndexer()
    idx = indexer.parse_file(source)
    
    print("Test: Package-qualified receiver")
    
    chunk_start = source.encode('utf-8').find(b'func (s *pkg.Service)')
    chunk_end = len(source.encode('utf-8'))
    
    mapping = indexer.locate_for_chunk(idx, chunk_start, chunk_end)
    
    print(f"  node_kind: {mapping['node_kind']}")
    print(f"  primary_symbol: {mapping['primary_symbol']}")
    print(f"  type_name: {mapping['type_name']}")
    print(f"  receiver: {mapping['receiver']}")
    print(f"  ast_path: {mapping['ast_path']}")
    
    assert mapping["node_kind"] == "method", f"Expected 'method', got '{mapping['node_kind']}'"
    assert mapping["primary_symbol"] == "Do", f"Expected 'Do', got '{mapping['primary_symbol']}'"
    assert mapping["type_name"] == "Service", f"Expected 'Service', got '{mapping['type_name']}'"
    # ast_path should strip package prefix
    assert "(*Service).Do" in mapping["ast_path"], f"Expected '(*Service).Do' in ast_path, got '{mapping['ast_path']}'"
    
    print("  ✓ Passed\n")


def main():
    """Run all tests."""
    print("Running Tree-sitter Go detection tests...\n")
    
    try:
        test_method_pointer_receiver_body_chunk()
        test_function_free_function()
        test_type_struct_and_interface()
        test_multidecl_chunk_primary_and_defs()
        test_header_chunk()
        test_unknown_comment_only()
        test_package_qualified_receiver()
        
        print("🎉 All Tree-sitter Go detection tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

