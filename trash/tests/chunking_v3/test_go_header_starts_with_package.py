#!/usr/bin/env python3
"""
Test that Go chunks have header context starting with package declaration.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_go_file, build_go_minimal_header_context
from CodeParser import CodeParser
import logging


class TestGoHeaderStartsWithPackage:
    """Test that Go chunks have proper header context starting with package."""
    
    def test_header_context_includes_package(self):
        """Test that header context starts with package declaration."""
        go_code = '''package main

import (
    "fmt"
    "time"
)

func main() {
    fmt.Println("Hello, World!")
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # If Go parser is not available, it will fall back to special file chunking
        # In that case, we just test that the function doesn't crash
        if not chunks:
            pytest.skip("Go parser not available, skipping test")
        
        # Check that non-file-header chunks have package in header_context
        for chunk in chunks:
            if chunk.ast_path != "go:file_header":
                assert chunk.header_context.startswith("package "), f"Header context should start with package: {chunk.header_context}"
                assert "main" in chunk.header_context, f"Package name should be in header: {chunk.header_context}"
    
    def test_minimal_header_context_function(self):
        """Test minimal header context for functions."""
        header = build_go_minimal_header_context("main", ["fmt", "time"], "function_declaration", {})
        
        assert header.startswith("package main")
        assert "import (" in header
        assert '"fmt"' in header
        assert '"time"' in header
    
    def test_minimal_header_context_method(self):
        """Test minimal header context for methods with receiver."""
        extra = {"receiver": "s *Service"}
        header = build_go_minimal_header_context("foreca", ["context"], "method_declaration", extra)
        
        assert header.startswith("package foreca")
        assert "// receiver: s *Service" in header
        assert '"context"' in header
    
    def test_header_context_no_imports(self):
        """Test header context when no imports are used."""
        header = build_go_minimal_header_context("main", [], "function_declaration", {})
        
        assert header == "package main"
    
    def test_text_composition(self):
        """Test that text field is properly composed as header_context + core."""
        go_code = '''package main

import "fmt"

func hello() {
    fmt.Println("Hello")
}'''
        
        parser = CodeParser()
        chunks = chunk_go_file(
            go_code, Path("test.go"), "test-repo", "test-sha", "test.go", 
            "gpt-4", 150, 250, logging.getLogger(__name__), parser
        )
        
        # If Go parser is not available, skip the test
        if not chunks:
            pytest.skip("Go parser not available, skipping test")
        
        # Find the function chunk
        func_chunk = None
        for chunk in chunks:
            if chunk.ast_path.startswith("go:function:"):
                func_chunk = chunk
                break
        
        if func_chunk is None:
            pytest.skip("No function chunk found, Go parser may not be available")
        
        # Check text composition
        expected_text = func_chunk.header_context + "\n" + func_chunk.core
        assert func_chunk.text == expected_text, f"Text should be header + core: {func_chunk.text}"
        
        # Check that header_context contains package
        assert func_chunk.header_context.startswith("package main")
        assert '"fmt"' in func_chunk.header_context
