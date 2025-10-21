"""
Test Go import formatting and ordering.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from build_chunks_v3 import build_go_minimal_header_context


class TestGoImportFormattingAndOrder:
    """Test Go import formatting and ordering."""
    
    def test_single_import_formatting(self):
        """Test that single import is formatted as import "module/path"."""
        package_name = "main"
        imports_used = ["context"]
        node_type = "function_declaration"
        extra = {}
        
        header_context = build_go_minimal_header_context(package_name, imports_used, node_type, extra)
        
        # Should use single-line import format
        assert 'import "context"' in header_context, f"Should use single-line import format, got: {header_context}"
        assert 'import (' not in header_context, f"Should not use multi-line import format for single import, got: {header_context}"
    
    def test_multiple_imports_formatting(self):
        """Test that multiple imports are formatted with parentheses and sorted alphabetically."""
        package_name = "main"
        imports_used = ["context", "time", "fmt"]
        node_type = "function_declaration"
        extra = {}
        
        header_context = build_go_minimal_header_context(package_name, imports_used, node_type, extra)
        
        # Should use multi-line import format
        assert 'import (' in header_context, f"Should use multi-line import format, got: {header_context}"
        assert ')' in header_context, f"Should close import block, got: {header_context}"
        
        # Should be sorted alphabetically
        lines = header_context.split('\n')
        import_lines = [line for line in lines if line.strip().startswith('"')]
        assert len(import_lines) == 3, f"Should have 3 import lines, got: {import_lines}"
        
        # Check alphabetical order
        import_paths = [line.strip().strip('"') for line in import_lines]
        assert import_paths == ['context', 'fmt', 'time'], f"Imports should be sorted alphabetically, got: {import_paths}"
    
    def test_import_alphabetical_ordering(self):
        """Test that imports are sorted alphabetically by module path."""
        package_name = "main"
        imports_used = ["zap", "context", "time", "fmt", "errors"]
        node_type = "function_declaration"
        extra = {}
        
        header_context = build_go_minimal_header_context(package_name, imports_used, node_type, extra)
        
        # Extract import lines
        lines = header_context.split('\n')
        import_lines = [line for line in lines if line.strip().startswith('"')]
        import_paths = [line.strip().strip('"') for line in import_lines]
        
        # Should be sorted alphabetically
        expected_order = ['context', 'errors', 'fmt', 'time', 'zap']
        assert import_paths == expected_order, f"Imports should be sorted alphabetically, got: {import_paths}, expected: {expected_order}"
    
    def test_package_declaration_always_present(self):
        """Test that package declaration is always present for non-file-header chunks."""
        package_name = "main"
        imports_used = ["context"]
        node_type = "function_declaration"
        extra = {}
        
        header_context = build_go_minimal_header_context(package_name, imports_used, node_type, extra)
        
        # Should start with package declaration
        assert header_context.startswith('package main'), f"Should start with package declaration, got: {header_context}"
    
    def test_package_fallback_to_main(self):
        """Test that package falls back to 'main' when not provided."""
        package_name = ""
        imports_used = ["context"]
        node_type = "function_declaration"
        extra = {}
        
        header_context = build_go_minimal_header_context(package_name, imports_used, node_type, extra)
        
        # Should fall back to 'package main'
        assert header_context.startswith('package main'), f"Should fall back to 'package main', got: {header_context}"
    
    def test_receiver_comment_for_methods(self):
        """Test that receiver comment is added for method declarations."""
        package_name = "main"
        imports_used = ["context"]
        node_type = "method_declaration"
        extra = {"receiver": "s *Service"}
        
        header_context = build_go_minimal_header_context(package_name, imports_used, node_type, extra)
        
        # Should include receiver comment
        assert '// receiver: s *Service' in header_context, f"Should include receiver comment, got: {header_context}"
    
    def test_no_receiver_comment_for_functions(self):
        """Test that receiver comment is not added for function declarations."""
        package_name = "main"
        imports_used = ["context"]
        node_type = "function_declaration"
        extra = {}
        
        header_context = build_go_minimal_header_context(package_name, imports_used, node_type, extra)
        
        # Should not include receiver comment
        assert '// receiver:' not in header_context, f"Should not include receiver comment for functions, got: {header_context}"
    
    def test_import_limit_respected(self):
        """Test that import limit (5 imports) is respected."""
        package_name = "main"
        imports_used = ["context", "time", "fmt", "errors", "log", "os", "net"]  # 7 imports
        node_type = "function_declaration"
        extra = {}
        
        header_context = build_go_minimal_header_context(package_name, imports_used, node_type, extra)
        
        # Should only include first 5 imports
        lines = header_context.split('\n')
        import_lines = [line for line in lines if line.strip().startswith('"')]
        assert len(import_lines) == 5, f"Should limit to 5 imports, got {len(import_lines)}: {import_lines}"
        
        # Should include the first 5 alphabetically sorted imports
        import_paths = [line.strip().strip('"') for line in import_lines]
        expected_first_5 = ['context', 'errors', 'fmt', 'log', 'net']
        assert import_paths == expected_first_5, f"Should include first 5 alphabetically sorted imports, got: {import_paths}"
    
    def test_empty_imports_no_import_block(self):
        """Test that empty imports list doesn't create import block."""
        package_name = "main"
        imports_used = []
        node_type = "function_declaration"
        extra = {}
        
        header_context = build_go_minimal_header_context(package_name, imports_used, node_type, extra)
        
        # Should only have package declaration
        assert header_context == 'package main', f"Should only have package declaration, got: {header_context}"
        assert 'import' not in header_context, f"Should not have import block for empty imports, got: {header_context}"
    
    def test_complex_import_paths_ordering(self):
        """Test that complex import paths are ordered correctly."""
        package_name = "main"
        imports_used = [
            "github.com/gin-gonic/gin",
            "github.com/gorilla/mux",
            "go.impalastudios.com/weather/foreca_proxy/internal/foreca",
            "context",
            "time"
        ]
        node_type = "function_declaration"
        extra = {}
        
        header_context = build_go_minimal_header_context(package_name, imports_used, node_type, extra)
        
        # Extract import lines
        lines = header_context.split('\n')
        import_lines = [line for line in lines if line.strip().startswith('"')]
        import_paths = [line.strip().strip('"') for line in import_lines]
        
        # Should be sorted alphabetically
        expected_order = [
            "context",
            "github.com/gin-gonic/gin",
            "github.com/gorilla/mux",
            "go.impalastudios.com/weather/foreca_proxy/internal/foreca",
            "time"
        ]
        assert import_paths == expected_order, f"Complex imports should be sorted alphabetically, got: {import_paths}, expected: {expected_order}"


if __name__ == '__main__':
    pytest.main([__file__])
