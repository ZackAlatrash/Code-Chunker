#!/usr/bin/env python3
"""
Test that Go file header chunks contain both package and imports.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import create_go_file_header_chunk


class TestGoFileHeaderHasPackageAndImports:
    """Test that Go file header chunks contain both package and imports."""
    
    def test_file_header_contains_package_and_imports(self):
        """Test that file header chunk contains both package and import declarations."""
        lines = [
            "package foreca",
            "",
            "import (",
            '    "context"',
            '    "time"',
            '    "go.opentelemetry.io/otel/attribute"',
            '    "go.uber.org/zap"',
            ")"
        ]
        
        go_nodes = [
            ('package_declaration', (1, 1), {'package_name': 'foreca'}),
            ('import_declaration', (3, 7), {'imports': [
                'context', 'time', 'go.opentelemetry.io/otel/attribute', 'go.uber.org/zap'
            ]})
        ]
        
        all_imports = ['context', 'time', 'go.opentelemetry.io/otel/attribute', 'go.uber.org/zap']
        
        chunk = create_go_file_header_chunk(
            lines, go_nodes, 'foreca', all_imports, 'test_repo', 
            Path('internal/foreca/service.go'), 'sha123', 'internal/foreca/service.go',
            'gpt-4', None
        )
        
        assert chunk is not None
        assert chunk.ast_path == "go:file_header"
        assert chunk.language == "go"
        
        # Check that text contains both package and imports
        assert 'package foreca' in chunk.text
        assert 'import (' in chunk.text
        assert '"context"' in chunk.text
        assert '"time"' in chunk.text
        assert '"go.opentelemetry.io/otel/attribute"' in chunk.text
        assert '"go.uber.org/zap"' in chunk.text
        
        # Check summary mentions package name and notable imports
        assert 'Go package foreca' in chunk.summary_1l
        assert 'otel' in chunk.summary_1l or 'zap' in chunk.summary_1l
        
        # Check QA terms include package and notable imports
        assert 'foreca' in chunk.qa_terms
        assert 'otel' in chunk.qa_terms or 'zap' in chunk.qa_terms
    
    def test_file_header_package_only_no_imports(self):
        """Test file header with package declaration but no imports."""
        lines = [
            "package main"
        ]
        
        go_nodes = [
            ('package_declaration', (1, 1), {'package_name': 'main'})
        ]
        
        chunk = create_go_file_header_chunk(
            lines, go_nodes, 'main', [], 'test_repo', 
            Path('main.go'), 'sha123', 'main.go',
            'gpt-4', None
        )
        
        assert chunk is not None
        assert chunk.ast_path == "go:file_header"
        assert 'package main' in chunk.text
        assert chunk.summary_1l == "Go package main"
        assert 'main' in chunk.qa_terms
    
    def test_file_header_with_single_import(self):
        """Test file header with single import (not parenthesized)."""
        lines = [
            "package utils",
            'import "fmt"'
        ]
        
        go_nodes = [
            ('package_declaration', (1, 1), {'package_name': 'utils'}),
            ('import_declaration', (2, 2), {'imports': ['fmt']})
        ]
        
        chunk = create_go_file_header_chunk(
            lines, go_nodes, 'utils', ['fmt'], 'test_repo', 
            Path('pkg/utils/helper.go'), 'sha123', 'pkg/utils/helper.go',
            'gpt-4', None
        )
        
        assert chunk is not None
        assert 'package utils' in chunk.text
        assert 'import "fmt"' in chunk.text
        assert chunk.summary_1l == "Go package utils"
    
    def test_file_header_span_covers_package_through_imports(self):
        """Test that header chunk span covers from package line through import block."""
        lines = [
            "package foreca",
            "",
            "import (",
            '    "context"',
            '    "time"',
            ")",
            "",
            "type Service struct {"
        ]
        
        go_nodes = [
            ('package_declaration', (1, 1), {'package_name': 'foreca'}),
            ('import_declaration', (3, 5), {'imports': ['context', 'time']})
        ]
        
        chunk = create_go_file_header_chunk(
            lines, go_nodes, 'foreca', ['context', 'time'], 'test_repo', 
            Path('internal/foreca/service.go'), 'sha123', 'internal/foreca/service.go',
            'gpt-4', None
        )
        
        assert chunk is not None
        # Should span from line 1 (package) through line 5 (end of imports)
        assert chunk.start_line == 1
        assert chunk.end_line == 5
        
        # Text should include both package and imports
        header_text = '\n'.join(lines[0:5])  # Lines 1-5
        assert chunk.text == header_text


if __name__ == '__main__':
    pytest.main([__file__])
