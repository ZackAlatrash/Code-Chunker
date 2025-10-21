#!/usr/bin/env python3
"""
Unit tests for chunk_doctor Go validation rules.

Tests each validation rule with minimal chunk examples to ensure
proper detection of violations and application of fixes.
"""

import pytest
import sys
from pathlib import Path

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))

from chunk_doctor import ChunkDoctor, RuleViolation


class TestChunkDoctorGo:
    """Test Go chunk validation and fixing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.doctor = ChunkDoctor()
    
    def test_go_header_present_first_violation_missing(self):
        """Test GO_HEADER_PRESENT_FIRST when no file header exists."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_HEADER_PRESENT_FIRST'
        assert 'No go:file_header chunk found' in violations[0].message
    
    def test_go_header_present_first_violation_multiple(self):
        """Test GO_HEADER_PRESENT_FIRST when multiple file headers exist."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:file_header',
                'language': 'go',
                'start_line': 1,
                'end_line': 5
            },
            {
                'chunk_id': 'chunk2',
                'ast_path': 'go:file_header',
                'language': 'go',
                'start_line': 6,
                'end_line': 10
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 2
        assert all(v.code == 'GO_HEADER_PRESENT_FIRST' for v in violations)
        assert all('Multiple go:file_header chunks found' in v.message for v in violations)
    
    def test_go_header_present_first_violation_not_first(self):
        """Test GO_HEADER_PRESENT_FIRST when header is not first."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10
            },
            {
                'chunk_id': 'chunk2',
                'ast_path': 'go:file_header',
                'language': 'go',
                'start_line': 11,
                'end_line': 15
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_HEADER_PRESENT_FIRST'
        assert 'go:file_header chunk is not first' in violations[0].message
    
    def test_go_header_present_first_fix(self):
        """Test fixing GO_HEADER_PRESENT_FIRST by moving header to front."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10
            },
            {
                'chunk_id': 'chunk2',
                'ast_path': 'go:file_header',
                'language': 'go',
                'start_line': 11,
                'end_line': 15
            }
        ]
        
        fixed_chunks = self.doctor.fix_file_chunks(chunks)
        
        assert fixed_chunks[0]['ast_path'] == 'go:file_header'
        assert fixed_chunks[1]['ast_path'] == 'go:type:Service'
        assert 'Moved go:file_header to first position' in self.doctor.fixes_applied
    
    def test_go_neighbor_chain_violation_first_has_prev(self):
        """Test GO_NEIGHBOR_CHAIN when first chunk has prev."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:file_header',
                'language': 'go',
                'start_line': 1,
                'end_line': 5,
                'neighbors': {'prev': 'some_prev', 'next': 'chunk2'}
            },
            {
                'chunk_id': 'chunk2',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 6,
                'end_line': 10,
                'neighbors': {'prev': 'chunk1', 'next': None}
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_NEIGHBOR_CHAIN'
        assert 'First chunk should have prev = null' in violations[0].message
    
    def test_go_neighbor_chain_violation_wrong_prev(self):
        """Test GO_NEIGHBOR_CHAIN when prev is wrong."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:file_header',
                'language': 'go',
                'start_line': 1,
                'end_line': 5,
                'neighbors': {'prev': None, 'next': 'chunk2'}
            },
            {
                'chunk_id': 'chunk2',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 6,
                'end_line': 10,
                'neighbors': {'prev': 'wrong_prev', 'next': None}
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_NEIGHBOR_CHAIN'
        assert 'prev should be chunk1' in violations[0].message
    
    def test_go_neighbor_chain_fix(self):
        """Test fixing GO_NEIGHBOR_CHAIN by rebuilding chain."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:file_header',
                'language': 'go',
                'start_line': 1,
                'end_line': 5,
                'neighbors': {'prev': 'wrong', 'next': 'wrong'}
            },
            {
                'chunk_id': 'chunk2',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 6,
                'end_line': 10,
                'neighbors': {'prev': 'wrong', 'next': 'wrong'}
            }
        ]
        
        fixed_chunks = self.doctor.fix_file_chunks(chunks)
        
        assert fixed_chunks[0]['neighbors']['prev'] is None
        assert fixed_chunks[0]['neighbors']['next'] == 'chunk2'
        assert fixed_chunks[1]['neighbors']['prev'] == 'chunk1'
        assert fixed_chunks[1]['neighbors']['next'] is None
        assert 'Rebuilt neighbor chain' in self.doctor.fixes_applied
    
    def test_go_ast_path_format_violation_empty(self):
        """Test GO_AST_PATH_FORMAT when ast_path is empty."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': '',
                'language': 'go',
                'start_line': 1,
                'end_line': 10
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_AST_PATH_FORMAT'
        assert 'ast_path is empty' in violations[0].message
    
    def test_go_ast_path_format_violation_invalid_type(self):
        """Test GO_AST_PATH_FORMAT when type name is invalid."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:invalidType',
                'language': 'go',
                'start_line': 1,
                'end_line': 10
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_AST_PATH_FORMAT'
        assert 'Invalid type name in ast_path' in violations[0].message
    
    def test_go_ast_path_format_violation_invalid_function(self):
        """Test GO_AST_PATH_FORMAT when function name is invalid."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:function:invalidFunction',
                'language': 'go',
                'start_line': 1,
                'end_line': 10
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_AST_PATH_FORMAT'
        assert 'Invalid function name in ast_path' in violations[0].message
    
    def test_go_ast_path_format_violation_invalid_method(self):
        """Test GO_AST_PATH_FORMAT when method format is invalid."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:method:InvalidMethod',
                'language': 'go',
                'start_line': 1,
                'end_line': 10
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_AST_PATH_FORMAT'
        assert 'Invalid method format in ast_path' in violations[0].message
    
    def test_go_header_context_package_violation(self):
        """Test GO_HEADER_CONTEXT_PACKAGE when header doesn't start with package."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'header_context': 'import "fmt"\n// some comment'
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_HEADER_CONTEXT_PACKAGE'
        assert "header_context must start with 'package '" in violations[0].message
    
    def test_go_header_context_package_fix(self):
        """Test fixing GO_HEADER_CONTEXT_PACKAGE by adding package declaration."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'header_context': 'import "fmt"'
            },
            {
                'chunk_id': 'chunk2',
                'ast_path': 'go:file_header',
                'language': 'go',
                'start_line': 1,
                'end_line': 5,
                'text': 'package main\nimport "fmt"'
            }
        ]
        
        fixed_chunks = self.doctor.fix_file_chunks(chunks)
        
        assert fixed_chunks[0]['header_context'].startswith('package main')
        assert 'Added package main to header_context' in self.doctor.fixes_applied
    
    def test_go_imports_minimal_alphabetical_violation_unsorted(self):
        """Test GO_IMPORTS_MINIMAL_ALPHABETICAL when imports are not sorted."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'imports_used': ['fmt', 'context', 'time']
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_IMPORTS_MINIMAL_ALPHABETICAL'
        assert 'imports_used must be alphabetically sorted' in violations[0].message
    
    def test_go_imports_minimal_alphabetical_fix(self):
        """Test fixing GO_IMPORTS_MINIMAL_ALPHABETICAL by sorting imports."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'imports_used': ['fmt', 'context', 'time'],
                'header_context': 'package main\nimport (\n\t"fmt"\n\t"context"\n\t"time"\n)'
            }
        ]
        
        fixed_chunks = self.doctor.fix_file_chunks(chunks)
        
        assert fixed_chunks[0]['imports_used'] == ['context', 'fmt', 'time']
        assert 'Reformatted imports (alphabetized and proper formatting)' in self.doctor.fixes_applied
    
    def test_go_imports_builtin_leak_violation(self):
        """Test GO_IMPORTS_BUILTIN_LEAK when builtin types are in imports."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'imports_used': ['fmt', 'error', 'context']
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_IMPORTS_BUILTIN_LEAK'
        assert "Builtin 'error' should not be in imports_used" in violations[0].message
    
    def test_go_imports_builtin_leak_fix(self):
        """Test fixing GO_IMPORTS_BUILTIN_LEAK by removing builtin imports."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'imports_used': ['fmt', 'error', 'nil', 'context']
            }
        ]
        
        fixed_chunks = self.doctor.fix_file_chunks(chunks)
        
        assert fixed_chunks[0]['imports_used'] == ['context', 'fmt']
        assert 'Removed builtin imports' in self.doctor.fixes_applied
    
    def test_go_symbols_defined_present_violation(self):
        """Test GO_SYMBOLS_DEFINED_PRESENT when function has no symbols_defined."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:function:NewService',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'symbols_defined': []
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_SYMBOLS_DEFINED_PRESENT'
        assert 'Functions and types must have symbols_defined' in violations[0].message
    
    def test_go_symbols_self_reference_violation(self):
        """Test GO_SYMBOLS_SELF_REFERENCE when symbols_referenced contains self."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:function:NewService',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'symbols_defined': ['NewService'],
                'symbols_referenced': ['NewService', 'Service', 'context']
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_SYMBOLS_SELF_REFERENCE'
        assert 'symbols_referenced contains self-references' in violations[0].message
    
    def test_go_symbols_self_reference_fix(self):
        """Test fixing GO_SYMBOLS_SELF_REFERENCE by removing self-references."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:function:NewService',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'symbols_defined': ['NewService'],
                'symbols_referenced': ['NewService', 'Service', 'context']
            }
        ]
        
        fixed_chunks = self.doctor.fix_file_chunks(chunks)
        
        assert fixed_chunks[0]['symbols_referenced'] == ['Service', 'context']
        assert 'Removed self-references from symbols_referenced' in self.doctor.fixes_applied
    
    def test_go_symbols_qualified_pref_violation(self):
        """Test GO_SYMBOLS_QUALIFIED_PREF when both qualified and bare versions exist."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'symbols_referenced': ['context.Context', 'Context', 'time.Time', 'Time']
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 2  # Both Context and Time violations
        assert all(v.code == 'GO_SYMBOLS_QUALIFIED_PREF' for v in violations)
        assert any('Both qualified' in v.message and 'Context' in v.message for v in violations)
        assert any('Both qualified' in v.message and 'Time' in v.message for v in violations)
    
    def test_go_symbols_qualified_pref_fix(self):
        """Test fixing GO_SYMBOLS_QUALIFIED_PREF by preferring qualified symbols."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'symbols_referenced': ['context.Context', 'Context', 'time.Time', 'Time']
            }
        ]
        
        fixed_chunks = self.doctor.fix_file_chunks(chunks)
        
        assert fixed_chunks[0]['symbols_referenced'] == ['context.Context', 'time.Time']
        assert 'Preferred qualified symbols over bare names' in self.doctor.fixes_applied
    
    def test_go_path_id_normalized_violation_absolute_path(self):
        """Test GO_PATH_ID_NORMALIZED when path is absolute."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'path': '/absolute/path/to/file.go'
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_PATH_ID_NORMALIZED'
        assert 'path should be repo-relative (no leading slash)' in violations[0].message
    
    def test_go_path_id_normalized_fix(self):
        """Test fixing GO_PATH_ID_NORMALIZED by normalizing path and chunk_id."""
        chunks = [
            {
                'chunk_id': 'old_chunk_id',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'path': '/absolute/path/to/file.go',
                'repo': 'test_repo',
                'file_sha': 'abc123'
            }
        ]
        
        fixed_chunks = self.doctor.fix_file_chunks(chunks)
        
        assert fixed_chunks[0]['path'] == 'absolute/path/to/file.go'
        assert fixed_chunks[0]['chunk_id'] != 'old_chunk_id'  # Should be recomputed
        assert 'Normalized path and recomputed chunk_id' in self.doctor.fixes_applied
    
    def test_go_no_empty_chunks_violation_empty_text(self):
        """Test GO_NO_EMPTY_CHUNKS when text is empty."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'text': '',
                'core': 'some content'
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_NO_EMPTY_CHUNKS'
        assert 'Chunk text and core must not be empty' in violations[0].message
    
    def test_go_no_empty_chunks_violation_empty_core(self):
        """Test GO_NO_EMPTY_CHUNKS when core is empty."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'text': 'some content',
                'core': ''
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_NO_EMPTY_CHUNKS'
        assert 'Chunk text and core must not be empty' in violations[0].message
    
    def test_go_summary_template_violation_provider_client(self):
        """Test GO_SUMMARY_TEMPLATE when providerClient doesn't use specific template."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:providerClient',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'summary_1l': 'Go interface providerClient for weather forecasting'
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_SUMMARY_TEMPLATE'
        assert 'providerClient should use specific template' in violations[0].message
    
    def test_go_summary_template_fix(self):
        """Test fixing GO_SUMMARY_TEMPLATE by trimming generic suffix."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:providerClient',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'summary_1l': 'Go interface that exposes GetForecastForLocation using context.Context and time.Location for the Foreca proxy for weather forecasting'
            }
        ]
        
        fixed_chunks = self.doctor.fix_file_chunks(chunks)
        
        assert 'for weather forecasting' not in fixed_chunks[0]['summary_1l']
        assert 'Trimmed generic suffix from specific template' in self.doctor.fixes_applied
    
    def test_go_qa_terms_count_range_violation_too_few(self):
        """Test GO_QA_TERMS_COUNT_RANGE when terms are too few."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'qa_terms': 'foreca, weather'
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_QA_TERMS_COUNT_RANGE'
        assert 'QA terms should be 6-12 terms, got 2' in violations[0].message
    
    def test_go_qa_terms_count_range_violation_too_many(self):
        """Test GO_QA_TERMS_COUNT_RANGE when terms are too many."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'qa_terms': 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o'
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 1
        assert violations[0].code == 'GO_QA_TERMS_COUNT_RANGE'
        assert 'QA terms should be 6-12 terms, got 15' in violations[0].message
    
    def test_go_qa_terms_count_range_valid(self):
        """Test GO_QA_TERMS_COUNT_RANGE when terms are in valid range."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'qa_terms': 'foreca, weather, proxy, service, cache, time'
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 0
    
    def test_non_go_chunks_ignored(self):
        """Test that non-Go chunks are ignored."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'yaml:key:value',
                'language': 'yaml',
                'start_line': 1,
                'end_line': 10
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 0
    
    def test_empty_chunks_list(self):
        """Test that empty chunks list returns no violations."""
        chunks = []
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        assert len(violations) == 0
    
    def test_multiple_violations_same_chunk(self):
        """Test that multiple violations can be detected in the same chunk."""
        chunks = [
            {
                'chunk_id': 'chunk1',
                'ast_path': 'go:type:Service',
                'language': 'go',
                'start_line': 1,
                'end_line': 10,
                'header_context': 'import "fmt"',  # Missing package
                'imports_used': ['fmt', 'error'],  # Unsorted and has builtin
                'symbols_defined': ['Service'],
                'symbols_referenced': ['Service', 'context'],  # Self-reference
                'text': '',  # Empty text
                'qa_terms': 'a, b'  # Too few terms
            }
        ]
        
        violations = self.doctor.validate_file_chunks(chunks)
        
        # Should have multiple violations
        assert len(violations) >= 5
        violation_codes = {v.code for v in violations}
        expected_codes = {
            'GO_HEADER_CONTEXT_PACKAGE',
            'GO_IMPORTS_MINIMAL_ALPHABETICAL',
            'GO_IMPORTS_BUILTIN_LEAK',
            'GO_SYMBOLS_SELF_REFERENCE',
            'GO_NO_EMPTY_CHUNKS',
            'GO_QA_TERMS_COUNT_RANGE'
        }
        assert expected_codes.issubset(violation_codes)


if __name__ == '__main__':
    pytest.main([__file__])
