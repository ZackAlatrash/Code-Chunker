"""
Test Go summary and QA terms for specific interfaces and structs.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from build_chunks_v3 import generate_go_summary, generate_go_qa_terms


class TestGoSummaryAndTermsSpecific:
    """Test Go summary and QA terms for specific interfaces and structs."""
    
    def test_providerclient_interface_summary_and_terms(self):
        """Test providerClient interface has correct summary and QA terms."""
        extra = {
            'type_name': 'providerClient',
            'type_kind': 'interface'
        }
        file_path = Path('internal/foreca/service.go')
        imports_used = ['context', 'time']
        
        # Test summary
        summary = generate_go_summary('type_declaration', extra, file_path)
        expected_summary = "Go interface that exposes GetForecastForLocation using context.Context and time.Location for the Foreca proxy."
        assert summary == expected_summary, f"Expected: {expected_summary}, got: {summary}"
        
        # Test QA terms
        qa_terms = generate_go_qa_terms('type_declaration', extra, imports_used, file_path)
        expected_terms = ['providerClient', 'GetForecastForLocation', 'context.Context', 'time.Location', 'forecast', 'foreca', 'proxy']
        for term in expected_terms:
            assert term in qa_terms, f"QA terms should contain '{term}', got: {qa_terms}"
    
    def test_mappingsrepository_interface_summary_and_terms(self):
        """Test mappingsRepository interface has correct summary and QA terms."""
        extra = {
            'type_name': 'mappingsRepository',
            'type_kind': 'interface'
        }
        file_path = Path('internal/foreca/repository.go')
        imports_used = ['context']
        
        # Test summary
        summary = generate_go_summary('type_declaration', extra, file_path)
        expected_summary = "Repository interface to load Mapping by id; used by the Foreca service."
        assert summary == expected_summary, f"Expected: {expected_summary}, got: {summary}"
        
        # Test QA terms
        qa_terms = generate_go_qa_terms('type_declaration', extra, imports_used, file_path)
        expected_terms = ['mappingsRepository', 'Mapping', 'repository', 'foreca', 'service']
        for term in expected_terms:
            assert term in qa_terms, f"QA terms should contain '{term}', got: {qa_terms}"
    
    def test_cacheclient_interface_summary_and_terms(self):
        """Test cacheClient interface has correct summary and QA terms."""
        extra = {
            'type_name': 'cacheClient',
            'type_kind': 'interface'
        }
        file_path = Path('internal/foreca/cache.go')
        imports_used = ['context']
        
        # Test summary
        summary = generate_go_summary('type_declaration', extra, file_path)
        expected_summary = "Cache interface for getting/setting forecast items used by the Foreca proxy."
        assert summary == expected_summary, f"Expected: {expected_summary}, got: {summary}"
        
        # Test QA terms
        qa_terms = generate_go_qa_terms('type_declaration', extra, imports_used, file_path)
        expected_terms = ['cacheClient', 'cache.Item', 'get', 'set', 'cache', 'foreca', 'proxy']
        for term in expected_terms:
            assert term in qa_terms, f"QA terms should contain '{term}', got: {qa_terms}"
    
    def test_service_struct_summary_and_terms(self):
        """Test Service struct has correct summary and QA terms."""
        extra = {
            'type_name': 'Service',
            'type_kind': 'struct'
        }
        file_path = Path('internal/foreca/service.go')
        imports_used = ['context', 'time', 'singleflight']
        
        # Test summary
        summary = generate_go_summary('type_declaration', extra, file_path)
        expected_summary = "Service aggregates singleflight, provider, mappings, and cache clients with a TTL for the Foreca proxy."
        assert summary == expected_summary, f"Expected: {expected_summary}, got: {summary}"
        
        # Test QA terms
        qa_terms = generate_go_qa_terms('type_declaration', extra, imports_used, file_path)
        expected_terms = ['Service', 'singleflight', 'provider', 'mappings', 'cache', 'TTL', 'foreca', 'proxy']
        for term in expected_terms:
            assert term in qa_terms, f"QA terms should contain '{term}', got: {qa_terms}"
    
    def test_gomock_struct_summary_and_terms(self):
        """Test gomock struct has correct summary and QA terms."""
        extra = {
            'type_name': 'MockhttpClient',
            'type_kind': 'struct'
        }
        file_path = Path('internal/foreca/mocks.go')
        imports_used = ['go.uber.org/mock/gomock']
        
        # Test summary
        summary = generate_go_summary('type_declaration', extra, file_path)
        expected_summary = "MockhttpClient is a gomock-generated test double; stores a Controller and a recorder to define expectations."
        assert summary == expected_summary, f"Expected: {expected_summary}, got: {summary}"
        
        # Test QA terms
        qa_terms = generate_go_qa_terms('type_declaration', extra, imports_used, file_path)
        expected_terms = ['gomock', 'mock', 'controller', 'recorder', 'EXPECT', 'test double']
        for term in expected_terms:
            assert term in qa_terms, f"QA terms should contain '{term}', got: {qa_terms}"
    
    def test_expect_method_summary_and_terms(self):
        """Test EXPECT method has correct summary and QA terms."""
        extra = {
            'method_name': 'EXPECT',
            'receiver': 'm *MockhttpClient'
        }
        file_path = Path('internal/foreca/mocks.go')
        imports_used = ['go.uber.org/mock/gomock']
        
        # Test summary
        summary = generate_go_summary('method_declaration', extra, file_path)
        expected_summary = "Returns the gomock recorder to define expectations on MockhttpClient."
        assert summary == expected_summary, f"Expected: {expected_summary}, got: {summary}"
        
        # Test QA terms
        qa_terms = generate_go_qa_terms('method_declaration', extra, imports_used, file_path)
        expected_terms = ['EXPECT', 'gomock', 'recorder', 'mock', 'expectations']
        for term in expected_terms:
            assert term in qa_terms, f"QA terms should contain '{term}', got: {qa_terms}"
    
    def test_newmock_constructor_summary_and_terms(self):
        """Test NewMock constructor has correct summary and QA terms."""
        extra = {
            'function_name': 'NewMockhttpClient'
        }
        file_path = Path('internal/foreca/mocks.go')
        imports_used = ['go.uber.org/mock/gomock']
        
        # Test summary
        summary = generate_go_summary('function_declaration', extra, file_path)
        expected_summary = "Constructor returning a MockhttpClient: wires gomock.Controller and initializes the recorder."
        assert summary == expected_summary, f"Expected: {expected_summary}, got: {summary}"
        
        # Test QA terms
        qa_terms = generate_go_qa_terms('function_declaration', extra, imports_used, file_path)
        expected_terms = ['NewMockhttpClient', 'gomock', 'controller', 'recorder', 'mock']
        for term in expected_terms:
            assert term in qa_terms, f"QA terms should contain '{term}', got: {qa_terms}"
    
    def test_generic_interface_summary_and_terms(self):
        """Test generic interface has correct summary and QA terms."""
        extra = {
            'type_name': 'GenericInterface',
            'type_kind': 'interface'
        }
        file_path = Path('internal/foreca/generic.go')
        imports_used = ['context']
        
        # Test summary (should fall back to generic template)
        summary = generate_go_summary('type_declaration', extra, file_path)
        expected_summary = "Go interface GenericInterface for weather forecasting"
        assert summary == expected_summary, f"Expected: {expected_summary}, got: {summary}"
        
        # Test QA terms (should include type name and kind)
        qa_terms = generate_go_qa_terms('type_declaration', extra, imports_used, file_path)
        assert 'GenericInterface' in qa_terms, f"QA terms should contain type name, got: {qa_terms}"
        assert 'interface' in qa_terms, f"QA terms should contain type kind, got: {qa_terms}"
        assert 'foreca' in qa_terms, f"QA terms should contain domain terms, got: {qa_terms}"
    
    def test_summary_contains_role_words(self):
        """Test that summaries contain role words like 'interface', 'struct', 'method'."""
        # Test interface
        interface_extra = {'type_name': 'TestInterface', 'type_kind': 'interface'}
        interface_summary = generate_go_summary('type_declaration', interface_extra, Path('test.go'))
        assert 'interface' in interface_summary.lower(), f"Interface summary should contain 'interface', got: {interface_summary}"
        
        # Test struct
        struct_extra = {'type_name': 'TestStruct', 'type_kind': 'struct'}
        struct_summary = generate_go_summary('type_declaration', struct_extra, Path('test.go'))
        assert 'struct' in struct_summary.lower(), f"Struct summary should contain 'struct', got: {struct_summary}"
        
        # Test method
        method_extra = {'method_name': 'TestMethod', 'receiver': 's *Service'}
        method_summary = generate_go_summary('method_declaration', method_extra, Path('test.go'))
        assert 'method' in method_summary.lower(), f"Method summary should contain 'method', got: {method_summary}"
    
    def test_qa_terms_non_empty(self):
        """Test that QA terms are non-empty for all node types."""
        # Test interface
        interface_extra = {'type_name': 'TestInterface', 'type_kind': 'interface'}
        interface_terms = generate_go_qa_terms('type_declaration', interface_extra, ['context'], Path('test.go'))
        assert interface_terms, f"Interface QA terms should be non-empty, got: {interface_terms}"
        
        # Test struct
        struct_extra = {'type_name': 'TestStruct', 'type_kind': 'struct'}
        struct_terms = generate_go_qa_terms('type_declaration', struct_extra, ['context'], Path('test.go'))
        assert struct_terms, f"Struct QA terms should be non-empty, got: {struct_terms}"
        
        # Test method
        method_extra = {'method_name': 'TestMethod', 'receiver': 's *Service'}
        method_terms = generate_go_qa_terms('method_declaration', method_extra, ['context'], Path('test.go'))
        assert method_terms, f"Method QA terms should be non-empty, got: {method_terms}"


if __name__ == '__main__':
    pytest.main([__file__])
