#!/usr/bin/env python3
"""
Tests for enrichment quick fixes.

Tests the four main fixes:
1. primary_kind always set correctly
2. concise English summaries (≤160 chars)
3. linted keywords (6-8 items, lowercase, 2-4 words, nouns only)
4. file-level synopsis generation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from nwe_v3_enrich.go_heuristics import detect_node_kind, extract_signature_info
from nwe_v3_enrich.adapter import generate_file_synopsis, infer_go_structure
from nwe_v3_enrich.utils import lint_keywords_enhanced, validate_summary_en, generate_fallback_summary


def test_primary_kind_detection():
    """Test that primary_kind is always set correctly."""
    print("Testing primary_kind detection...")
    
    # Test cases
    test_cases = [
        # Header chunk
        ('package main\n\nimport "fmt"', "header"),
        # Method chunk
        ('func (s *Service) GetForecast() error {', "method"),
        # Function chunk
        ('func NewService() *Service {', "function"),
        # Type chunk
        ('type ForecastResponse struct {', "type"),
        # Method body chunk (no signature)
        ('    return s.client.Get("/forecast")', "method"),
    ]
    
    for code, expected in test_cases:
        result = detect_node_kind(code)
        assert result == expected, f"Expected {expected}, got {result} for code: {code[:50]}..."
        print(f"  ✓ {expected}: {code[:30]}...")
    
    print("  ✓ All primary_kind tests passed\n")


def test_signature_extraction():
    """Test signature information extraction."""
    print("Testing signature extraction...")
    
    # Method signature
    method_code = 'func (s *Service) GetForecast() error {'
    sig_info = extract_signature_info(method_code)
    assert sig_info["receiver"] == "s *Service"
    assert sig_info["method_name"] == "GetForecast"
    assert sig_info["type_name"] == "Service"
    print("  ✓ Method signature extraction")
    
    # Function signature
    func_code = 'func NewService() *Service {'
    sig_info = extract_signature_info(func_code)
    assert sig_info["function_name"] == "NewService"
    print("  ✓ Function signature extraction")
    
    # Type signature
    type_code = 'type ForecastResponse struct {'
    sig_info = extract_signature_info(type_code)
    assert sig_info["type_name"] == "ForecastResponse"
    assert sig_info["type_kind"] == "struct"
    print("  ✓ Type signature extraction")
    
    print("  ✓ All signature extraction tests passed\n")


def test_keyword_linting():
    """Test enhanced keyword linting."""
    print("Testing keyword linting...")
    
    # Test input with various issues
    raw_keywords = [
        "GetForecast",  # Verb, should be filtered
        "Service.GetForecast",  # Dotted, should be filtered
        "weather forecast",  # Good
        "API client",  # Good
        "error handling",  # Good
        "singleflight",  # Good
        "cache",  # Good
        "opentelemetry",  # Good
        "timezone",  # Good
        "location",  # Good
        "FORECAST",  # All caps, should be filtered
        "a",  # Too short, should be filtered
    ]
    
    cleaned = lint_keywords_enhanced(raw_keywords)
    
    # Should have 6-8 items
    assert 6 <= len(cleaned) <= 8, f"Expected 6-8 keywords, got {len(cleaned)}"
    
    # All should be lowercase
    for kw in cleaned:
        assert kw.islower(), f"Keyword should be lowercase: {kw}"
    
    # No verbs should remain
    verb_stoplist = {"get", "set", "handle", "create", "make", "build"}
    for kw in cleaned:
        words = kw.split()
        for word in words:
            assert word not in verb_stoplist, f"Verb should be filtered: {word} in {kw}"
    
    # No dots should remain
    for kw in cleaned:
        assert "." not in kw, f"Dotted keyword should be filtered: {kw}"
    
    print(f"  ✓ Linted keywords: {cleaned}")
    print("  ✓ All keyword linting tests passed\n")


def test_summary_validation():
    """Test summary validation."""
    print("Testing summary validation...")
    
    # Valid summary
    valid_summary = "Method GetForecast retrieves weather data using singleflight cache."
    assert validate_summary_en(valid_summary), "Valid summary should pass validation"
    print("  ✓ Valid summary passes")
    
    # Too long summary
    long_summary = "This is a very long summary that exceeds the 160 character limit and should fail validation because it's too verbose and contains unnecessary details that make it longer than allowed."
    assert not validate_summary_en(long_summary), "Long summary should fail validation"
    print("  ✓ Long summary fails validation")
    
    # Summary with newlines
    multiline_summary = "Method GetForecast\nretrieves weather data."
    assert not validate_summary_en(multiline_summary), "Multiline summary should fail validation"
    print("  ✓ Multiline summary fails validation")
    
    print("  ✓ All summary validation tests passed\n")


def test_fallback_summary():
    """Test fallback summary generation."""
    print("Testing fallback summary generation...")
    
    # Method fallback
    signature_info = {"receiver": "s *Service", "method_name": "GetForecast"}
    identifiers = ["GetForecast", "Service", "weather"]
    fallback = generate_fallback_summary("method", signature_info, identifiers)
    assert "Method Service.GetForecast" in fallback
    assert "getforecast" in fallback  # Should contain the extracted noun
    print("  ✓ Method fallback summary")
    
    # Function fallback
    signature_info = {"function_name": "NewService"}
    identifiers = ["NewService", "Service"]
    fallback = generate_fallback_summary("function", signature_info, identifiers)
    assert "Function NewService" in fallback
    print("  ✓ Function fallback summary")
    
    # Type fallback
    signature_info = {"type_name": "ForecastResponse", "type_kind": "struct"}
    identifiers = ["ForecastResponse", "weather"]
    fallback = generate_fallback_summary("type", signature_info, identifiers)
    assert "Type ForecastResponse" in fallback
    print("  ✓ Type fallback summary")
    
    print("  ✓ All fallback summary tests passed\n")


def test_file_synopsis():
    """Test file synopsis generation."""
    print("Testing file synopsis generation...")
    
    # Sample Go file
    go_file = '''package foreca

import (
    "context"
    "time"
)

type Service struct {
    client *http.Client
}

func (s *Service) GetForecast(ctx context.Context) (*ForecastResponse, error) {
    return s.client.Get("/forecast")
}

func NewService() *Service {
    return &Service{}
}
'''
    
    synopsis = generate_file_synopsis(go_file, "service.go")
    
    # Should contain package name
    assert "foreca" in synopsis
    # Should mention types, methods, functions
    assert "type" in synopsis or "method" in synopsis or "function" in synopsis
    # Should be reasonable length
    assert len(synopsis) <= 250
    # Should end with period
    assert synopsis.endswith(".")
    
    print(f"  ✓ Generated synopsis: {synopsis}")
    print("  ✓ All file synopsis tests passed\n")


def test_integration():
    """Test integration of all components."""
    print("Testing integration...")
    
    # Sample Go code
    go_code = '''func (s *Service) GetForecast(ctx context.Context) (*ForecastResponse, error) {
    return s.client.Get("/forecast")
}'''
    
    # Test infer_go_structure
    result = infer_go_structure(go_code)
    
    # Should have correct primary_kind
    assert result["node_kind"] == "method"
    assert result["primary_symbol"] == "GetForecast"
    assert result["receiver"] == "s *Service"
    assert result["type_name"] == "Service"
    assert result["ast_path"] == "go:method:(*Service).GetForecast"
    
    print("  ✓ Integration test passed")
    print("  ✓ All integration tests passed\n")


def main():
    """Run all tests."""
    print("Running enrichment quick fixes tests...\n")
    
    try:
        test_primary_kind_detection()
        test_signature_extraction()
        test_keyword_linting()
        test_summary_validation()
        test_fallback_summary()
        test_file_synopsis()
        test_integration()
        
        print("🎉 All tests passed! The enrichment quick fixes are working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
