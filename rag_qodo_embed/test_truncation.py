#!/usr/bin/env python3
"""
Unit tests for truncation utilities.
"""
import unittest
from .truncation import truncate_text, truncate_text_smart


class TestTruncation(unittest.TestCase):
    """Test cases for text truncation functions."""
    
    def test_truncate_text_basic(self):
        """Test basic truncation functionality."""
        text = "Hello, World! This is a test."
        result = truncate_text(text, 10)
        self.assertLessEqual(len(result), 10)
        self.assertTrue(result.startswith("Hello, Wor"))
    
    def test_truncate_text_no_truncation_needed(self):
        """Test when truncation is not needed."""
        text = "Short"
        result = truncate_text(text, 10)
        self.assertEqual(result, text)
    
    def test_truncate_text_empty(self):
        """Test with empty text."""
        result = truncate_text("", 10)
        self.assertEqual(result, "")
    
    def test_truncate_text_unicode(self):
        """Test UTF-8 safe truncation."""
        text = "Hello 世界! This has unicode characters."
        result = truncate_text(text, 15)
        # Should not break UTF-8 sequences
        self.assertLessEqual(len(result), 15)
        # Should be valid UTF-8
        result.encode('utf-8').decode('utf-8')
    
    def test_truncate_text_invalid_max_chars(self):
        """Test with invalid max_chars."""
        with self.assertRaises(ValueError):
            truncate_text("test", 0)
        
        with self.assertRaises(ValueError):
            truncate_text("test", -1)
    
    def test_truncate_text_smart_basic(self):
        """Test smart truncation with line boundaries."""
        text = "Line 1\nLine 2\nLine 3"
        result = truncate_text_smart(text, 10)
        # Should end at a line boundary if possible
        self.assertLessEqual(len(result), 10)
    
    def test_truncate_text_smart_no_preserve(self):
        """Test smart truncation with preserve_lines=False."""
        text = "Line 1\nLine 2\nLine 3"
        result = truncate_text_smart(text, 10, preserve_lines=False)
        # Should behave like regular truncation
        self.assertEqual(len(result), 10)


if __name__ == '__main__':
    unittest.main()
