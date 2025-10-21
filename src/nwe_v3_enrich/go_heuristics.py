"""
Go-specific utilities for symbol extraction.

This module provides utility functions for Go code analysis.
Structure detection is now handled by Tree-sitter (see treesitter_go.py).
"""

import re
from typing import Optional, Set

# Go builtins that should be excluded from symbol references
GO_BUILTINS = {
    "error", "nil", "true", "false", "iota",
    "string", "int", "int8", "int16", "int32", "int64",
    "uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
    "byte", "rune", "float32", "float64", "complex64", "complex128",
    "bool", "make", "new", "len", "cap", "append", "copy", "delete",
    "close", "panic", "recover", "print", "println",
    "if", "else", "for", "range", "switch", "case", "default",
    "break", "continue", "goto", "fallthrough", "defer", "go", "select",
    "chan", "map", "interface", "struct", "type", "var", "const", "func",
    "package", "import", "return", "and", "or", "not",
}

# Package name to import path mapping
PKG_TO_IMPORT = {
    "time": "time",
    "json": "encoding/json",
    "protoimpl": "google.golang.org/protobuf/runtime/protoimpl",
    "protoreflect": "google.golang.org/protobuf/reflect/protoreflect",
    "zap": "go.uber.org/zap",
    "singleflight": "golang.org/x/sync/singleflight",
    "xotel": "go.impalastudios.com/otel",
}

# Compiled regex patterns for symbol extraction (not structure detection)
PACKAGE_PATTERN = re.compile(r"^package\s+([A-Za-z0-9_]+)", re.MULTILINE)
QUALIFIED_IDENT_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\b")
CAPITALIZED_IDENT_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\b")


def guess_import_for_pkg(pkg: str) -> Optional[str]:
    """Guess the import path for a package name."""
    return PKG_TO_IMPORT.get(pkg)


def find_package(code: str) -> str:
    """Extract package name from Go code (used as fallback only)."""
    match = PACKAGE_PATTERN.search(code)
    return match.group(1) if match else ""


def extract_qualified_identifiers(code: str) -> Set[str]:
    """Extract qualified identifiers like pkg.Symbol."""
    return set(QUALIFIED_IDENT_PATTERN.findall(code))


def extract_capitalized_identifiers(code: str) -> Set[str]:
    """Extract capitalized identifiers (likely types or exported functions)."""
    return set(CAPITALIZED_IDENT_PATTERN.findall(code))


# NOTE: All structure detection functions (detect_node_kind, extract_signature_info, etc.)
# have been removed. Use Tree-sitter based detection from treesitter_go.py instead.
