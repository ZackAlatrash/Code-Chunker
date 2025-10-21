"""
nwe_v3_enrich: Enhanced chunking with structural metadata and LLM enrichment.

This module provides Go-specific utilities, Tree-sitter based detection,
adapters for enrichment, and optional LLM summarization via local Ollama.
"""

from .go_heuristics import (
    GO_BUILTINS,
    guess_import_for_pkg,
    find_package,
    extract_qualified_identifiers,
    extract_capitalized_identifiers,
)

from .adapter import (
    FileContext,
    parse_file_context,
    clean_code_for_symbols,
    is_generated_file,
    normalize_rel_path,
    compute_minimal_imports,
    compute_symbols_referenced,
    build_header_context_minimal,
    infer_go_structure,
)

from .utils import (
    looks_english,
    lint_keywords,
    contains_forbidden_terms,
)

from .llm_qwen import (
    summarize_chunk_qwen,
    digest_for_cache,
)

# Tree-sitter support (optional)
try:
    from .treesitter_go import GoTSIndexer, TSFileIndex, HAS_TREESITTER
    __all_treesitter__ = ["GoTSIndexer", "TSFileIndex", "HAS_TREESITTER"]
except ImportError:
    __all_treesitter__ = []

__all__ = [
    # go_heuristics (symbol extraction only)
    "GO_BUILTINS",
    "guess_import_for_pkg",
    "find_package",
    "extract_qualified_identifiers",
    "extract_capitalized_identifiers",
    # adapter
    "FileContext",
    "parse_file_context",
    "clean_code_for_symbols",
    "is_generated_file",
    "normalize_rel_path",
    "compute_minimal_imports",
    "compute_symbols_referenced",
    "build_header_context_minimal",
    "infer_go_structure",
    # utils
    "looks_english",
    "lint_keywords",
    "contains_forbidden_terms",
    # llm_qwen
    "summarize_chunk_qwen",
    "digest_for_cache",
] + __all_treesitter__
