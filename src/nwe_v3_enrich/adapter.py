"""
Adapter functions for enriching chunks with structural metadata and context.

This module provides the main enrichment logic that combines Go heuristics
with context building and symbol extraction.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

from .go_heuristics import (
    GO_BUILTINS,
    guess_import_for_pkg,
    find_package,
    extract_qualified_identifiers,
    extract_capitalized_identifiers,
)
from .utils import (
    looks_english,
    lint_keywords,
    contains_forbidden_terms,
    lint_keywords_enhanced,
    validate_summary_en,
    generate_fallback_summary,
)

# Tree-sitter support (optional, with fallback to regex)
try:
    from .treesitter_go import GoTSIndexer, TSFileIndex, HAS_TREESITTER
    USE_TREESITTER = HAS_TREESITTER and os.environ.get('GO_ENRICH_AST', '1') == '1'
except ImportError:
    HAS_TREESITTER = False
    USE_TREESITTER = False
    GoTSIndexer = None
    TSFileIndex = None


@dataclass
class FileContext:
    """Per-file context for Go code analysis."""
    package_name: str
    file_imports: List[str]  # normalized import paths
    import_alias_map: Dict[str, str]  # alias -> full import path
    abs_path: Optional[str] = None  # Absolute file path for Tree-sitter caching
    file_text: Optional[str] = None  # Full file text for Tree-sitter parsing


# Tree-sitter index cache (file_path -> TSFileIndex)
_ts_index_cache: Dict[str, Any] = {}


# English stopwords and protoc banner words
STOPWORDS = {
    "Code", "generated", "DO", "NOT", "EDIT", "Verify", "The", "Use", "instead",
    "Deprecated", "Use", "instead", "This", "That", "These", "Those", "A", "An",
    "And", "Or", "But", "In", "On", "At", "To", "For", "Of", "With", "By",
    "From", "Up", "About", "Into", "Through", "During", "Before", "After",
    "Above", "Below", "Between", "Among", "Under", "Over", "Inside", "Outside"
}


def parse_file_context(full_file_text: str) -> FileContext:
    """
    Parse the entire file to extract package and import context.
    
    Args:
        full_file_text: Complete file content
        
    Returns:
        FileContext with package name, imports, and alias mapping
    """
    # Extract package name from full file
    package_name = find_package(full_file_text)
    
    # Extract import blocks
    import_pattern = re.compile(r'import\s+(?:"([^"]+)"|\(([\s\S]*?)\))', re.MULTILINE)
    file_imports = []
    import_alias_map = {}
    
    for match in import_pattern.finditer(full_file_text):
        if match.group(1):  # Single import: import "path"
            import_path = match.group(1)
            file_imports.append(import_path)
            # Use last segment as alias
            alias = import_path.split('/')[-1]
            import_alias_map[alias] = import_path
        else:  # Multi-line import block
            import_block = match.group(2)
            for line in import_block.split('\n'):
                line = line.strip()
                if line.startswith('"') and line.endswith('"'):
                    import_path = line.strip('"')
                    file_imports.append(import_path)
                    # Use last segment as alias
                    alias = import_path.split('/')[-1]
                    import_alias_map[alias] = import_path
                elif ' ' in line and '"' in line:
                    # Handle explicit alias: alias "path"
                    parts = line.split('"')
                    if len(parts) >= 2:
                        alias = parts[0].strip()
                        import_path = parts[1]
                        file_imports.append(import_path)
                        import_alias_map[alias] = import_path
    
    return FileContext(
        package_name=package_name,
        file_imports=sorted(list(set(file_imports))),
        import_alias_map=import_alias_map
    )


def clean_code_for_symbols(code: str) -> str:
    """
    Remove comments and string literals from code for cleaner symbol extraction.
    
    Args:
        code: Raw code text
        
    Returns:
        Cleaned code with comments and strings removed
    """
    # Remove single-line comments
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    
    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # Remove string literals (both single and double quotes)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '', code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", '', code)
    
    # Remove backtick strings
    code = re.sub(r'`[^`]*`', '', code)
    
    return code


def is_generated_file(path: str, text: str) -> bool:
    """
    Determine if a file is generated code based on filename patterns and content.
    
    Args:
        path: File path
        text: File content
        
    Returns:
        True if the file appears to be generated code
    """
    # Check filename patterns
    filename = os.path.basename(path).lower()
    generated_patterns = [
        "*.pb.go", "*_mock.go", "zz_generated.*", "*generated*", "*mocks*"
    ]
    
    for pattern in generated_patterns:
        if pattern.startswith("*") and pattern.endswith("*"):
            if pattern[1:-1] in filename:
                return True
        elif pattern.startswith("*"):
            if filename.endswith(pattern[1:]):
                return True
        elif pattern.endswith("*"):
            if filename.startswith(pattern[:-1]):
                return True
        elif pattern == filename:
            return True
    
    # Check content for generation markers
    text_lower = text.lower()
    generation_markers = ["code generated", "do not edit"]
    
    # Check first 1000 characters for generation markers
    header = text_lower[:1000]
    for marker in generation_markers:
        if marker in header:
            return True
    
    return False


def normalize_rel_path(abs_or_rel: str, repo_root: Optional[Path]) -> str:
    """
    Normalize a path to a POSIX relative path.
    
    Args:
        abs_or_rel: Absolute or relative path
        repo_root: Optional repository root for normalization
        
    Returns:
        POSIX relative path (never absolute)
    """
    path = Path(abs_or_rel)
    
    if repo_root and path.is_absolute():
        try:
            rel_path = path.relative_to(repo_root)
            return str(rel_path).replace("\\", "/")
        except ValueError:
            # Path is not under repo_root, return as-is but normalized
            pass
    
    # Convert to POSIX format
    return str(path).replace("\\", "/")


def compute_minimal_imports(code: str, file_ctx: FileContext) -> List[str]:
    """
    Compute minimal imports needed based on qualified identifiers in code.
    
    Args:
        code: Go code content
        file_ctx: File context with available imports
        
    Returns:
        List of import paths that are actually available in the file
    """
    # Clean code before extracting symbols
    cleaned_code = clean_code_for_symbols(code)
    qualified_idents = extract_qualified_identifiers(cleaned_code)
    imports = set()
    
    for ident in qualified_idents:
        if "." in ident:
            pkg = ident.split(".", 1)[0]
            # First try to map through file's import alias map
            if pkg in file_ctx.import_alias_map:
                import_path = file_ctx.import_alias_map[pkg]
                if import_path in file_ctx.file_imports:
                    imports.add(import_path)
            else:
                # Fallback to guessing, but only if it's in file imports
                import_path = guess_import_for_pkg(pkg)
                if import_path and import_path in file_ctx.file_imports:
                    imports.add(import_path)
    
    return sorted(list(imports))


def compute_symbols_referenced(code: str) -> List[str]:
    """
    Compute symbols referenced in the code, excluding Go builtins and stopwords.
    
    Args:
        code: Go code content
        
    Returns:
        List of referenced symbols
    """
    # Clean code before extracting symbols
    cleaned_code = clean_code_for_symbols(code)
    qualified_idents = extract_qualified_identifiers(cleaned_code)
    capitalized_idents = extract_capitalized_identifiers(cleaned_code)
    
    # Combine qualified and capitalized identifiers
    symbols = set(qualified_idents)
    symbols.update(capitalized_idents)
    
    # Remove Go builtins and stopwords
    symbols -= GO_BUILTINS
    symbols -= STOPWORDS
    
    # Filter out all-uppercase words and very short identifiers
    filtered_symbols = set()
    for symbol in symbols:
        if len(symbol) >= 3 and not symbol.isupper():
            filtered_symbols.add(symbol)
        elif "." in symbol:  # Keep qualified identifiers
            filtered_symbols.add(symbol)
    
    # Preserve qualified forms when present
    result = []
    for symbol in sorted(filtered_symbols):
        if "." in symbol:
            result.append(symbol)
        elif symbol not in GO_BUILTINS and symbol not in STOPWORDS:
            result.append(symbol)
    
    return result


def build_header_context_minimal(package: str, imports: List[str], node_kind: str, receiver: str, file_ctx: Optional[FileContext] = None) -> str:
    """
    Build minimal header context for a Go chunk.
    
    Args:
        package: Package name (fallback if file_ctx not provided)
        imports: List of import paths
        node_kind: Kind of node (header, method, function, type, block)
        receiver: Method receiver (if applicable)
        file_ctx: File context (preferred source for package name)
        
    Returns:
        Formatted header context string
    """
    # Prefer file context package name
    actual_package = file_ctx.package_name if file_ctx and file_ctx.package_name else package
    lines = [f"package {actual_package}"]
    
    if imports:
        if len(imports) == 1:
            lines.append(f'import "{imports[0]}"')
        else:
            lines.append("import (")
            for imp in imports[:5]:  # Limit to 5 imports
                lines.append(f'    "{imp}"')
            lines.append(")")
    
    # Add receiver info for methods
    if node_kind == "method" and receiver:
        lines.append(f"// receiver: {receiver}")
    
    return "\n".join(lines)


def generate_file_synopsis(full_file_text: str, file_path: str) -> str:
    """
    Generate a file-level synopsis for context in chunk enrichment.
    
    Args:
        full_file_text: Complete file content
        file_path: File path for context
        
    Returns:
        Short synopsis string (≤250 chars)
    """
    # Extract basic info from file
    package = find_package(full_file_text)
    
    # Count different types of declarations
    method_count = len(re.findall(r'^\s*func\s*\([^)]+\)\s+\w+', full_file_text, re.MULTILINE))
    func_count = len(re.findall(r'^\s*func\s+\w+(?!\s*\([^)]*\)\s+\w+)', full_file_text, re.MULTILINE))
    type_count = len(re.findall(r'^\s*type\s+\w+\s+(struct|interface)', full_file_text, re.MULTILINE))
    
    # Build synopsis
    parts = [f"Go package {package}"]
    
    if type_count > 0:
        parts.append(f"with {type_count} type{'s' if type_count > 1 else ''}")
    if method_count > 0:
        parts.append(f"{method_count} method{'s' if method_count > 1 else ''}")
    if func_count > 0:
        parts.append(f"{func_count} function{'s' if func_count > 1 else ''}")
    
    synopsis = ", ".join(parts) + "."
    
    # Truncate if too long
    if len(synopsis) > 250:
        synopsis = synopsis[:247] + "..."
    
    return synopsis


def infer_go_structure_ts(code: str, file_ctx: Optional[FileContext] = None, 
                          start_byte: Optional[int] = None, end_byte: Optional[int] = None) -> Dict[str, Any]:
    """
    Tree-sitter based Go structure inference.
    
    Args:
        code: Go code content (used as fallback if bytes not provided)
        file_ctx: File context with abs_path and file_text
        start_byte: Chunk start byte offset in file
        end_byte: Chunk end byte offset in file
        
    Returns:
        Dictionary with structural metadata
    """
    if not USE_TREESITTER or not file_ctx or not file_ctx.abs_path or not file_ctx.file_text:
        # Fallback to regex
        return {"node_kind": "unknown", "primary_symbol": "", "ast_path": "go:block", "is_header": False}
    
    try:
        # Get or create index
        idx = _ts_index_cache.get(file_ctx.abs_path)
        if idx is None:
            indexer = GoTSIndexer()
            idx = indexer.parse_file(file_ctx.file_text)
            _ts_index_cache[file_ctx.abs_path] = idx
        
        # Derive byte offsets if not provided
        if start_byte is None or end_byte is None:
            # Try to derive from code position in file
            code_bytes = code.encode('utf-8')
            file_bytes = file_ctx.file_text.encode('utf-8')
            start_byte = file_bytes.find(code_bytes)
            if start_byte == -1:
                # Can't locate chunk in file, fall back
                return {"node_kind": "unknown", "primary_symbol": "", "ast_path": "go:block", "is_header": False}
            end_byte = start_byte + len(code_bytes)
        
        # Locate chunk in AST
        indexer = GoTSIndexer()
        mapping = indexer.locate_for_chunk(idx, start_byte, end_byte)
        
        # Add package info
        mapping["package"] = idx.package_name or ""
        
        return mapping
        
    except Exception as e:
        # Log warning and fall back
        print(f"Warning: Tree-sitter parsing failed for {file_ctx.abs_path}: {e}")
        return {"node_kind": "unknown", "primary_symbol": "", "ast_path": "go:block", "is_header": False}


def infer_go_structure(code: str, file_ctx: Optional[FileContext] = None, chunks: Optional[List[Dict]] = None, current_index: int = 0, start_byte: Optional[int] = None, end_byte: Optional[int] = None) -> Dict[str, Any]:
    """
    Infer Go structure from code and return comprehensive metadata.
    Uses Tree-sitter exclusively for structure detection.
    
    Args:
        code: Go code content
        file_ctx: File context (required for Tree-sitter)
        chunks: List of chunks (deprecated, not used with Tree-sitter)
        current_index: Index of current chunk (deprecated, not used with Tree-sitter)
        start_byte: Chunk start byte offset (for Tree-sitter)
        end_byte: Chunk end byte offset (for Tree-sitter)
        
    Returns:
        Dictionary with structural metadata
    """
    # Use Tree-sitter exclusively
    if USE_TREESITTER and file_ctx and file_ctx.abs_path and file_ctx.file_text:
        return infer_go_structure_ts(code, file_ctx, start_byte, end_byte)
    
    # If Tree-sitter not available, return unknown
    # This should not happen in production as Tree-sitter is required
    print(f"Warning: Tree-sitter not available for Go enrichment. Install tree-sitter-languages.")
    return {
        "package": file_ctx.package_name if file_ctx else find_package(code),
        "node_kind": "unknown",
        "receiver": "",
        "method_name": "",
        "function_name": "",
        "type_name": "",
        "type_kind": "",
        "ast_path": "go:block",
        "primary_symbol": "",
        "is_header": False,
    }
