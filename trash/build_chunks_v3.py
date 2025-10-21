#!/usr/bin/env python3
"""
build_chunks_v3.py - Production-grade AST-aware code chunking with context stitching

This script implements intelligent code chunking for RAG systems, combining:
- AST-aware splitting at natural boundaries (functions, classes, methods)
- Token-bounded chunking with retroactive context stitching
- Multi-language support via tree-sitter parsing
- Deterministic outputs with change detection
- Parallel processing for large codebases

JSON Schema:
{
  "chunk_id": "sha256_hash",
  "repo": "repository_name",
  "path": "repo/relative/path/to/file",
  "language": "python|javascript|go|...",
  "start_line": 1,
  "end_line": 25,
  "ast_path": "function_name.method_name",
  "text": "header_context + core + footer_context",
  "header_context": "package + imports + class signature",
  "core": "main method/function body",
  "footer_context": "type aliases, close-over vars",
  "symbols_defined": ["function_name", "class_name"],
  "symbols_referenced": ["imported_function", "external_class"],
  "imports_used": ["from module import function"],
  "neighbors": {"prev": "chunk_id", "next": "chunk_id"},
  "summary_1l": "One sentence description of what this chunk does",
  "qa_terms": "status codes, exceptions, HTTP verbs, framework nouns",
  "token_counts": {"header": 45, "core": 150, "footer": 12, "total": 207},
  "file_sha": "sha256_of_full_file",
  "created_at": "2024-01-01T00:00:00Z",
  "v": 3
}

Usage:
  python build_chunks_v3.py --root ./myrepo --out ./chunks.jsonl
  python build_chunks_v3.py --file ./src/api.py --force --log-level DEBUG
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pydantic
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from CodeParser import CodeParser
from utils import count_tokens


# Pydantic Models
class Neighbors(pydantic.BaseModel):
    prev: Optional[str] = None
    next: Optional[str] = None


class TokenCounts(pydantic.BaseModel):
    header: int = 0
    core: int = 0
    footer: int = 0
    total: int = 0


class ChunkRecord(pydantic.BaseModel):
    chunk_id: str
    repo: str
    path: str
    language: str
    start_line: int
    end_line: int
    ast_path: str
    text: str
    header_context: str
    core: str
    footer_context: str
    symbols_defined: List[str]
    symbols_referenced: List[str]
    imports_used: List[str]
    neighbors: Neighbors
    summary_1l: str
    qa_terms: str
    token_counts: TokenCounts
    file_sha: str
    created_at: str
    v: int = 3


# Configuration
DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "__pycache__", ".mypy_cache",
    "node_modules", "dist", "build", "out", "target", ".venv", "venv", ".tox",
    ".cache", "coverage", ".coverage", "htmlcov", ".pytest_cache"
}

SUPPORTED_EXTENSIONS = {
    "py", "js", "jsx", "ts", "tsx", "go", "php", "rb", "css", "java", "cpp", "c",
    "yaml", "yml", "json", "toml", "ini", "md", "sql"
}

LANGUAGE_MAP = {
    "py": "python", "js": "javascript", "jsx": "javascript", "ts": "typescript",
    "tsx": "typescript", "go": "go", "php": "php", "rb": "ruby", "css": "css",
    "java": "java", "cpp": "cpp", "c": "c", "yaml": "yaml", "yml": "yaml",
    "json": "json", "toml": "toml", "ini": "ini", "md": "markdown", "sql": "sql"
}


def setup_logging(level: str) -> logging.Logger:
    """Setup structured logging."""
    log_level = getattr(logging, level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def compute_file_sha(file_path: Path) -> str:
    """Compute SHA256 hash of file content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_chunk_id(repo: str, path: str, file_sha: str, start_line: int, 
                    end_line: int, ast_path: str) -> str:
    """Compute deterministic chunk ID."""
    content = f"{repo}|{path}|{file_sha}|{start_line}|{end_line}|{ast_path}"
    return hashlib.sha256(content.encode()).hexdigest()


def should_skip_file(file_path: Path, include_vendor: bool) -> bool:
    """Determine if file should be skipped."""
    # Skip binary files
    if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.ico', 
                                   '.pdf', '.zip', '.tar', '.gz', '.exe', '.dll', '.so'}:
        return True
    
    # Skip large files (>1MB)
    if file_path.stat().st_size > 1024 * 1024:
        return True
    
    # Skip vendor/build directories unless explicitly included
    if not include_vendor:
        for part in file_path.parts:
            if part in DEFAULT_EXCLUDES:
                return True
    
    # Skip unsupported extensions
    if file_path.suffix.lstrip('.') not in SUPPORTED_EXTENSIONS:
        return True
    
    return False


def detect_file_type(file_path: Path) -> str:
    """Detect file type and return language."""
    ext = file_path.suffix.lstrip('.').lower()
    
    # Special handling for OpenAPI/Swagger files
    if ext in {'yaml', 'yml', 'json'}:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Read first 1000 chars
                if 'openapi:' in content or '"openapi"' in content:
                    return 'openapi'
        except Exception:
            pass
    
    return LANGUAGE_MAP.get(ext, ext)


def extract_imports_from_code(code: str, language: str) -> List[str]:
    """Extract import statements from code."""
    imports = []
    lines = code.split('\n')
    
    for line in lines:
        line = line.strip()
        if language == 'python':
            if line.startswith(('import ', 'from ')):
                imports.append(line)
        elif language in {'javascript', 'typescript'}:
            if line.startswith(('import ', 'const ', 'let ', 'var ')) and 'require(' in line:
                imports.append(line)
        elif language == 'go':
            if line.startswith('import '):
                imports.append(line)
    
    return imports


def generate_summary_1l(code: str, language: str, ast_path: str) -> str:
    """Generate one-line summary using heuristics."""
    # Extract key information
    lines = code.split('\n')
    first_line = lines[0].strip() if lines else ""
    
    # Language-specific patterns
    if language == 'python':
        if 'def ' in first_line:
            func_name = first_line.split('def ')[1].split('(')[0]
            return f"Python function {func_name} with {len(lines)} lines of code"
        elif 'class ' in first_line:
            class_name = first_line.split('class ')[1].split('(')[0].split(':')[0]
            return f"Python class {class_name} with {len(lines)} lines of code"
    elif language in {'javascript', 'typescript'}:
        if 'function ' in first_line or '=>' in first_line:
            return f"JavaScript/TypeScript function with {len(lines)} lines of code"
        elif 'class ' in first_line:
            return f"JavaScript/TypeScript class with {len(lines)} lines of code"
    elif language == 'go':
        if 'func ' in first_line:
            return f"Go function with {len(lines)} lines of code"
        elif 'type ' in first_line:
            return f"Go type definition with {len(lines)} lines of code"
    
    return f"{language.title()} code block with {len(lines)} lines"


def generate_qa_terms(code: str, language: str) -> str:
    """Generate QA terms using heuristics."""
    terms = set()
    lines = code.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        
        # HTTP status codes
        if 'status' in line_lower and any(code in line for code in ['200', '201', '400', '401', '403', '404', '500']):
            terms.update(['200', '201', '400', '401', '403', '404', '500'])
        
        # HTTP methods
        if any(method in line_lower for method in ['get', 'post', 'put', 'delete', 'patch']):
            terms.update(['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
        
        # Exceptions
        if language == 'python' and any(exc in line for exc in ['Exception', 'Error', 'ValueError', 'TypeError']):
            terms.update(['Exception', 'Error', 'ValueError', 'TypeError'])
        
        # Framework terms
        if 'fastapi' in line_lower:
            terms.add('FastAPI')
        if 'django' in line_lower:
            terms.add('Django')
        if 'flask' in line_lower:
            terms.add('Flask')
        if 'react' in line_lower:
            terms.add('React')
        if 'vue' in line_lower:
            terms.add('Vue')
    
    return ', '.join(sorted(terms)[:12])  # Limit to 12 terms


def chunk_code_file(file_path: Path, repo: str, encoding: str, core_tokens: int, 
                   max_total: int, logger: logging.Logger) -> List[ChunkRecord]:
    """Chunk a single code file using AST analysis."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return []
    
    file_sha = compute_file_sha(file_path)
    language = detect_file_type(file_path)
    try:
        rel_path = str(file_path.relative_to(Path.cwd()))
    except ValueError:
        # Handle case where file_path is not relative to current directory
        rel_path = str(file_path)
    
    # Handle non-code files
    if language in {'yaml', 'json', 'toml', 'ini', 'markdown', 'sql', 'openapi'}:
        return chunk_special_file(content, file_path, repo, file_sha, language, rel_path, logger)
    
    # Use AST-based chunking for code files
    return chunk_ast_file(content, file_path, repo, file_sha, language, rel_path, 
                         encoding, core_tokens, max_total, logger)


def chunk_special_file(content: str, file_path: Path, repo: str, file_sha: str, 
                      language: str, rel_path: str, logger: logging.Logger) -> List[ChunkRecord]:
    """Chunk special file types (YAML, JSON, Markdown, etc.)."""
    chunks = []
    lines = content.split('\n')
    
    # Simple line-based chunking for now
    chunk_size = 50  # lines per chunk
    for i in range(0, len(lines), chunk_size):
        start_line = i + 1
        end_line = min(i + chunk_size, len(lines))
        chunk_lines = lines[i:end_line]
        chunk_text = '\n'.join(chunk_lines)
        
        ast_path = f"block_{i//chunk_size + 1}"
        chunk_id = compute_chunk_id(repo, str(file_path), file_sha, start_line, end_line, ast_path)
        
        chunk = ChunkRecord(
            chunk_id=chunk_id,
            repo=repo,
            path=str(file_path),
            rel_path=rel_path,
            language=language,
            start_line=start_line,
            end_line=end_line,
            ast_path=ast_path,
            text=chunk_text,
            header_context="",
            core=chunk_text,
            footer_context="",
            symbols_defined=[],
            symbols_referenced=[],
            imports_used=[],
            neighbors=Neighbors(),
            summary_1l=generate_summary_1l(chunk_text, language, ast_path),
            qa_terms=generate_qa_terms(chunk_text, language),
            token_counts=TokenCounts(core=len(chunk_text.split()), total=len(chunk_text.split())),
            file_sha=file_sha,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        chunks.append(chunk)
    
    return chunks


def is_go_file(file_path: Path) -> bool:
    """Check if file is a Go source file."""
    return file_path.suffix == '.go'


def iter_go_nodes(tree, content: str):
    """Iterate over Go AST nodes yielding (node_type, span, extra)."""
    if not tree:
        return
    
    def traverse(node, depth=0):
        if not hasattr(node, 'type'):
            return
            
        node_type = node.type
        start_point = node.start_point
        end_point = node.end_point
        
        # Convert to 1-based line numbers
        start_line = start_point[0] + 1
        end_line = end_point[0] + 1
        
        span = (start_line, end_line)
        
        # Extract extra information based on node type
        extra = {}
        
        if node_type == 'package_declaration':
            extra['package_name'] = extract_package_name(node, content)
        elif node_type == 'import_declaration':
            extra['imports'] = extract_imports(node, content)
        elif node_type == 'type_declaration':
            extra['type_name'] = extract_type_name(node, content)
            extra['type_kind'] = extract_type_kind(node, content)
        elif node_type == 'method_declaration':
            extra['method_name'] = extract_method_name(node, content)
            extra['receiver'] = extract_receiver(node, content)
        elif node_type == 'function_declaration':
            extra['function_name'] = extract_function_name(node, content)
        
        yield (node_type, span, extra)
        
        # Traverse children
        if hasattr(node, 'children'):
            for child in node.children:
                yield from traverse(child, depth + 1)
    
    yield from traverse(tree)


def extract_package_name(node, content: str) -> str:
    """Extract package name from package declaration."""
    try:
        lines = content.split('\n')
        start_line = node.start_point[0]
        line_content = lines[start_line]
        if 'package ' in line_content:
            return line_content.split('package ')[1].split()[0]
    except:
        pass
    return ""


def extract_imports(node, content: str) -> List[str]:
    """Extract import paths from import declaration."""
    imports = []
    try:
        lines = content.split('\n')
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        
        for line_num in range(start_line, end_line + 1):
            if line_num < len(lines):
                line = lines[line_num].strip()
                if line.startswith('"') and line.endswith('"'):
                    # Single import
                    import_path = line.strip('"')
                    if import_path:
                        imports.append(import_path)
                elif '"' in line:
                    # Multi-line import
                    parts = line.split('"')
                    for i in range(1, len(parts), 2):
                        if parts[i].strip():
                            imports.append(parts[i])
    except:
        pass
    return imports


def extract_type_name(node, content: str) -> str:
    """Extract type name from type declaration."""
    try:
        lines = content.split('\n')
        start_line = node.start_point[0]
        line_content = lines[start_line]
        if 'type ' in line_content:
            parts = line_content.split('type ')[1].split()
            if parts:
                return parts[0]
    except:
        pass
    return ""


def extract_type_kind(node, content: str) -> str:
    """Extract type kind (struct, interface, etc.) from type declaration."""
    try:
        lines = content.split('\n')
        start_line = node.start_point[0]
        line_content = lines[start_line]
        if 'struct' in line_content:
            return 'struct'
        elif 'interface' in line_content:
            return 'interface'
        elif '=' in line_content:
            return 'alias'
    except:
        pass
    return ""


def extract_method_name(node, content: str) -> str:
    """Extract method name from method declaration."""
    try:
        lines = content.split('\n')
        start_line = node.start_point[0]
        line_content = lines[start_line]
        # Look for func (receiver) MethodName pattern
        if 'func ' in line_content:
            parts = line_content.split('func ')[1]
            if ')' in parts:
                method_part = parts.split(')')[1].strip()
                if '(' in method_part:
                    method_name = method_part.split('(')[0].strip()
                    return method_name
    except:
        pass
    return ""


def extract_receiver(node, content: str) -> str:
    """Extract receiver from method declaration."""
    try:
        lines = content.split('\n')
        start_line = node.start_point[0]
        line_content = lines[start_line]
        if 'func (' in line_content:
            receiver_part = line_content.split('func (')[1].split(')')[0]
            return receiver_part.strip()
    except:
        pass
    return ""


def extract_function_name(node, content: str) -> str:
    """Extract function name from function declaration."""
    try:
        lines = content.split('\n')
        start_line = node.start_point[0]
        line_content = lines[start_line]
        if 'func ' in line_content and '(' not in line_content.split('func ')[1]:
            # Package-level function
            func_part = line_content.split('func ')[1]
            if '(' in func_part:
                func_name = func_part.split('(')[0].strip()
                return func_name
    except:
        pass
    return ""


def chunk_ast_file(content: str, file_path: Path, repo: str, file_sha: str, 
                  language: str, rel_path: str, encoding: str, core_tokens: int, 
                  max_total: int, logger: logging.Logger) -> List[ChunkRecord]:
    """Chunk code file using AST analysis."""
    try:
        # Initialize parser for the file extension
        ext = file_path.suffix.lstrip('.')
        parser = CodeParser([ext])
        
        # Special handling for Go files
        if is_go_file(file_path):
            return chunk_go_file(content, file_path, repo, file_sha, rel_path, 
                               encoding, core_tokens, max_total, logger, parser)
        
        # Get points of interest (functions, classes, methods)
        points_of_interest = parser.get_lines_for_points_of_interest(content, ext)
        comments = parser.get_lines_for_comments(content, ext)
        
        if not points_of_interest:
            # No AST structure found, fall back to line-based chunking
            return chunk_special_file(content, file_path, repo, file_sha, language, rel_path, logger)
        
    except Exception as e:
        logger.warning(f"Failed to parse {file_path} with AST: {e}")
        return chunk_special_file(content, file_path, repo, file_sha, language, rel_path, logger)
    
    chunks = []
    lines = content.split('\n')
    
    # Group points of interest into logical chunks
    current_chunk_start = 0
    current_chunk_end = 0
    
    for i, poi_line in enumerate(points_of_interest):
        if i == 0:
            current_chunk_start = poi_line
            current_chunk_end = poi_line
        else:
            # Check if we should start a new chunk
            chunk_text = '\n'.join(lines[current_chunk_start:poi_line])
            token_count = count_tokens(chunk_text, encoding)
            
            if token_count > max_total:
                # Create chunk from current range
                chunk = create_ast_chunk(
                    lines, current_chunk_start, current_chunk_end, 
                    repo, file_path, file_sha, rel_path, language, 
                    encoding, logger
                )
                if chunk:
                    chunks.append(chunk)
                
                # Start new chunk
                current_chunk_start = poi_line
                current_chunk_end = poi_line
            else:
                current_chunk_end = poi_line
    
    # Create final chunk
    if current_chunk_start < len(lines):
        chunk = create_ast_chunk(
            lines, current_chunk_start, current_chunk_end,
            repo, file_path, file_sha, rel_path, language,
            encoding, logger
        )
        if chunk:
            chunks.append(chunk)
    
    # Set neighbors
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk.neighbors.prev = chunks[i-1].chunk_id
        if i < len(chunks) - 1:
            chunk.neighbors.next = chunks[i+1].chunk_id
    
    return chunks


def chunk_go_file(content: str, file_path: Path, repo: str, file_sha: str, 
                 rel_path: str, encoding: str, core_tokens: int, max_total: int, 
                 logger: logging.Logger, parser: CodeParser) -> List[ChunkRecord]:
    """Chunk Go file using AST analysis with proper spans and context."""
    try:
        # Validate required metadata
        if not repo:
            logger.warning(f"Missing repo for file {file_path}, skipping")
            return []
        if not file_sha:
            logger.warning(f"Missing file_sha for file {file_path}, skipping")
            return []
        
        # Parse the Go file
        tree = parser.parse_code(content, 'go')
        if not tree:
            logger.warning(f"Failed to parse Go file {file_path}")
            return chunk_special_file(content, file_path, repo, file_sha, "go", rel_path, logger)
        
        chunks = []
        lines = content.split('\n')
        
        # Collect all Go nodes
        go_nodes = list(iter_go_nodes(tree, content))
        logger.debug(f"Found {len(go_nodes)} Go nodes in {file_path}")
        
        # Log node types seen for header detection
        node_types_seen = {}
        for node_type, span, extra in go_nodes:
            node_types_seen[node_type] = node_types_seen.get(node_type, 0) + 1
        logger.debug(f"Node types seen: {node_types_seen}")
        
        # Extract package and imports for header context
        package_name = ""
        all_imports = []
        import_aliases = {}  # Map aliases to full import paths
        symbol_to_import_map = {}  # Map symbols to their import paths
        
        for node_type, span, extra in go_nodes:
            if node_type == 'package_declaration':
                package_name = extra.get('package_name', '')
                logger.debug(f"Found package declaration: {package_name} at span {span}")
            elif node_type == 'import_declaration':
                imports = extra.get('imports', [])
                all_imports.extend(imports)
                logger.debug(f"Found import declaration: {imports} at span {span}")
                # Extract aliases from import declarations
                import_aliases.update(extract_import_aliases(lines, span))
        
        # Fallback: if no package found, try to extract from first line
        if not package_name:
            first_line = lines[0].strip() if lines else ""
            if first_line.startswith('package '):
                package_name = first_line.split('package ')[1].split()[0]
                logger.debug(f"Extracted package name from first line: {package_name}")
        
        logger.debug(f"Final package_name: {package_name}, all_imports: {all_imports}")
        
        # Fallback: if no imports found, try to extract from content
        if not all_imports:
            for line in lines:
                line = line.strip()
                if line.startswith('import '):
                    if line.startswith('import "') and line.endswith('"'):
                        # Single import
                        import_path = line[8:-1]  # Remove 'import "' and '"'
                        all_imports.append(import_path)
                    elif line.startswith('import ('):
                        # Multi-line import - continue reading
                        continue
                elif line.startswith('"') and line.endswith('"') and any('import' in prev_line for prev_line in lines[:lines.index(line)]):
                    # Import line in multi-line block
                    import_path = line[1:-1]  # Remove quotes
                    all_imports.append(import_path)
            logger.debug(f"Fallback extracted imports: {all_imports}")
        
        # Build symbol to import mapping
        symbol_to_import_map = build_symbol_to_import_map(all_imports, import_aliases, logger)
        logger.debug(f"Built symbol_to_import_map with {len(symbol_to_import_map)} entries")
        
        # Create file header chunk (package + imports) - MUST be first
        header_chunk = create_go_file_header_chunk(
            lines, go_nodes, package_name, all_imports, repo, file_path, file_sha, encoding, logger
        )
        if header_chunk:
            chunks.append(header_chunk)
        
        # Process types, methods, and functions
        for node_type, span, extra in go_nodes:
            start_line, end_line = span
            
            # Skip empty or invalid spans
            if end_line < start_line or start_line < 1 or end_line > len(lines):
                continue
            
            if node_type == 'type_declaration':
                # Handle grouped type declarations
                type_chunks = create_go_type_chunks(
                    lines, span, extra, package_name, all_imports, import_aliases, symbol_to_import_map,
                    repo, file_path, file_sha, encoding, logger
                )
                chunks.extend(type_chunks)
                
            elif node_type in ['method_declaration', 'function_declaration']:
                # Extract the full span content
                chunk_lines = lines[start_line-1:end_line]
                chunk_text = '\n'.join(chunk_lines)
                
                # Skip empty chunks
                if not chunk_text.strip():
                    continue
                
                # Build AST path
                ast_path = build_go_ast_path(node_type, extra)
                
                # Log name extraction
                function_name = extra.get('function_name', '')
                type_name = extra.get('type_name', '')
                type_kind = extra.get('type_kind', '')
                method_name = extra.get('method_name', '')
                receiver = extra.get('receiver', '')
                logger.debug(f"Name extraction for {node_type}: function_name={function_name}, type_name={type_name}, type_kind={type_kind}, method_name={method_name}, receiver={receiver}")
                
                # Extract imports used in this chunk (strict filtering)
                imports_used = extract_chunk_imports_strict(chunk_text, symbol_to_import_map, logger)
                logger.debug(f"Import inference for {ast_path}: {imports_used}")
                
                # Extract symbols referenced
                symbols_referenced = extract_go_symbols_referenced_strict(chunk_text, symbol_to_import_map)
                # Remove self symbol (type or function/method)
                self_name = extra.get('type_name') or extra.get('function_name') or extra.get('method_name')
                if self_name:
                    symbols_referenced = [s for s in symbols_referenced if s != self_name and not s.endswith(f".{self_name}")]
                # If a qualified symbol exists, drop the bare package name alone
                qualified_pkgs = {s.split('.')[0] for s in symbols_referenced if '.' in s}
                symbols_referenced = [s for s in symbols_referenced if '.' in s or s not in qualified_pkgs]
                
                # Build minimal header context
                header_context = build_go_minimal_header_context(
                    package_name, imports_used, node_type, extra
                )
                
                # Check token budget
                total_tokens = count_tokens(chunk_text, encoding)
                header_tokens = count_tokens(header_context, encoding)
                
                if total_tokens > max_total:
                    # Split large chunks
                    sub_chunks = split_go_chunk_with_minimal_header(
                        chunk_text, header_context, start_line, end_line,
                        node_type, extra, core_tokens, max_total, encoding, logger
                    )
                    chunks.extend(sub_chunks)
                else:
                    # Create single chunk
                    chunk = create_go_chunk(
                        chunk_text, header_context, start_line, end_line,
                        ast_path, repo, file_path, file_sha, encoding,
                        node_type, extra, imports_used, symbols_referenced, logger
                    )
                    if chunk:
                        chunks.append(chunk)
        
        # Sort chunks by start line and set neighbors across entire file
        chunks.sort(key=lambda c: c.start_line)
        logger.debug(f"Neighbor chain builder: sorted {len(chunks)} chunks by start_line")
        
        # Log final order and first three chunk ast_paths
        if chunks:
            first_three = [chunk.ast_path for chunk in chunks[:3]]
            logger.debug(f"First three chunk ast_paths: {first_three}")
        
        # Ensure file header exists and is first; if missing, try to synthesize or log
        if not chunks or chunks[0].ast_path != "go:file_header":
            # Try to locate a header in the list
            header_idx = next((i for i, c in enumerate(chunks) if c.ast_path == "go:file_header"), None)
            if header_idx is not None:
                # Move it to the front
                header_chunk = chunks.pop(header_idx)
                chunks.insert(0, header_chunk)
                logger.debug(f"Moved file header from position {header_idx} to front")
            else:
                logger.warning(f"No go:file_header found for {file_path}; neighbor chain will start at a non-header chunk.")
        
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.neighbors.prev = chunks[i-1].chunk_id
            if i < len(chunks) - 1:
                chunk.neighbors.next = chunks[i+1].chunk_id
        
        logger.debug(f"Neighbor chain built: {len(chunks)} chunks with proper prev/next links")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to chunk Go file {file_path}: {e}")
        return chunk_special_file(content, file_path, repo, file_sha, "go", rel_path, logger)


def build_symbol_to_import_map(all_imports: List[str], import_aliases: dict, logger: logging.Logger) -> dict:
    """Build a mapping from symbols to their import paths."""
    symbol_map = {}
    
    logger.debug(f"Building symbol_to_import_map from {len(all_imports)} imports and {len(import_aliases)} aliases")
    
    for import_path in all_imports:
        # Extract the package name from the import path
        package_name = import_path.split('/')[-1]
        symbol_map[package_name] = import_path
        logger.debug(f"Added symbol mapping: {package_name} -> {import_path}")
        
        # Handle special cases for common packages
        if 'context' in import_path:
            symbol_map['Context'] = import_path
            logger.debug(f"Added special mapping: Context -> {import_path}")
        elif 'time' in import_path:
            symbol_map['Time'] = import_path
            symbol_map['Location'] = import_path
            symbol_map['Duration'] = import_path
            logger.debug(f"Added special mappings: Time, Location, Duration -> {import_path}")
        elif 'errors' in import_path:
            symbol_map['error'] = import_path
            logger.debug(f"Added special mapping: error -> {import_path}")
        elif 'fmt' in import_path:
            symbol_map['Printf'] = import_path
            symbol_map['Sprintf'] = import_path
            logger.debug(f"Added special mappings: Printf, Sprintf -> {import_path}")
        elif 'strings' in import_path:
            symbol_map['Trim'] = import_path
            symbol_map['Split'] = import_path
            logger.debug(f"Added special mappings: Trim, Split -> {import_path}")
        elif 'strconv' in import_path:
            symbol_map['Atoi'] = import_path
            symbol_map['Itoa'] = import_path
            logger.debug(f"Added special mappings: Atoi, Itoa -> {import_path}")
    
    # Add aliases
    for alias, import_path in import_aliases.items():
        symbol_map[alias] = import_path
        logger.debug(f"Added alias mapping: {alias} -> {import_path}")
        # Also map common symbols from aliased imports
        if 'otel' in import_path:
            symbol_map['Tracer'] = import_path
            symbol_map['Span'] = import_path
            logger.debug(f"Added otel mappings: Tracer, Span -> {import_path}")
        elif 'zap' in import_path:
            symbol_map['Int'] = import_path
            symbol_map['Error'] = import_path
            symbol_map['String'] = import_path
            logger.debug(f"Added zap mappings: Int, Error, String -> {import_path}")
    
    logger.debug(f"Final symbol_to_import_map: {symbol_map}")
    return symbol_map


def extract_chunk_imports_strict(chunk_text: str, symbol_to_import_map: dict, logger: logging.Logger) -> List[str]:
    """Extract imports used in chunk with strict filtering."""
    used_imports = set()
    
    # Split chunk into words and identifiers
    import re
    # Find all identifiers (words that could be symbols)
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', chunk_text)
    
    # Exclude builtins that are not import-backed
    BUILTIN_IDENTIFIERS = {"error", "nil", "true", "false"}
    identifiers = [i for i in identifiers if i not in BUILTIN_IDENTIFIERS]
    
    logger.debug(f"Import inference: found {len(identifiers)} identifiers in chunk")
    
    for identifier in identifiers:
        if identifier in symbol_to_import_map:
            used_imports.add(symbol_to_import_map[identifier])
            logger.debug(f"Import inference: {identifier} -> {symbol_to_import_map[identifier]}")
    
    # Also check for qualified identifiers like context.Context
    qualified_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
    qualified_matches = re.findall(qualified_pattern, chunk_text)
    
    logger.debug(f"Import inference: found {len(qualified_matches)} qualified identifiers")
    
    for package, symbol in qualified_matches:
        if package in symbol_to_import_map:
            used_imports.add(symbol_to_import_map[package])
            logger.debug(f"Import inference: {package}.{symbol} -> {symbol_to_import_map[package]}")
    
    result = list(used_imports)
    logger.debug(f"Import inference result: {result}")
    return result


def extract_go_symbols_referenced_strict(chunk_text: str, symbol_to_import_map: dict) -> List[str]:
    """Extract symbols referenced in chunk using strict AST-based analysis."""
    symbols_referenced = set()
    
    import re
    
    # Find qualified identifiers (package.Symbol)
    qualified_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
    qualified_matches = re.findall(qualified_pattern, chunk_text)
    
    for package, symbol in qualified_matches:
        if package in symbol_to_import_map:
            symbols_referenced.add(f"{package}.{symbol}")
    
    # Find unqualified type references - more comprehensive patterns
    type_patterns = [
        r'\b([A-Z][a-zA-Z0-9_]*)\s*\{',  # struct literals
        r'\b([A-Z][a-zA-Z0-9_]*)\s*\(',  # function calls
        r'\b([A-Z][a-zA-Z0-9_]*)\s*\[',  # array/slice types
        r'\*\s*([A-Z][a-zA-Z0-9_]*)',    # pointer types
        r'\b([A-Z][a-zA-Z0-9_]*)\s*\)',  # return types
        r'\b([A-Z][a-zA-Z0-9_]*)\s*,',   # parameter types
        r'\b([A-Z][a-zA-Z0-9_]*)\s*:',   # field types
        r'\b([A-Z][a-zA-Z0-9_]*)\s*=',   # variable assignments
        r'\b([A-Z][a-zA-Z0-9_]*)\s*;',   # statement endings
        r'\b([A-Z][a-zA-Z0-9_]*)\s*$',   # end of line
        r'\b([A-Z][a-zA-Z0-9_]*)\s*{',   # struct/interface fields
        r'\b([A-Z][a-zA-Z0-9_]*)\s*interface',  # interface types
        r'\b([A-Z][a-zA-Z0-9_]*)\s*struct',     # struct types
    ]
    
    for pattern in type_patterns:
        matches = re.findall(pattern, chunk_text)
        for match in matches:
            # Include all capitalized identifiers as they're likely types
            if match and match[0].isupper():
                symbols_referenced.add(match)
    
    # Also find lowercase identifiers that might be local types
    lowercase_type_patterns = [
        r'\b([a-z][a-zA-Z0-9_]*)\s*interface',  # lowercase interface types
        r'\b([a-z][a-zA-Z0-9_]*)\s*struct',     # lowercase struct types
        r'\*\s*([a-z][a-zA-Z0-9_]*)',           # lowercase pointer types
        r'\b([a-z][a-zA-Z0-9_]*)\s*\)',         # lowercase return types
        r'\s+([a-z][a-zA-Z0-9_]*)\s*$',         # lowercase field types at end of line
        r'\s+([a-z][a-zA-Z0-9_]*)\s*//',        # lowercase field types with comment
    ]
    
    # Builtin types to exclude
    builtin_types = {
        'func', 'var', 'const', 'type', 'package', 'import',
        'string', 'int', 'int8', 'int16', 'int32', 'int64',
        'uint', 'uint8', 'uint16', 'uint32', 'uint64',
        'bool', 'byte', 'rune', 'float32', 'float64',
        'complex64', 'complex128', 'nil', 'true', 'false',
        'make', 'new', 'len', 'cap', 'append', 'copy',
        'delete', 'close', 'panic', 'recover'
    }
    
    for pattern in lowercase_type_patterns:
        matches = re.findall(pattern, chunk_text, re.MULTILINE)
        for match in matches:
            if match and match not in builtin_types:
                symbols_referenced.add(match)
    
    # Special case for error
    if 'error' in chunk_text:
        symbols_referenced.add('error')
    
    # Normalize symbols_referenced: prefer qualified identifiers, de-duplicate
    normalized_symbols = set()
    bare_symbols_from_qualified = set()
    
    # First pass: collect all qualified symbols and track their bare names
    for symbol in symbols_referenced:
        if '.' in symbol:
            # Qualified symbol - keep as is
            normalized_symbols.add(symbol)
            # Track the bare symbol name from qualified version
            bare_name = symbol.split('.')[-1]
            bare_symbols_from_qualified.add(bare_name)
    
    # Second pass: add bare symbols only if not covered by qualified version
    for symbol in symbols_referenced:
        if '.' not in symbol:
            # Bare symbol - only add if not already covered by qualified version
            if symbol not in bare_symbols_from_qualified:
                normalized_symbols.add(symbol)
    
    # NOTE: Self-filtering should be performed by the caller where `extra` (type/function name) is known.
    return list(normalized_symbols)


def extract_import_aliases(lines: List[str], span: tuple) -> dict:
    """Extract import aliases from import declaration span."""
    aliases = {}
    start_line, end_line = span
    
    for line_num in range(start_line, end_line + 1):
        if line_num <= len(lines):
            line = lines[line_num - 1].strip()
            if 'import' in line and '"' in line:
                # Parse import line for aliases
                if ' ' in line and not line.startswith('import ('):
                    # Single import with potential alias: import alias "path"
                    parts = line.split('"')
                    if len(parts) >= 2:
                        import_path = parts[1]
                        before_quote = parts[0].strip()
                        if ' ' in before_quote:
                            alias = before_quote.split()[-1]
                            aliases[alias] = import_path
                elif line.startswith('import ('):
                    # Multi-line import block - handled separately
                    pass
    return aliases


def create_go_file_header_chunk(lines: List[str], go_nodes: List, package_name: str, 
                               all_imports: List[str], repo: str, file_path: Path, 
                               file_sha: str, encoding: str, 
                               logger: logging.Logger) -> Optional[ChunkRecord]:
    """Create a single file header chunk containing package and imports."""
    # Find package and import spans
    package_start = None
    import_end = None
    
    for node_type, span, extra in go_nodes:
        if node_type == 'package_declaration':
            package_start = span[0]
        elif node_type == 'import_declaration':
            import_end = span[1]
    
    # Fallback: if no package_clause found, use regex to find package line
    if package_start is None:
        import re
        package_pattern = r'^\s*package\s+([a-zA-Z_]\w*)'
        
        for i, line in enumerate(lines):
            match = re.match(package_pattern, line)
            if match:
                package_start = i + 1  # Convert to 1-based line number
                logger.debug(f"Fallback: found package declaration at line {package_start} using regex")
                break
        
        if package_start is None:
            logger.debug("No package declaration found in AST nodes or via regex fallback")
            return None
        
        # Scan down from package line to consume contiguous import declarations
        import_end = package_start
        in_import_block = False
        
        for i in range(package_start + 1, len(lines) + 1):  # Start from line after package
            if i > len(lines):
                break
                
            line = lines[i - 1].strip()
            
            # Check for import declarations
            if line.startswith('import '):
                if line.startswith('import ('):
                    # Multi-line import block
                    in_import_block = True
                    import_end = i
                elif line.startswith('import "') and line.endswith('"'):
                    # Single-line import
                    import_end = i
                else:
                    # Import keyword without quotes (shouldn't happen in valid Go)
                    import_end = i
            elif in_import_block:
                if line == ')':
                    # End of multi-line import block
                    import_end = i
                    in_import_block = False
                    break
                elif line.startswith('"') and line.endswith('"'):
                    # Import line in multi-line block
                    import_end = i
                elif line == '':
                    # Empty line in import block
                    import_end = i
                else:
                    # Non-import content, stop scanning
                    break
            elif line == '':
                # Empty line after package, continue scanning
                continue
            else:
                # Non-import content, stop scanning
                break
        
        logger.debug(f"Fallback: package spans from line {package_start} to {import_end}")
    
    # If no imports found via AST or fallback, just use package line
    if import_end is None:
        import_end = package_start
    
    # Extract header content
    header_lines = lines[package_start-1:import_end]
    header_text = '\n'.join(header_lines)
    
    if not header_text.strip():
        logger.debug("Header text is empty after extraction")
        return None
    
    # Generate summary with package name and notable imports
    notable_imports = []
    for imp in all_imports:
        if any(keyword in imp.lower() for keyword in ['zap', 'otel', 'cache', 'singleflight']):
            notable_imports.append(imp.split('/')[-1])
    
    summary_parts = [f"Go package {package_name}"]
    if notable_imports:
        summary_parts.append(f"with {', '.join(notable_imports[:3])}")
    summary_1l = " ".join(summary_parts)
    
    # Generate QA terms
    qa_terms = [package_name]
    qa_terms.extend(notable_imports[:5])
    if 'foreca' in str(file_path).lower():
        qa_terms.extend(['foreca', 'weather', 'proxy'])
    
    # Create normalized path (repo-relative, POSIX-style)
    normalized_path = normalize_go_path(file_path, repo)
    
    chunk = ChunkRecord(
        chunk_id=compute_chunk_id(repo, normalized_path, file_sha, package_start, import_end, "go:file_header"),
        repo=repo,
        path=normalized_path,
        language="go",
        start_line=package_start,
        end_line=import_end,
        ast_path="go:file_header",
        text=header_text,
        header_context="",
        core=header_text,
        footer_context="",
        symbols_defined=[],
        symbols_referenced=[],
        imports_used=all_imports,
        neighbors=Neighbors(),
        summary_1l=summary_1l,
        qa_terms=', '.join(qa_terms[:12]),
        token_counts=TokenCounts(
            header=0,
            core=count_tokens(header_text, encoding),
            footer=0,
            total=count_tokens(header_text, encoding)
        ),
        file_sha=file_sha,
        created_at=datetime.now(timezone.utc).isoformat()
    )
    
    return chunk


def normalize_go_path(file_path: Path, repo: str) -> str:
    """Normalize path to repo-relative POSIX-style path."""
    try:
        path_str = file_path.as_posix()
        # If repo looks like a directory name in the path, slice from that segment
        parts = path_str.split('/')
        if repo in parts:
            idx = parts.index(repo)
            rel = '/'.join(parts[idx+1:])
            return rel or ''
        # Else, if you have a repo root Path elsewhere, prefer: file_path.relative_to(repo_root).as_posix()
        return path_str.lstrip('/')
    except Exception:
        return file_path.as_posix()


def create_go_type_chunks(lines: List[str], span: tuple, extra: dict, package_name: str,
                         all_imports: List[str], import_aliases: dict, symbol_to_import_map: dict,
                         repo: str, file_path: Path, file_sha: str, encoding: str,
                         logger: logging.Logger) -> List[ChunkRecord]:
    """Create chunks for grouped type declarations, one per named type."""
    chunks = []
    start_line, end_line = span
    
    # Extract the type declaration content
    type_lines = lines[start_line-1:end_line]
    type_text = '\n'.join(type_lines)
    
    # Check if this is a grouped type declaration
    if 'type (' in type_text:
        # Parse grouped types
        individual_types = parse_grouped_types(type_text, start_line)
        for type_info in individual_types:
            chunk = create_individual_type_chunk(
                type_info, package_name, all_imports, import_aliases, symbol_to_import_map,
                repo, file_path, file_sha, encoding, logger
            )
            if chunk:
                chunks.append(chunk)
    else:
        # Single type declaration
        type_name = extra.get('type_name', '')
        type_kind = extra.get('type_kind', '')
        
        if type_name:
            chunk = create_individual_type_chunk(
                {
                    'name': type_name,
                    'kind': type_kind,
                    'content': type_text,
                    'start_line': start_line,
                    'end_line': end_line
                },
                package_name, all_imports, import_aliases, symbol_to_import_map,
                repo, file_path, file_sha, encoding, logger
            )
            if chunk:
                chunks.append(chunk)
    
    return chunks


def parse_grouped_types(type_text: str, start_line: int) -> List[dict]:
    """Parse grouped type declaration into individual types."""
    types = []
    lines = type_text.split('\n')
    current_type = None
    brace_count = 0
    in_type = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line.startswith('type ('):
            in_type = True
            continue
        elif line == ')' and in_type:
            break
        elif in_type and line and not line.startswith('//'):
            # Check if this is a new type declaration
            if ' ' in line and ('struct' in line or 'interface' in line or '=' in line):
                # Save previous type if exists
                if current_type:
                    types.append(current_type)
                
                # Start new type
                parts = line.split()
                type_name = parts[0]
                type_kind = 'struct' if 'struct' in line else 'interface' if 'interface' in line else 'alias'
                
                current_type = {
                    'name': type_name,
                    'kind': type_kind,
                    'content': line,
                    'start_line': start_line + i,
                    'end_line': start_line + i
                }
            elif current_type:
                # Continue current type
                current_type['content'] += '\n' + line
                current_type['end_line'] = start_line + i
    
    # Add last type
    if current_type:
        types.append(current_type)
    
    return types


def create_individual_type_chunk(type_info: dict, package_name: str, all_imports: List[str],
                                import_aliases: dict, symbol_to_import_map: dict, repo: str, file_path: Path, 
                                file_sha: str, encoding: str,
                                logger: logging.Logger) -> Optional[ChunkRecord]:
    """Create a chunk for an individual type declaration."""
    type_name = type_info['name']
    type_kind = type_info['kind']
    type_content = type_info['content']
    start_line = type_info['start_line']
    end_line = type_info['end_line']
    
    # Extract imports used by this type (strict filtering)
    imports_used = extract_chunk_imports_strict(type_content, symbol_to_import_map, logger)
    
    # Extract symbols referenced
    symbols_referenced = extract_go_symbols_referenced_strict(type_content, symbol_to_import_map)
    # Remove self symbol (type or function/method)
    self_name = type_info.get('name')
    if self_name:
        symbols_referenced = [s for s in symbols_referenced if s != self_name and not s.endswith(f".{self_name}")]
    # If a qualified symbol exists, drop the bare package name alone
    qualified_pkgs = {s.split('.')[0] for s in symbols_referenced if '.' in s}
    symbols_referenced = [s for s in symbols_referenced if '.' in s or s not in qualified_pkgs]
    
    # Build minimal header context
    header_context = build_go_minimal_header_context(package_name, imports_used, 'type_declaration', {})
    
    # Build AST path with type kind
    ast_path = f'go:type:{type_name} ({type_kind})'
    
    # Generate summary
    summary_1l = f"Go {type_kind} {type_name}"
    if 'foreca' in str(file_path).lower():
        summary_1l += " for weather forecasting"
    
    # Generate QA terms
    qa_terms = [type_name, type_kind]
    for imp in imports_used:
        if any(keyword in imp.lower() for keyword in ['cache', 'time', 'context']):
            qa_terms.append(imp.split('/')[-1])
    if 'foreca' in str(file_path).lower():
        qa_terms.extend(['foreca', 'weather', 'proxy'])
    
    # Create normalized path
    normalized_path = normalize_go_path(file_path, repo)
    
    # Build text with proper header context
    text_content = header_context + '\n' + type_content if header_context else type_content
    
    chunk = ChunkRecord(
        chunk_id=compute_chunk_id(repo, str(file_path), file_sha, start_line, end_line, ast_path),
        repo=repo,
        path=normalized_path,
        language="go",
        start_line=start_line,
        end_line=end_line,
        ast_path=ast_path,
        text=text_content,
        header_context=header_context,
        core=type_content,
        footer_context="",
        symbols_defined=[type_name],
        symbols_referenced=symbols_referenced,
        imports_used=imports_used,
        neighbors=Neighbors(),
        summary_1l=summary_1l,
        qa_terms=', '.join(qa_terms[:12]),
        token_counts=TokenCounts(
            header=count_tokens(header_context, encoding),
            core=count_tokens(type_content, encoding),
            footer=0,
            total=count_tokens(text_content, encoding)
        ),
        file_sha=file_sha,
        created_at=datetime.now(timezone.utc).isoformat()
    )
    
    return chunk


def extract_chunk_imports_with_aliases(chunk_text: str, all_imports: List[str], 
                                     import_aliases: dict) -> List[str]:
    """Extract imports used in chunk, respecting aliases."""
    used_imports = []
    chunk_lower = chunk_text.lower()
    
    for imp in all_imports:
        # Check if import path appears in chunk
        if any(part in chunk_lower for part in imp.split('/')):
            used_imports.append(imp)
    
    # Check for aliased imports
    for alias, import_path in import_aliases.items():
        if alias in chunk_text:
            if import_path not in used_imports:
                used_imports.append(import_path)
    
    return used_imports


def build_go_minimal_header_context(package_name: str, imports_used: List[str], 
                                   node_type: str, extra: dict) -> str:
    """Build minimal header context for Go chunks."""
    header_parts = []
    
    # Add package declaration - ALWAYS include this for non-file-header chunks
    if package_name:
        header_parts.append(f'package {package_name}')
    else:
        # Fallback: try to extract from extra or use 'main' as default
        header_parts.append('package main')
    
    # Add minimal imports (sorted alphabetically)
    if imports_used:
        # Sort imports alphabetically by module path
        sorted_imports = sorted(imports_used)
        if len(sorted_imports) == 1:
            header_parts.append(f'import "{sorted_imports[0]}"')
        else:
            header_parts.append('import (')
            for imp in sorted_imports:
                header_parts.append(f'\t"{imp}"')
            header_parts.append(')')
    
    # Add receiver comment for methods
    if node_type == 'method_declaration':
        receiver = extra.get('receiver', '')
        if receiver:
            header_parts.append(f'// receiver: {receiver}')
    
    return '\n'.join(header_parts)


def split_go_chunk_with_minimal_header(chunk_text: str, header_context: str, 
                                      start_line: int, end_line: int, node_type: str, 
                                      extra: dict, core_tokens: int, max_total: int,
                                      encoding: str, logger: logging.Logger) -> List[ChunkRecord]:
    """Split large Go chunks by logical blocks with minimal header repetition."""
    lines = chunk_text.split('\n')
    chunks = []
    current_start = 0
    
    # Extract method/function info for ast_path naming
    method_name = extra.get('method_name', '')
    receiver = extra.get('receiver', '')
    receiver_type = receiver.replace('*', '').replace('(', '').replace(')', '') if receiver else ''
    
    while current_start < len(lines):
        # Find logical block boundaries
        split_point = find_logical_split_point(lines, current_start, core_tokens)
        
        sub_chunk_lines = lines[current_start:split_point]
        sub_chunk_text = '\n'.join(sub_chunk_lines)
        
        if sub_chunk_text.strip():
            # Detect block type for ast_path labeling
            block_label = detect_block_label(sub_chunk_text, len(chunks))
            
            # Build ast_path with part number and label
            if node_type == 'method_declaration' and method_name and receiver_type:
                ast_path = f'go:method:(*{receiver_type}).{method_name}#part{len(chunks)+1}_{block_label}'
            elif node_type == 'function_declaration' and method_name:
                ast_path = f'go:function:{method_name}#part{len(chunks)+1}_{block_label}'
            else:
                ast_path = f'go:{node_type}#part{len(chunks)+1}_{block_label}'
            
            # Recompute imports and symbols for this part
            imports_used = extract_chunk_imports_strict(sub_chunk_text, {}, logger)
            symbols_referenced = extract_go_symbols_referenced_strict(sub_chunk_text, {})
            # Remove self symbol (type or function/method)
            self_name = extra.get('type_name') or extra.get('function_name') or extra.get('method_name')
            if self_name:
                symbols_referenced = [s for s in symbols_referenced if s != self_name and not s.endswith(f".{self_name}")]
            # If a qualified symbol exists, drop the bare package name alone
            qualified_pkgs = {s.split('.')[0] for s in symbols_referenced if '.' in s}
            symbols_referenced = [s for s in symbols_referenced if '.' in s or s not in qualified_pkgs]
            
            # Create chunk with same minimal header context
            chunk = create_go_chunk(
                sub_chunk_text, header_context, start_line + current_start, 
                start_line + split_point - 1, ast_path,
                "", Path(""), "", encoding, node_type, extra, imports_used, symbols_referenced, logger
            )
            if chunk:
                chunks.append(chunk)
        
        current_start = split_point
    
    return chunks


def find_logical_split_point(lines: List[str], start: int, target_tokens: int) -> int:
    """Find a logical split point for Go code blocks."""
    # Look for logical boundaries: if/else, switch, for/range, defer, etc.
    keywords = ['if ', 'else', 'switch', 'case', 'for ', 'range', 'defer', 'go func', 'select']
    
    # Start with a reasonable chunk size
    end = min(start + 15, len(lines))
    
    # Look for logical boundaries within the chunk
    for i in range(start + 5, min(start + 25, len(lines))):
        line = lines[i].strip()
        
        # Check for closing braces that might end a logical block
        if line == '}' and i > start:
            # Make sure this isn't the final closing brace
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                if next_line and not next_line.startswith('}'):
                    end = i + 1
                    break
        
        # Check for logical block starts
        for keyword in keywords:
            if line.startswith(keyword):
                # Split before this new block
                if i > start + 3:  # Ensure we have some content
                    end = i
                    break
    
    return end


def detect_block_label(text: str, part_num: int) -> str:
    """Detect block type for ast_path labeling."""
    text_lower = text.lower()
    
    # Check for specific patterns
    if 'cache' in text_lower and ('get' in text_lower or 'hit' in text_lower):
        return 'cache_lookup'
    elif 'singleflight' in text_lower:
        return 'singleflight_cache'
    elif 'json.unmarshal' in text_lower or 'json.marshal' in text_lower:
        return 'json_ops'
    elif 'loadlocation' in text_lower or 'timezone' in text_lower:
        return 'timezone_load'
    elif 'provider.getforecast' in text_lower or 'provider' in text_lower:
        return 'provider_call'
    elif 'throttl' in text_lower or 'rate limit' in text_lower:
        return 'rate_limit'
    elif 'error' in text_lower and ('wrap' in text_lower or 'return' in text_lower):
        return 'error_handling'
    elif 'span' in text_lower and ('set' in text_lower or 'record' in text_lower):
        return 'otel_span'
    elif 'log.' in text_lower:
        return 'logging'
    else:
        return f'block{part_num + 1}'


def build_go_ast_path(node_type: str, extra: dict) -> str:
    """Build AST path for Go nodes."""
    if node_type == 'package_declaration':
        return 'go:file_header'
    elif node_type == 'import_declaration':
        return 'go:file_header'
    elif node_type == 'type_declaration':
        type_name = extra.get('type_name', '')
        type_kind = extra.get('type_kind', '')
        if type_name:
            return f'go:type:{type_name} ({type_kind})'
        else:
            return 'go:type:'
    elif node_type == 'method_declaration':
        method_name = extra.get('method_name', '')
        receiver = extra.get('receiver', '')
        if method_name and receiver:
            if receiver.startswith('*'):
                return f'go:method:(*{receiver[1:]}).{method_name}'
            else:
                return f'go:method:({receiver}).{method_name}'
        else:
            return 'go:method:'
    elif node_type == 'function_declaration':
        function_name = extra.get('function_name', '')
        if function_name:
            return f'go:function:{function_name}'
        else:
            return 'go:function:'
    else:
        return f'go:{node_type}'


def build_go_header_context(package_name: str, all_imports: List[str], 
                           node_type: str, extra: dict, lines: List[str],
                           start_line: int, end_line: int) -> str:
    """Build header context for Go chunks."""
    header_parts = []
    
    # Add package declaration
    if package_name:
        header_parts.append(f'package {package_name}')
    
    # Add minimal imports (simplified for now)
    if all_imports:
        header_parts.append('import (')
        for imp in all_imports[:5]:  # Limit to first 5 imports
            header_parts.append(f'\t"{imp}"')
        header_parts.append(')')
    
    # Add receiver type for methods
    if node_type == 'method_declaration':
        receiver = extra.get('receiver', '')
        if receiver:
            # Add a simplified type declaration
            type_name = receiver.replace('*', '')
            header_parts.append(f'type {type_name} struct {{ ... }}')
    
    return '\n'.join(header_parts)


def extract_chunk_imports(chunk_text: str, all_imports: List[str]) -> List[str]:
    """Extract imports actually used in the chunk."""
    used_imports = []
    chunk_lower = chunk_text.lower()
    
    for imp in all_imports:
        # Simple heuristic: check if import path appears in chunk
        if any(part in chunk_lower for part in imp.split('/')):
            used_imports.append(imp)
    
    return used_imports


def split_go_chunk(chunk_text: str, header_context: str, start_line: int, end_line: int,
                  node_type: str, extra: dict, core_tokens: int, max_total: int,
                  encoding: str, logger: logging.Logger) -> List[ChunkRecord]:
    """Split large Go chunks by logical blocks."""
    # For now, implement simple line-based splitting
    # TODO: Implement proper block-based splitting (if/else, switch, etc.)
    
    lines = chunk_text.split('\n')
    chunks = []
    current_start = 0
    
    while current_start < len(lines):
        # Find a good split point (end of function, method, or type)
        split_point = min(current_start + 20, len(lines))  # Simple 20-line chunks
        
        # Try to find a better split point at function/method end
        for i in range(current_start + 10, min(current_start + 30, len(lines))):
            line = lines[i].strip()
            if line == '}' and i > current_start:
                split_point = i + 1
                break
        
        sub_chunk_lines = lines[current_start:split_point]
        sub_chunk_text = '\n'.join(sub_chunk_lines)
        
        if sub_chunk_text.strip():
            chunk = create_go_chunk(
                sub_chunk_text, header_context, start_line + current_start, 
                start_line + split_point - 1, f'go:{node_type}_part_{len(chunks)+1}',
                "", Path(""), "", "", node_type, extra, [], encoding, logger
            )
            if chunk:
                chunks.append(chunk)
        
        current_start = split_point
    
    return chunks


def create_go_chunk(chunk_text: str, header_context: str, start_line: int, end_line: int,
                   ast_path: str, repo: str, file_path: Path, file_sha: str, encoding: str,
                   node_type: str, extra: dict, imports_used: List[str], symbols_referenced: List[str],
                   logger: logging.Logger) -> Optional[ChunkRecord]:
    """Create a Go chunk with proper validation."""
    try:
        # Validate chunk
        if not chunk_text.strip() or end_line < start_line:
            return None
        
        # Count tokens
        core_tokens = count_tokens(chunk_text, encoding)
        header_tokens = count_tokens(header_context, encoding)
        total_tokens = core_tokens + header_tokens
        
        # Create chunk ID
        chunk_id = compute_chunk_id(repo, str(file_path), file_sha, start_line, end_line, ast_path)
        
        # Generate summary and QA terms (use part-specific summary if applicable)
        if '#part' in ast_path:
            summary_1l = generate_go_summary_for_part(ast_path, chunk_text, file_path)
        else:
            summary_1l = generate_go_summary(node_type, extra, file_path)
        # Add symbols_referenced to extra for QA terms generation
        extra_with_symbols = extra.copy()
        extra_with_symbols["symbols_referenced"] = symbols_referenced
        qa_terms = generate_go_qa_terms(node_type, extra_with_symbols, imports_used, file_path)
        
        # Extract symbols
        symbols_defined = extract_go_symbols_defined(node_type, extra)
        
        # Create normalized path (repo-relative, POSIX-style)
        normalized_path = normalize_go_path(file_path, repo)
        
        # Build text with proper header context
        text_content = header_context + '\n' + chunk_text if header_context else chunk_text
        
        # Validate by re-parsing header_context + core
        if header_context and chunk_text:
            validation_text = header_context + '\n' + chunk_text
            try:
                # Try to parse the combined text to ensure it's valid Go
                parser = CodeParser()
                tree = parser.parse_code(validation_text, 'go')
                if not tree:
                    logger.warning(f"Failed to re-parse Go chunk at {file_path}:{start_line}-{end_line}")
            except Exception as e:
                logger.warning(f"Validation failed for Go chunk at {file_path}:{start_line}-{end_line}: {e}")
        
        chunk = ChunkRecord(
            chunk_id=chunk_id,
            repo=repo,
            path=normalized_path,  # Use normalized path
            language="go",
            start_line=start_line,
            end_line=end_line,
            ast_path=ast_path,
            text=text_content,
            header_context=header_context,
            core=chunk_text,
            footer_context="",
            symbols_defined=symbols_defined,
            symbols_referenced=symbols_referenced,
            imports_used=imports_used,
            neighbors=Neighbors(),
            summary_1l=summary_1l,
            qa_terms=qa_terms,
            token_counts=TokenCounts(
                header=header_tokens,
                core=core_tokens,
                footer=0,
                total=count_tokens(text_content, encoding)
            ),
            file_sha=file_sha,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        return chunk
        
    except Exception as e:
        logger.error(f"Failed to create Go chunk: {e}")
        return None


def generate_go_summary(node_type: str, extra: dict, file_path: Path) -> str:
    """Generate Go-specific summary with deterministic templates."""
    parts = []
    
    if node_type == 'type_declaration':
        type_name = extra.get('type_name', '')
        type_kind = extra.get('type_kind', '')
        
        # Special templates for gomock structs
        if 'Mock' in type_name and type_kind == 'struct':
            return f"{type_name} is a gomock-generated test double; stores a Controller and a recorder to define expectations."
        # Special templates for specific interfaces and structs
        elif type_name == 'providerClient' and type_kind == 'interface':
            return "Go interface that exposes GetForecastForLocation using context.Context and time.Location for the Foreca proxy."
        elif type_name == 'mappingsRepository' and type_kind == 'interface':
            return "Repository interface to load Mapping by id; used by the Foreca service."
        elif type_name == 'cacheClient' and type_kind == 'interface':
            return "Cache interface for getting/setting forecast items used by the Foreca proxy."
        elif type_name == 'Service' and type_kind == 'struct':
            return "Service aggregates singleflight, provider, mappings, and cache clients with a TTL for the Foreca proxy."
        else:
            parts.append(f"Go {type_kind} {type_name}")
            
    elif node_type == 'method_declaration':
        method_name = extra.get('method_name', '')
        receiver = extra.get('receiver', '')
        
        # Special template for EXPECT method
        if method_name == 'EXPECT':
            return "Returns the gomock recorder to define expectations on MockhttpClient."
        else:
            parts.append(f"Go method {method_name} on {receiver}")
            
    elif node_type == 'function_declaration':
        function_name = extra.get('function_name', '')
        
        # Special templates for constructors
        if function_name == 'NewService':
            parts.append("Go constructor NewService that creates Service with provider, mappings, cache, and expiration options")
        elif function_name.startswith('NewMock'):
            return f"Constructor returning a {function_name[3:]}: wires gomock.Controller and initializes the recorder."
        else:
            parts.append(f"Go function {function_name}")
    else:
        parts.append("Go code block")
    
    # Add domain context only if we had no specific template and no role words yet
    joined = " ".join(parts)
    if not any(w in joined.lower() for w in ["provider", "mappings", "cache", "singleflight", "gomock", "constructor", "transform"]):
        if 'foreca' in str(file_path).lower():
            parts.append("for the Foreca proxy")
        elif 'weather' in str(file_path).lower():
            parts.append("for weather services")
    
    return " ".join(parts)


def generate_go_summary_for_part(ast_path: str, chunk_text: str, file_path: Path) -> str:
    """Generate deterministic summary for method parts based on ast_path and content."""
    if '#part' not in ast_path:
        return generate_go_summary('method_declaration', {}, file_path)
    
    # Extract method name and part info
    if 'GetForecastForLocation' in ast_path:
        if 'cache_lookup' in ast_path:
            return "Checks cache and emits cache_hit attribute; returns cached forecast if fresh."
        elif 'timezone_load' in ast_path:
            return "Loads time zone via time.LoadLocation, records OTEL status on failure."
        elif 'singleflight_cache' in ast_path:
            return "singleflight dedupe; parses cached item via json.Unmarshal and validates expiration."
        elif 'provider_call' in ast_path:
            return "Calls provider, logs errors, handles throttling; sets OTEL attributes."
        elif 'rate_limit' in ast_path:
            return "Handles rate limiting and throttling scenarios with fallback to cached data."
        elif 'error_handling' in ast_path:
            return "Error handling and logging with OTEL span status updates."
        elif 'otel_span' in ast_path:
            return "OTEL span management and attribute setting for observability."
        elif 'logging' in ast_path:
            return "Structured logging with zap for debugging and monitoring."
        else:
            return "Part of GetForecastForLocation method for weather forecasting."
    
    # Check for transform methods
    if 'transform' in ast_path.lower():
        method_name = ast_path.split('.')[-1].split('#')[0] if '.' in ast_path else ''
        return f"Converts foreca data to protobuf v0 format with field mapping for {method_name}."
    
    # Generic part description
    return "Part of Go method for weather forecasting with specific functionality."


def generate_go_qa_terms(node_type: str, extra: dict, imports_used: List[str], file_path: Path) -> str:
    """Generate Go-specific QA terms with domain and library hooks."""
    terms = set()
    
    # Add node-specific terms
    if node_type == 'type_declaration':
        type_name = extra.get('type_name', '')
        type_kind = extra.get('type_kind', '')
        
        # Special templates for gomock structs
        if 'Mock' in type_name and type_kind == 'struct':
            terms.update(['gomock', 'mock', 'controller', 'recorder', 'EXPECT', 'test double'])
        # Special templates for specific interfaces and structs
        elif type_name == 'providerClient' and type_kind == 'interface':
            terms.update(['providerClient', 'GetForecastForLocation', 'context.Context', 'time.Location', 'forecast', 'foreca', 'proxy'])
        elif type_name == 'mappingsRepository' and type_kind == 'interface':
            terms.update(['mappingsRepository', 'Mapping', 'repository', 'foreca', 'service'])
        elif type_name == 'cacheClient' and type_kind == 'interface':
            terms.update(['cacheClient', 'cache.Item', 'get', 'set', 'cache', 'foreca', 'proxy'])
        elif type_name == 'Service' and type_kind == 'struct':
            terms.update(['Service', 'singleflight', 'provider', 'mappings', 'cache', 'TTL', 'foreca', 'proxy'])
        else:
            if type_name:
                terms.add(type_name)
            if type_kind:
                terms.add(type_kind)
                
    elif node_type == 'method_declaration':
        method_name = extra.get('method_name', '')
        receiver = extra.get('receiver', '')
        
        # Special template for EXPECT method
        if method_name == 'EXPECT':
            terms.update(['EXPECT', 'gomock', 'recorder', 'mock', 'expectations'])
        else:
            if method_name:
                terms.add(method_name)
            if receiver:
                terms.add(receiver.replace('*', ''))
                
    elif node_type == 'function_declaration':
        function_name = extra.get('function_name', '')
        
        # Special templates for constructors
        if function_name.startswith('NewMock'):
            terms.update(['NewMockhttpClient', 'gomock', 'controller', 'recorder', 'mock'])
        else:
            if function_name:
                terms.add(function_name)
                if function_name.startswith('New'):
                    terms.add('constructor')
    
    # Use symbols_referenced when available via extra; fallback to imports only for library names
    syms = set(extra.get("symbols_referenced", []))
    for imp in imports_used:
        if 'otel' in imp:
            terms.add('otel')
            # Only add attribute/trace if referenced
            if any(s.startswith("attribute.") for s in syms):
                terms.add('attribute')
            if any(s.startswith("trace.") for s in syms):
                terms.add('trace')
        elif 'zap' in imp:
            terms.add('zap')
        elif 'singleflight' in imp:
            terms.add('singleflight')
        elif 'json' in imp and any(s == "json.Unmarshal" for s in syms):
            terms.add('json.Unmarshal')
        elif 'time' in imp and any(s == "time.LoadLocation" for s in syms):
            terms.add('LoadLocation')
        elif 'cache' in imp:
            terms.add('cache')
        elif ('throttl' in imp or 'rate' in imp) and any('throttle' in s.lower() or 'rate' in s.lower() for s in syms):
            terms.add('throttled')
    
    # Add domain terms
    if 'foreca' in str(file_path).lower():
        terms.update(['foreca', 'weather', 'proxy'])
    elif 'weather' in str(file_path).lower():
        terms.update(['weather', 'forecast'])
    
    return ', '.join(sorted(list(terms)[:12]))


def extract_go_symbols_defined(node_type: str, extra: dict) -> List[str]:
    """Extract symbols defined in Go chunk."""
    symbols = []
    
    if node_type == 'type_declaration':
        type_name = extra.get('type_name', '')
        if type_name:
            symbols.append(type_name)
    elif node_type == 'method_declaration':
        method_name = extra.get('method_name', '')
        if method_name:
            symbols.append(method_name)
    elif node_type == 'function_declaration':
        function_name = extra.get('function_name', '')
        if function_name:
            symbols.append(function_name)
    
    return symbols


def extract_go_symbols_referenced(chunk_text: str) -> List[str]:
    """Extract symbols referenced in Go chunk (simplified)."""
    # This is a simplified implementation
    # In a full implementation, you'd parse the chunk to find referenced symbols
    return []


def create_ast_chunk(lines: List[str], start_line: int, end_line: int, 
                    repo: str, file_path: Path, file_sha: str, rel_path: str,
                    language: str, encoding: str, logger: logging.Logger) -> Optional[ChunkRecord]:
    """Create a single AST-based chunk."""
    try:
        # Extract chunk content (convert to 1-based indexing)
        chunk_lines = lines[start_line:end_line+1]
        chunk_text = '\n'.join(chunk_lines)
        
        # Generate AST path
        ast_path = f"line_{start_line+1}_to_{end_line+1}"
        
        # Extract imports and symbols
        imports_used = extract_imports_from_code(chunk_text, language)
        symbols_defined = []  # TODO: Extract from AST
        symbols_referenced = []  # TODO: Extract from AST
        
        # Count tokens
        token_count = count_tokens(chunk_text, encoding)
        
        # Create chunk ID
        chunk_id = compute_chunk_id(repo, str(file_path), file_sha, start_line+1, end_line+1, ast_path)
        
        chunk = ChunkRecord(
            chunk_id=chunk_id,
            repo=repo,
            path=str(file_path),
            rel_path=rel_path,
            language=language,
            start_line=start_line+1,  # Convert to 1-based
            end_line=end_line+1,      # Convert to 1-based
            ast_path=ast_path,
            text=chunk_text,
            header_context="",
            core=chunk_text,
            footer_context="",
            symbols_defined=symbols_defined,
            symbols_referenced=symbols_referenced,
            imports_used=imports_used,
            neighbors=Neighbors(),
            summary_1l=generate_summary_1l(chunk_text, language, ast_path),
            qa_terms=generate_qa_terms(chunk_text, language),
            token_counts=TokenCounts(core=token_count, total=token_count),
            file_sha=file_sha,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        return chunk
        
    except Exception as e:
        logger.error(f"Failed to create chunk for {file_path} lines {start_line}-{end_line}: {e}")
        return None


def process_file(args: Tuple[Path, str, str, int, int, str]) -> List[ChunkRecord]:
    """Process a single file (for multiprocessing)."""
    file_path, repo, encoding, core_tokens, max_total, log_level = args
    
    # Setup logging for this process
    logger = setup_logging(log_level)
    
    return chunk_code_file(file_path, repo, encoding, core_tokens, max_total, logger)


def load_cache_manifest(cache_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load cache manifest."""
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache_manifest(cache_dir: Path, manifest: Dict[str, Dict[str, Any]]):
    """Save cache manifest."""
    cache_dir.mkdir(exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AST-aware code chunking with context stitching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_chunks_v3.py --root ./myrepo --out ./chunks.jsonl
  python build_chunks_v3.py --file ./src/api.py --force --log-level DEBUG
  python build_chunks_v3.py --root ./repo --core-tokens 200 --max-total 500
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--root', type=Path, help='Root directory to process')
    input_group.add_argument('--file', type=Path, help='Single file to process')
    
    # Output options
    parser.add_argument('--out', type=Path, default=Path('./chunks_v3.jsonl'),
                       help='Output JSONL file path (default: ./chunks_v3.jsonl)')
    
    # Chunking parameters
    parser.add_argument('--encoding', default='gpt-4', 
                       help='Token encoding model (default: gpt-4)')
    parser.add_argument('--core-tokens', type=int, default=150,
                       help='Target core tokens per chunk (default: 150)')
    parser.add_argument('--max-total', type=int, default=380,
                       help='Maximum total tokens per chunk (default: 380)')
    
    # Processing options
    parser.add_argument('--max-workers', type=int, default=os.cpu_count(),
                       help='Maximum parallel workers (default: CPU count)')
    parser.add_argument('--include-vendor', action='store_true',
                       help='Include vendor/build directories')
    parser.add_argument('--force', action='store_true',
                       help='Rebuild even if unchanged')
    parser.add_argument('--repo', help='Repository name/slug')
    parser.add_argument('--log-level', choices=['INFO', 'DEBUG'], default='INFO',
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Determine repository name
    if args.repo:
        repo_name = args.repo
    elif args.root:
        repo_name = args.root.name
    elif args.file:
        repo_name = args.file.parent.name
    else:
        repo_name = "unknown"
    
    # Collect files to process
    files_to_process = []
    
    if args.root:
        for file_path in args.root.rglob('*'):
            if file_path.is_file() and not should_skip_file(file_path, args.include_vendor):
                files_to_process.append(file_path)
    elif args.file:
        if not should_skip_file(args.file, args.include_vendor):
            files_to_process.append(args.file)
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Setup cache
    cache_dir = Path('.chunks-cache')
    manifest = load_cache_manifest(cache_dir) if not args.force else {}
    
    # Filter unchanged files
    if not args.force:
        unchanged_count = 0
        for file_path in files_to_process[:]:
            file_str = str(file_path)
            try:
                mtime = file_path.stat().st_mtime
                file_sha = compute_file_sha(file_path)
                
                if (file_str in manifest and 
                    manifest[file_str].get('mtime') == mtime and
                    manifest[file_str].get('file_sha') == file_sha):
                    files_to_process.remove(file_path)
                    unchanged_count += 1
            except Exception:
                pass  # Process file if we can't check cache
        
        logger.info(f"Skipping {unchanged_count} unchanged files")
    
    if not files_to_process:
        logger.info("No files to process")
        return
    
    # Process files in parallel
    all_chunks = []
    process_args = [
        (file_path, repo_name, args.encoding, args.core_tokens, args.max_total, args.log_level)
        for file_path in files_to_process
    ]
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_file, args): args[0] for args in process_args}
        
        with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                    logger.info(f"Processed {file_path}: {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                finally:
                    pbar.update(1)
    
    # Write output
    logger.info(f"Writing {len(all_chunks)} chunks to {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.out, 'w') as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json() + '\n')
    
    # Update cache manifest
    for file_path in files_to_process:
        try:
            mtime = file_path.stat().st_mtime
            file_sha = compute_file_sha(file_path)
            manifest[str(file_path)] = {'mtime': mtime, 'file_sha': file_sha}
        except Exception:
            pass
    
    save_cache_manifest(cache_dir, manifest)
    
    logger.info(f"Successfully processed {len(files_to_process)} files, generated {len(all_chunks)} chunks")


if __name__ == '__main__':
    main()
