#!/usr/bin/env python3
"""
Build enhanced chunks (v3) with structural metadata and optional LLM enrichment.

Extends v2 with additional metadata:
- Structural: ast_path, package, node_kind, receiver, function_name, method_name, type_name, type_kind
- Context: header_context_minimal, imports_used_minimal, symbols_referenced_strict
- Quality: is_generated, rank_boost
- LLM: summary_llm, keywords_llm (optional)

Usage:
  python nwe_chunks_v3.py /path/to/repo \
    --out run/<repo>_chunks_v3.jsonl \
    --token-limit 1200 \
    --encoding gpt-4 \
    --add-structure \
    --add-minimal-context \
    --downweight-generated \
    --llm-enrich \
    --llm-model qwen2.5-coder:1.5b-instruct  # Optional: use faster 1.5B model (4x faster)

Notes:
- Uses your existing CodeChunker with enhanced Go-specific parsing
- New fields are additive - all v2 fields remain unchanged
- LLM enrichment requires local Ollama with qwen2.5-coder model
- For 4x faster enrichment, use --llm-model qwen2.5-coder:1.5b-instruct
"""

import os
import re
import sys
import json
import argparse
import hashlib
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Make src/ importable when running from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from Chunker import CodeChunker  # your existing
from utils import count_tokens   # your existing

# Import v3 enrichment modules
from nwe_v3_enrich.adapter import (
    FileContext,
    parse_file_context,
    is_generated_file,
    normalize_rel_path,
    compute_minimal_imports,
    compute_symbols_referenced,
    build_header_context_minimal,
    infer_go_structure,
)
from nwe_v3_enrich.llm_qwen import (
    summarize_chunk_qwen,
    digest_for_cache,
    LLMCache,
)

SUPPORTED_EXTS = {"py", "js", "jsx", "ts", "tsx", "go", "rb", "php", "java"}
DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "__pycache__", ".mypy_cache",
    "node_modules", "dist", "build", "out", "target", ".venv", "venv", ".tox",
}

# ---------- I/O helpers ----------

def iter_source_files(root_dir: str, allowed_exts, exclude_dirs):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".cache")]
        for fname in filenames:
            if "." not in fname:
                continue
            ext = fname.rsplit(".", 1)[-1].lower()
            if ext in allowed_exts:
                full = os.path.join(dirpath, fname)
                yield full, ext, os.path.relpath(full, root_dir)

def read_text_safely(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

# ---------- lightweight symbol extraction (regex-first, language-aware) ----------

# Simple patterns by language. This is intentionally minimal to avoid breaking anything.
PY_DEF = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
PY_CLASS = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:(]", re.MULTILINE)
TS_FN = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
TS_FN_EXPR = re.compile(r"^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s+)?\(", re.MULTILINE)
TS_CLASS = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.MULTILINE)
JS_FN = TS_FN
JS_FN_EXPR = TS_FN_EXPR
JS_CLASS = TS_CLASS
GO_FN = re.compile(r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
JAVA_CLASS = re.compile(r"^\s*(?:public\s+|protected\s+|private\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.MULTILINE)
JAVA_METHOD = re.compile(r"^\s*(?:public|protected|private|static|\s)+\s+[A-Za-z_<>\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
RB_DEF = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_!?]*)\b", re.MULTILINE)
PHP_FN = re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)

IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

def lang_from_ext(ext: str) -> str:
    return {
        "py": "python", "ts": "typescript", "tsx": "typescript",
        "js": "javascript", "jsx": "javascript",
        "go": "go", "rb": "ruby", "php": "php", "java": "java"
    }.get(ext, ext)

def dedupe_keep_order(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def take_doc_head(code: str, lang: str, max_chars: int = 1200) -> str:
    # crude: for Python, triple-quoted string near top; for others, leading block comments.
    head = ""
    if lang == "python":
        m = re.search(r'^\s*("""|\'\'\')(.*?)\1', code, flags=re.DOTALL|re.MULTILINE)
        if m:
            head = m.group(2)
    else:
        m = re.search(r"^\s*/\*\*(.*?)\*/", code, flags=re.DOTALL|re.MULTILINE) or \
            re.search(r"^\s*/\*(.*?)\*/", code, flags=re.DOTALL|re_MULTILINE) if False else None
        if m:
            head = m.group(1)
    if not head:
        # fallback: first 50 non-empty lines
        lines = [ln for ln in code.splitlines() if ln.strip()]
        head = "\n".join(lines[:50])
    
    # Smart truncation at line/word boundary to avoid dangling tokens
    if len(head) > max_chars:
        # Try to truncate at line boundary
        lines = head[:max_chars].split('\n')
        if len(lines) > 1:
            # Keep all complete lines
            head = '\n'.join(lines[:-1])
        else:
            # Truncate at last word boundary
            truncated = head[:max_chars]
            last_space = truncated.rfind(' ')
            last_newline = truncated.rfind('\n')
            cutoff = max(last_space, last_newline)
            if cutoff > max_chars * 0.8:  # Only if we don't lose too much
                head = truncated[:cutoff] + "..."
            else:
                head = truncated + "..."
    
    return head

def extract_defs(code: str, ext: str):
    """Return (primary_symbol, primary_kind, def_symbols:list)."""
    if ext == "py":
        classes = PY_CLASS.findall(code)
        funcs = PY_DEF.findall(code)
        if classes:
            return classes[0], "class", dedupe_keep_order(classes + funcs)
        if funcs:
            return funcs[0], "function", dedupe_keep_order(funcs)
    elif ext in ("ts", "tsx"):
        classes = TS_CLASS.findall(code)
        fns = TS_FN.findall(code) + TS_FN_EXPR.findall(code)
        if classes:
            return classes[0], "class", dedupe_keep_order(classes + fns)
        if fns:
            return fns[0], "function", dedupe_keep_order(fns)
    elif ext in ("js", "jsx"):
        classes = JS_CLASS.findall(code)
        fns = JS_FN.findall(code) + JS_FN_EXPR.findall(code)
        if classes:
            return classes[0], "class", dedupe_keep_order(classes + fns)
        if fns:
            return fns[0], "function", dedupe_keep_order(fns)
    elif ext == "go":
        fns = GO_FN.findall(code)
        if fns:
            return fns[0], "function", dedupe_keep_order(fns)
    elif ext == "java":
        classes = JAVA_CLASS.findall(code)
        meths = JAVA_METHOD.findall(code)
        if classes:
            return classes[0], "class", dedupe_keep_order(classes + meths)
        if meths:
            return meths[0], "method", dedupe_keep_order(meths)
    elif ext == "rb":
        defs = RB_DEF.findall(code)
        if defs:
            return defs[0], "function", dedupe_keep_order(defs)
    elif ext == "php":
        fns = PHP_FN.findall(code)
        if fns:
            return fns[0], "function", dedupe_keep_order(fns)
    return "", "", []

def guess_primary_span(chunk_start_line: int, chunk_end_line: int, code: str, primary_symbol: str):
    """
    Heuristic: if we can locate the line where the primary symbol is defined in this chunk,
    set span to [that_line, chunk_end_line]; otherwise fall back to the chunk bounds.
    """
    if not primary_symbol:
        return [chunk_start_line, chunk_end_line]
    # find first line containing the symbol and a def-ish token
    lines = code.splitlines()
    candidates = []
    for i, ln in enumerate(lines, start=0):
        if primary_symbol in ln and any(tok in ln for tok in ("def ", "class ", "function ", "func ", "export ")):
            candidates.append(i)
            break
    if candidates:
        start = chunk_start_line + candidates[0]
        return [start, chunk_end_line]
    return [chunk_start_line, chunk_end_line]

def extract_symbols_minimal(code: str, ext: str, chunk_start_line: int, chunk_end_line: int):
    """
    Minimal, fast symbol extraction without tree-sitter.
    You can swap this with a tree-sitter-backed version later.
    """
    primary_symbol, primary_kind, defs = extract_defs(code, ext)
    # symbols: all identifiers (cap to keep tight)
    idents = IDENT.findall(code)
    # basic de-noise: drop super-short identifiers
    idents = [s for s in idents if len(s) >= 3]
    # merge defs + refs, unique and keep order
    symbols = dedupe_keep_order(defs + idents)
    # doc_head near top (or top doc comment)
    lang = lang_from_ext(ext)
    doc_head = take_doc_head(code, lang)
    # span heuristic
    primary_span = guess_primary_span(chunk_start_line, chunk_end_line, code, primary_symbol)
    return {
        "primary_symbol": primary_symbol or "",
        "primary_kind": primary_kind or "",
        "primary_span": primary_span,
        "def_symbols": defs[:64],
        "symbols": symbols[:256],
        "doc_head": doc_head
        # parent_symbol, exports omitted in minimal version
    }

# ---------- main chunking ----------

def build_chunks_v3(root_dir: str, token_limit: int, encoding_name: str, output_jsonl: str, 
                   include_exts=None, exclude_dirs=None, 
                   add_structure: bool = True, add_minimal_context: bool = True,
                   downweight_generated: bool = True, llm_enrich: bool = False,
                   llm_model: str = "qwen2.5-coder:7b-instruct", llm_concurrency: int = 4,
                   prompt_version: str = "v1", repo_root: Optional[Path] = None,
                   text_chunker: str = None):
    include_exts = set(include_exts or SUPPORTED_EXTS)
    exclude_dirs = set(exclude_dirs or DEFAULT_EXCLUDES)
    repo_id = os.path.basename(os.path.abspath(root_dir))

    chunkers = {}
    total_files = 0
    total_chunks = 0
    
    # Determine text chunker to use
    # Priority: CLI arg > env var > default (v4)
    if text_chunker is None:
        text_chunker = os.environ.get("V4_TEXT_CHUNKER", "v4").lower()
    use_v2_text = (text_chunker == "v2")
    
    if use_v2_text:
        print(f"📝 Using v2 text chunker (CodeChunker with function boundaries)")
        # Import v2 chunker functions
        sys.path.append(os.path.join(os.path.dirname(__file__), "chunking_v4"))
        from chunking_v4.text_chunkers import chunk_code_text_v2, map_v2_chunk_to_v4_span
    else:
        print(f"📝 Using v4 text chunker (current/default)")
    
    # Initialize LLM cache if needed
    llm_cache = None
    if llm_enrich:
        llm_cache = LLMCache()

    # Debug logging
    debug_chunk_log = os.environ.get("DEBUG_CHUNK_LOG") == "1"

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as out:
        for full_path, ext, rel_path in iter_source_files(root_dir, include_exts, exclude_dirs):
            total_files += 1
            if ext not in chunkers:
                chunkers[ext] = CodeChunker(file_extension=ext, encoding_name=encoding_name)

            code = read_text_safely(full_path)
            if not code.strip():
                continue

            try:
                chunks = chunkers[ext].chunk(code, token_limit=token_limit)
            except Exception as e:
                print(f"[WARN] Skipping {full_path} ({ext}): {e}")
                continue

            # If using v2 text chunker, compute v2 windows for text replacement
            v2_text_chunks = None
            source_lines = None
            if use_v2_text:
                try:
                    v2_text_chunks = chunk_code_text_v2(
                        source_text=code,
                        file_extension=ext,
                        token_limit=token_limit,
                        encoding_name=encoding_name
                    )
                    source_lines = code.split("\n")
                except Exception as e:
                    print(f"[WARN] v2 text chunker failed for {full_path}, using v4 text: {e}")
                    v2_text_chunks = None

            # Parse file context for Go files
            file_ctx = None
            if ext == "go":
                file_ctx = parse_file_context(code)
                # Add abs_path and file_text for Tree-sitter support
                file_ctx.abs_path = str(full_path)
                file_ctx.file_text = code

            # Normalize paths
            rel_path_normalized = normalize_rel_path(rel_path, repo_root)
            
            # Generate file synopsis once per file (for LLM enrichment)
            file_synopsis = ""
            file_synopsis_hash = ""
            if llm_enrich:
                from src.nwe_v3_enrich.adapter import generate_file_synopsis
                file_synopsis = generate_file_synopsis(code, rel_path)
                file_synopsis_hash = hashlib.sha256(file_synopsis.encode("utf-8")).hexdigest()[:16]

            # Process chunks and collect for neighbor chaining
            file_chunks = []
            for cidx in sorted(chunks):
                chunk = chunks[cidx]
                if isinstance(chunk, dict):
                    text = chunk.get("text", "")
                    start_line = chunk.get("start_line")
                    end_line = chunk.get("end_line")
                else:
                    text = str(chunk)
                    start_line = None
                    end_line = None

                if not text.strip():
                    continue

                # Replace text with v2 chunker output if enabled
                original_text = text
                if use_v2_text and v2_text_chunks and source_lines and start_line and end_line:
                    try:
                        v2_text = map_v2_chunk_to_v4_span(
                            v2_chunks=v2_text_chunks,
                            v4_start_line=start_line,
                            v4_end_line=end_line,
                            source_lines=source_lines
                        )
                        if v2_text and v2_text.strip():
                            text = v2_text
                        else:
                            # Fallback to original if mapping produces empty text
                            if debug_chunk_log:
                                print(f"[WARN] v2 mapping produced empty text for {full_path}:{start_line}-{end_line}, using original")
                    except Exception as e:
                        if debug_chunk_log:
                            print(f"[WARN] v2 text mapping failed for {full_path}:{start_line}-{end_line}: {e}, using original")

                # symbol-aware metadata (v2 fields)
                meta = extract_symbols_minimal(text, ext, start_line or 1, end_line or (start_line or 1))

                n_tokens = count_tokens(text, encoding_name)
                chunk_id = f"{full_path}#{cidx}"
                record = {
                    "id": chunk_id,
                    "repo_id": repo_id,
                    "path": rel_path_normalized,  # Use repo-relative path
                    "abs_path": full_path,  # Keep absolute path
                    "rel_path": rel_path,
                    "ext": ext,
                    "language": lang_from_ext(ext),
                    "chunk_number": cidx,
                    "start_line": start_line,
                    "end_line": end_line,
                    "text": text,
                    "n_tokens": n_tokens,

                    # v2 fields (unchanged)
                    "primary_symbol": meta["primary_symbol"],
                    "primary_kind": meta["primary_kind"],
                    "primary_span": meta["primary_span"],
                    "def_symbols": meta["def_symbols"],
                    "symbols": meta["symbols"],
                    "doc_head": meta["doc_head"],
                }
                
                # v3 enrichment
                is_gen = is_generated_file(rel_path_normalized, text)
                
                # Add quality signals
                record["is_generated"] = is_gen
                record["rank_boost"] = 0.35 if is_gen and downweight_generated else 1.0
                
                # Add structural metadata and context for Go files
                if ext == "go" and (add_structure or add_minimal_context):
                    # Calculate byte offsets for Tree-sitter
                    start_byte = None
                    end_byte = None
                    if file_ctx and file_ctx.file_text:
                        # Find chunk position in file
                        text_bytes = text.encode('utf-8')
                        file_bytes = file_ctx.file_text.encode('utf-8')
                        start_byte = file_bytes.find(text_bytes)
                        if start_byte != -1:
                            end_byte = start_byte + len(text_bytes)
                    
                    # Pass chunks for neighbor analysis and byte offsets for Tree-sitter
                    struct = infer_go_structure(text, file_ctx, file_chunks, len(file_chunks), start_byte, end_byte)
                    
                    # Override old regex-based fields with Tree-sitter data
                    if struct.get("node_kind"):
                        record["primary_kind"] = struct["node_kind"]
                    if struct.get("primary_symbol"):
                        record["primary_symbol"] = struct["primary_symbol"]
                    # PHASE 2c: Use Tree-sitter def_symbols directly (excludes primary)
                    if "def_symbols" in struct:
                        # Tree-sitter def_symbols = secondary symbols only (may be empty list)
                        record["def_symbols"] = struct["def_symbols"]
                    
                    if add_structure:
                        record.update({
                            "ast_path": struct["ast_path"],
                            "package": struct["package"],
                            "node_kind": struct["node_kind"],
                            "receiver": struct["receiver"],
                            "function_name": struct["function_name"],
                            "method_name": struct["method_name"],
                            "type_name": struct["type_name"],
                            "type_kind": struct["type_kind"],
                            "primary_symbol": struct["primary_symbol"],
                            # Multi-declaration fields (v3 enhancement)
                            "is_multi_declaration": struct.get("is_multi_declaration", False),
                            "all_symbols": struct.get("all_symbols", []),
                            "all_kinds": struct.get("all_kinds", []),
                            "all_ast_paths": struct.get("all_ast_paths", []),
                            "all_roles": struct.get("all_roles", []),
                            "all_receivers": struct.get("all_receivers", []),
                            "all_type_names": struct.get("all_type_names", []),
                            "all_type_kinds": struct.get("all_type_kinds", []),
                            # Phase 1 fixes: byte ranges and primary_index
                            "all_start_bytes": struct.get("all_start_bytes", []),
                            "all_end_bytes": struct.get("all_end_bytes", []),
                            "primary_index": struct.get("primary_index", 0),
                            # Phase 2b: Normalized receivers
                            "all_receivers_normalized": struct.get("all_receivers_normalized", []),
                        })
                    
                    if add_minimal_context:
                        imports_min = compute_minimal_imports(text, file_ctx)
                        symbols_strict = compute_symbols_referenced(text)
                        hdr_min = build_header_context_minimal(
                            struct["package"], imports_min, struct["node_kind"], struct.get("receiver", ""), file_ctx
                        )
                        
                        record.update({
                            "imports_used_minimal": imports_min,
                            "symbols_referenced_strict": symbols_strict,
                            "header_context_minimal": hdr_min,
                        })
                
                # LLM enrichment (if enabled)
                if llm_enrich and llm_cache:
                    cache_key = digest_for_cache(record, prompt_version)
                    cached_result = llm_cache.get(cache_key)
                    
                    if cached_result:
                        record.update(cached_result)
                        # Add provenance for cached results using pre-computed file synopsis hash
                        file_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                        record["enrich_provenance"] = {
                            "model": llm_model,
                            "created_at": datetime.now().isoformat(),
                            "file_synopsis_hash": file_synopsis_hash,
                            "chunk_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                            "input_lang": "en",
                            "language_policy": {"language": "en", "skip_non_en": True},
                            "skipped_reason": None
                        }
                    else:
                        try:
                            # Use pre-computed file synopsis
                            llm_result = summarize_chunk_qwen(record, llm_model, prompt_version, file_synopsis=file_synopsis)
                            record.update(llm_result)
                            
                            # Add provenance using pre-computed file synopsis hash
                            file_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                            record["enrich_provenance"] = {
                                "model": llm_model,
                                "created_at": datetime.now().isoformat(),
                                "file_synopsis_hash": file_synopsis_hash,
                                "chunk_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                                "input_lang": "en",
                                "language_policy": {"language": "en", "skip_non_en": True},
                                "skipped_reason": None
                            }
                            
                            # Cache the result
                            llm_cache.set(cache_key, record["id"], file_sha, prompt_version, 
                                        record.get("ast_path", ""), llm_result)
                        except Exception as e:
                            print(f"Warning: LLM enrichment failed for {rel_path}: {e}")
                            record.update({
                                "summary_en": "LLM enrichment failed",
                                "keywords_en": ["error", "llm", "failed", "enrichment", "chunk", "code", "analysis", "unavailable"],
                                "enrich_provenance": {
                                    "model": llm_model,
                                    "created_at": datetime.now().isoformat(),
                                    "file_synopsis_hash": "",
                                    "chunk_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                                    "input_lang": "en",
                                    "language_policy": {"language": "en", "skip_non_en": True},
                                    "skipped_reason": "llm_error"
                                }
                            })

                file_chunks.append(record)
            
            # Add neighbor chaining
            for i, record in enumerate(file_chunks):
                neighbors = {}
                if i > 0:
                    neighbors["prev"] = file_chunks[i-1]["id"]
                if i < len(file_chunks) - 1:
                    neighbors["next"] = file_chunks[i+1]["id"]
                record["neighbors"] = neighbors
                
                # Debug logging
                if debug_chunk_log and ext == "go":
                    print(f"DEBUG: {record['rel_path']} chunk {record['chunk_number']} "
                          f"lines {record['start_line']}-{record['end_line']} "
                          f"kind={record.get('node_kind', 'N/A')} "
                          f"ast_path={record.get('ast_path', 'N/A')} "
                          f"package={record.get('package', 'N/A')} "
                          f"imports={len(record.get('imports_used_minimal', []))}")
                
                # Write chunk
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Finished.")
    print(f"Repo ID:        {repo_id}")
    print(f"Scanned files:  {total_files}")
    print(f"Wrote chunks:   {total_chunks}")
    print(f"Output JSONL:   {os.path.abspath(output_jsonl)}")

def main():
    ap = argparse.ArgumentParser(description="Build enhanced chunks (v3) with structural metadata and optional LLM enrichment.")
    ap.add_argument("directory", help="Root directory of the project to chunk")
    ap.add_argument("--token-limit", type=int, default=200, help="Max tokens per chunk")
    ap.add_argument("--encoding", default="gpt-4", help="Tokenizer name used by utils.count_tokens")
    ap.add_argument("--out", default="run/chunks_v3.jsonl", help="Output JSONL file")
    ap.add_argument("--include-exts", default=",".join(sorted(SUPPORTED_EXTS)),
                    help="Comma-separated extensions (no dots)")
    ap.add_argument("--exclude-dirs", default=",".join(sorted(DEFAULT_EXCLUDES)),
                    help="Comma-separated directory names to exclude")
    
    # v3 enrichment flags
    ap.add_argument("--add-structure", action="store_true", default=True,
                    help="Add structural metadata (ast_path, package, node_kind, etc.)")
    ap.add_argument("--add-minimal-context", action="store_true", default=True,
                    help="Add minimal context (imports_used_minimal, symbols_referenced_strict, header_context_minimal)")
    ap.add_argument("--downweight-generated", action="store_true", default=True,
                    help="Downweight generated code chunks (rank_boost=0.35)")
    ap.add_argument("--llm-enrich", action="store_true", default=False,
                    help="Enable LLM enrichment with local Ollama")
    ap.add_argument("--llm-model", default="qwen2.5-coder:7b-instruct",
                    help="Ollama model for LLM enrichment. Options: 'qwen2.5-coder:7b-instruct' (default, best quality) or 'qwen2.5-coder:1.5b-instruct' (4x faster, good quality)")
    ap.add_argument("--llm-concurrency", type=int, default=4,
                    help="Number of concurrent LLM requests")
    ap.add_argument("--prompt-version", default="v1",
                    help="Version of the LLM prompt")
    ap.add_argument("--repo-root", type=Path,
                    help="Repository root for path normalization")
    ap.add_argument("--text-chunker", choices=["v4", "v2"], default=None,
                    help="Text chunking strategy: v4 (current) or v2 (CodeChunker with function boundaries). Defaults to env var V4_TEXT_CHUNKER or 'v4'")
    
    args = ap.parse_args()

    include_exts = {e.strip().lower() for e in args.include_exts.split(",") if e.strip()}
    exclude_dirs = {d.strip() for d in args.exclude_dirs.split(",") if d.strip()}

    build_chunks_v3(
        root_dir=args.directory,
        token_limit=args.token_limit,
        encoding_name=args.encoding,
        output_jsonl=args.out,
        include_exts=include_exts,
        exclude_dirs=exclude_dirs,
        add_structure=args.add_structure,
        add_minimal_context=args.add_minimal_context,
        downweight_generated=args.downweight_generated,
        llm_enrich=args.llm_enrich,
        llm_model=args.llm_model,
        llm_concurrency=args.llm_concurrency,
        prompt_version=args.prompt_version,
        repo_root=args.repo_root,
        text_chunker=args.text_chunker,
    )

if __name__ == "__main__":
    main()