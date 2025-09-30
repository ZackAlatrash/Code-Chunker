#!/usr/bin/env python3
"""
Build symbol-aware chunks (v2) for code_chunks_v3.

Outputs JSONL with lean metadata:
- primary_symbol (str)
- primary_kind   (keyword-like str)
- primary_span   ([start_line, end_line])  -- best-effort
- def_symbols    (list[str])
- symbols        (list[str])               -- defs + refs (unique)
- doc_head       (short text near the primary definition)
- parent_symbol  (optional)
- exports        (optional)
Plus your existing fields:
- repo_id, path, rel_path, ext/language, chunk_number, start_line, end_line, text, n_tokens

Usage:
  python scripts/build_chunks_v2.py /path/to/repo \
    --out run/<repo>_chunks_v3.jsonl \
    --token-limit 1200 \
    --encoding gpt-4

Notes:
- Uses your existing CodeChunker. We add light regex-based symbol extraction
  so you can test immediately. If you want tree-sitter–exact spans later,
  you can swap impls in `extract_symbols_minimal()`.
"""

import os
import re
import sys
import json
import argparse
from datetime import datetime

# Make src/ importable when running from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from Chunker import CodeChunker  # your existing
from utils import count_tokens   # your existing

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
    return head[:max_chars]

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

def build_chunks_v2(root_dir: str, token_limit: int, encoding_name: str, output_jsonl: str, include_exts=None, exclude_dirs=None):
    include_exts = set(include_exts or SUPPORTED_EXTS)
    exclude_dirs = set(exclude_dirs or DEFAULT_EXCLUDES)
    repo_id = os.path.basename(os.path.abspath(root_dir))

    chunkers = {}
    total_files = 0
    total_chunks = 0

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

                # symbol-aware metadata
                meta = extract_symbols_minimal(text, ext, start_line or 1, end_line or (start_line or 1))

                n_tokens = count_tokens(text, encoding_name)
                record = {
                    "id": f"{full_path}#{cidx}",
                    "repo_id": repo_id,
                    "path": full_path,
                    "rel_path": rel_path,
                    "ext": ext,
                    "language": lang_from_ext(ext),
                    "chunk_number": cidx,
                    "start_line": start_line,
                    "end_line": end_line,
                    "text": text,
                    "n_tokens": n_tokens,

                    # NEW fields (lean set)
                    "primary_symbol": meta["primary_symbol"],
                    "primary_kind": meta["primary_kind"],
                    "primary_span": meta["primary_span"],
                    "def_symbols": meta["def_symbols"],
                    "symbols": meta["symbols"],
                    "doc_head": meta["doc_head"],
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Finished.")
    print(f"Repo ID:        {repo_id}")
    print(f"Scanned files:  {total_files}")
    print(f"Wrote chunks:   {total_chunks}")
    print(f"Output JSONL:   {os.path.abspath(output_jsonl)}")

def main():
    ap = argparse.ArgumentParser(description="Build symbol-aware chunks (v2) for code_chunks_v3.")
    ap.add_argument("directory", help="Root directory of the project to chunk")
    ap.add_argument("--token-limit", type=int, default=200, help="Max tokens per chunk")
    ap.add_argument("--encoding", default="gpt-4", help="Tokenizer name used by utils.count_tokens")
    ap.add_argument("--out", default="run/chunks_v3.jsonl", help="Output JSONL file")
    ap.add_argument("--include-exts", default=",".join(sorted(SUPPORTED_EXTS)),
                    help="Comma-separated extensions (no dots)")
    ap.add_argument("--exclude-dirs", default=",".join(sorted(DEFAULT_EXCLUDES)),
                    help="Comma-separated directory names to exclude")
    args = ap.parse_args()

    include_exts = {e.strip().lower() for e in args.include_exts.split(",") if e.strip()}
    exclude_dirs = {d.strip() for d in args.exclude_dirs.split(",") if d.strip()}

    build_chunks_v2(
        root_dir=args.directory,
        token_limit=args.token_limit,
        encoding_name=args.encoding,
        output_jsonl=args.out,
        include_exts=include_exts,
        exclude_dirs=exclude_dirs,
    )

if __name__ == "__main__":
    main()