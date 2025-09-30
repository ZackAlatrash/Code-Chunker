# chunk_project.py
import os
import re
import json
import argparse
import hashlib
import sys
from datetime import datetime
from typing import List, Optional

# Make src/ importable when running from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from Chunker import CodeChunker, Chunker
from utils import count_tokens

# Extensions your CodeParser/CodeChunker supports
SUPPORTED_EXTS = {"py", "js", "jsx", "css", "ts", "tsx", "php", "rb", "go"}

# Common directories to skip
DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "__pycache__", ".mypy_cache",
    "node_modules", "dist", "build", "out", "target", ".venv", "venv", ".tox",
    ".cache"
}

LANG_BY_EXT = {
    "py": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "php": "php",
    "rb": "ruby",
    "go": "go",
    "css": "css",
}

# --- heuristics & helpers -----------------------------------------------------

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "repo"

def derive_repo_id(root_dir: str) -> str:
    """
    Stable, human-friendly id from repo folder name.
    Example: /Users/me/projects/weather-api -> weather-api
    """
    base = os.path.basename(os.path.abspath(root_dir))
    return slugify(base)

def detect_language(ext: str) -> str:
    return LANG_BY_EXT.get(ext.lower(), "unknown")

def project_rel_path(root_dir: str, abs_path: str) -> str:
    try:
        return os.path.relpath(abs_path, start=root_dir)
    except Exception:
        return abs_path

def guess_module(rel_path: str, language: str) -> str:
    """
    Best-effort module/package hint for boosting and display.
    - For Python: convert path like src/foo/bar/baz.py -> src.foo.bar.baz
    - For JS/TS/Go/Ruby/PHP: use slash->dot convention (no guarantee)
    """
    p = rel_path.replace("\\", "/")
    p = re.sub(r"\.(py|js|jsx|ts|tsx|php|rb|go|css)$", "", p, flags=re.I)
    parts = [seg for seg in p.split("/") if seg not in ("", ".")]
    return ".".join(parts)

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

# --- lightweight symbol extraction (fallback if you don't have tree-sitter here) ----

PY_FUNC_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.M)
PY_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*[\(:]", re.M)

JS_FUNC_RE = re.compile(r"(?:function\s+([A-Za-z_]\w*)\s*\(|const\s+([A-Za-z_]\w*)\s*=\s*\(|async\s+function\s+([A-Za-z_]\w*)\s*\()", re.M)
JS_CLASS_RE = re.compile(r"class\s+([A-Za-z_]\w*)\s*", re.M)

PHP_FUNC_RE = re.compile(r"function\s+([A-Za-z_]\w*)\s*\(", re.M)
PHP_CLASS_RE = re.compile(r"class\s+([A-Za-z_]\w*)\s*", re.M)

RB_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*[!?=]?)", re.M)
RB_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M)

GO_FUNC_RE = re.compile(r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*)\s*\(", re.M)
GO_TYPE_RE = re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+struct", re.M)

IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b")

def extract_symbols_from_text(text: str, language: str, max_symbols: int = 50) -> List[str]:
    syms: List[str] = []
    try:
        if language == "python":
            syms += PY_FUNC_RE.findall(text)
            syms += PY_CLASS_RE.findall(text)
        elif language in ("javascript", "typescript"):
            for m in JS_FUNC_RE.findall(text):
                syms += [g for g in m if g]
            syms += JS_CLASS_RE.findall(text)
        elif language == "php":
            syms += PHP_FUNC_RE.findall(text)
            syms += PHP_CLASS_RE.findall(text)
        elif language == "ruby":
            syms += RB_DEF_RE.findall(text)
            syms += RB_CLASS_RE.findall(text)
        elif language == "go":
            syms += GO_FUNC_RE.findall(text)
            syms += GO_TYPE_RE.findall(text)
        else:
            # fallback: mine identifiers
            syms += IDENT_RE.findall(text)
    except Exception:
        pass

    # also mine some high-signal identifiers from the body (camel/snake split)
    extra = []
    for token in set(IDENT_RE.findall(text)):
        # split simple camelCase / snake_case
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", token)
        parts += token.split("_")
        parts = [p.lower() for p in parts if p and len(p) > 2]
        extra.extend(parts[:2])  # a couple per token

    all_syms = []
    seen = set()
    for s in syms + extra:
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        all_syms.append(s)
        if len(all_syms) >= max_symbols:
            break
    return all_syms

def guess_ast_kind(text: str, language: str) -> str:
    """
    Very rough label used for boosting/analytics (not critical).
    """
    t = text
    if language == "python":
        if "class " in t: return "class_or_type"
        if re.search(r"^\s*def\s+", t, re.M): return "function_or_method"
    if language in ("javascript","typescript"):
        if "class " in t: return "class_or_type"
        if re.search(r"\bfunction\b|\s*=>\s*\{", t): return "function_or_method"
    if language == "go":
        if "type " in t and "struct" in t: return "class_or_type"
        if re.search(r"^\s*func\s+", t, re.M): return "function_or_method"
    # fallbacks
    if "import " in t or "from " in t: return "imports"
    return "unknown"

# --- IO utils ----------------------------------------------------------------

def iter_source_files(root_dir: str, allowed_exts, exclude_dirs):
    """Yield (abs_path, ext) for files under root_dir with an allowed extension."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".")]
        for fname in filenames:
            if "." not in fname:
                continue
            ext = fname.rsplit(".", 1)[-1].lower()
            if ext in allowed_exts:
                yield os.path.join(dirpath, fname), ext

def read_text_safely(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    # last resort
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

# --- main ingestion -----------------------------------------------------------

def chunk_project(
    root_dir: str,
    token_limit: int,
    encoding_name: str,
    output_jsonl: str,
    include_exts=None,
    exclude_dirs=None,
):
    include_exts = set(include_exts or SUPPORTED_EXTS)
    exclude_dirs = set(exclude_dirs or DEFAULT_EXCLUDES)

    repo_id = derive_repo_id(root_dir)

    # one chunker per extension (faster)
    chunkers: dict[str, CodeChunker] = {}
    total_files = 0
    total_chunks = 0

    with open(output_jsonl, "w", encoding="utf-8") as out:
        for abs_path, ext in iter_source_files(root_dir, include_exts, exclude_dirs):
            total_files += 1
            if ext not in chunkers:
                chunkers[ext] = CodeChunker(file_extension=ext, encoding_name=encoding_name)

            code = read_text_safely(abs_path)
            if not code.strip():
                continue

            rel_path = project_rel_path(root_dir, abs_path)
            language = detect_language(ext)
            module = guess_module(rel_path, language)
            file_sha1 = sha1_bytes(code.encode("utf-8", errors="ignore"))

            try:
                chunks = chunkers[ext].chunk(code, token_limit=token_limit)
            except Exception as e:
                print(f"[WARN] Skipping {abs_path} ({ext}): {e}")
                continue

            # Precompute file-level symbols once (cheap) to reuse across chunks
            file_symbols = extract_symbols_from_text(code, language, max_symbols=80)

            for cidx in sorted(chunks):
                chunk = chunks[cidx]
                # Support both dict-valued chunks (new) and string-valued (legacy)
                if isinstance(chunk, dict):
                    text = chunk.get("text", "")
                    start_line = chunk.get("start_line")
                    end_line = chunk.get("end_line")
                else:
                    text = str(chunk)
                    start_line = None
                    end_line = None

                if not text:
                    continue

                n_tokens = count_tokens(text, encoding_name)
                byte_len = len(text.encode("utf-8", errors="ignore"))

                # Per-chunk symbol refinement (mix a few local ids to raise precision)
                local_symbols = extract_symbols_from_text(text, language, max_symbols=30)
                # keep some file-level symbols for recall
                symbols = (local_symbols + [s for s in file_symbols if s not in local_symbols])[:50]

                ast_kind = guess_ast_kind(text, language)

                record = {
                    "id": f"{abs_path}#{cidx}",
                    "repo_id": repo_id,
                    "language": language,
                    "path": abs_path,          # absolute path (for tooling)
                    "rel_path": rel_path,      # project-relative path (for UI/filtering)
                    "module": module,          # dotted module-ish path
                    "ext": ext,
                    "chunk_number": cidx,
                    "symbols": symbols,        # identifiers for BM25 and boosting
                    "ast_kind": ast_kind,      # rough label for analytics/boosts
                    "text": text,
                    "n_tokens": n_tokens,
                    "byte_len": byte_len,
                    "file_sha1": file_sha1,
                }
                if start_line is not None and end_line is not None:
                    record["start_line"] = start_line
                    record["end_line"] = end_line

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Finished.")
    print(f"Repo ID:        {repo_id}")
    print(f"Scanned files:  {total_files}")
    print(f"Wrote chunks:   {total_chunks}")
    print(f"Output JSONL:   {os.path.abspath(output_jsonl)}")

def main():
    ap = argparse.ArgumentParser(description="Chunk an entire codebase into JSONL chunks.")
    ap.add_argument("directory", help="Root directory of the project to chunk")
    ap.add_argument("--token-limit", type=int, default=400, help="Max tokens per chunk")
    ap.add_argument("--encoding", default="gpt-4", help="Tokenizer name used by utils.count_tokens")
    ap.add_argument("--out", default="chunks.jsonl", help="Output JSONL file")
    ap.add_argument("--include-exts", default=",".join(sorted(SUPPORTED_EXTS)),
                    help="Comma-separated list of file extensions to include (no dots)")
    ap.add_argument("--exclude-dirs", default=",".join(sorted(DEFAULT_EXCLUDES)),
                    help="Comma-separated list of directory names to exclude")
    args = ap.parse_args()

    include_exts = {e.strip().lower() for e in args.include_exts.split(",") if e.strip()}
    exclude_dirs = {d.strip() for d in args.exclude_dirs.split(",") if d.strip()}

    chunk_project(
        root_dir=args.directory,
        token_limit=args.token_limit,
        encoding_name=args.encoding,
        output_jsonl=args.out,
        include_exts=include_exts,
        exclude_dirs=exclude_dirs,
    )

if __name__ == "__main__":
    main()