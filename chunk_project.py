# chunk_project.py
import os
import json
import argparse
from datetime import datetime
from Chunker import CodeChunker, Chunker
from utils import count_tokens

# Extensions your CodeParser/CodeChunker supports
SUPPORTED_EXTS = {"py", "js", "jsx", "css", "ts", "tsx", "php", "rb", "go"}

# Common directories to skip
DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "__pycache__", ".mypy_cache",
    "node_modules", "dist", "build", "out", "target", ".venv", "venv", ".tox",
}

def iter_source_files(root_dir: str, allowed_exts, exclude_dirs):
    """Yield (path, ext) for files under root_dir with an allowed extension."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".cache")]
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

    # one chunker per extension (faster)
    chunkers: dict[str, CodeChunker] = {}
    total_files = 0
    total_chunks = 0

    with open(output_jsonl, "w", encoding="utf-8") as out:
        for path, ext in iter_source_files(root_dir, include_exts, exclude_dirs):
            total_files += 1
            if ext not in chunkers:
                chunkers[ext] = CodeChunker(file_extension=ext, encoding_name=encoding_name)

            code = read_text_safely(path)
            if not code.strip():
                continue

            try:
                chunks = chunkers[ext].chunk(code, token_limit=token_limit)
            except Exception as e:
                print(f"[WARN] Skipping {path} ({ext}): {e}")
                continue

            for cidx in sorted(chunks):
                text = chunks[cidx]
                n_tokens = count_tokens(text, encoding_name)
                record = {
                    "id": f"{path}#{cidx}",
                    "path": path,
                    "ext": ext,
                    "chunk_number": cidx,
                    "text": text,
                    "n_tokens": n_tokens,
                    "byte_len": len(text.encode('utf-8')),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Finished.")
    print(f"Scanned files:  {total_files}")
    print(f"Wrote chunks:   {total_chunks}")
    print(f"Output JSONL:   {os.path.abspath(output_jsonl)}")

def main():
    ap = argparse.ArgumentParser(description="Chunk an entire codebase into JSONL chunks.")
    ap.add_argument("directory", help="Root directory of the project to chunk")
    ap.add_argument("--token-limit", type=int, default=1200, help="Max tokens per chunk")
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