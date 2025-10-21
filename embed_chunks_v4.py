#!/usr/bin/env python3
"""
Embed all JSON chunks in ./ChunksV3 into OpenSearch index `code_chunks_v4`.

Usage:
  # 1) (Recommended) Use sentence-transformers locally (384-dim):
  pip install opensearch-py tqdm sentence-transformers

  # 2) Or provide an HTTP embedding endpoint that returns {"embedding": [384 floats]}:
  export EMBEDDING_API_URL=http://localhost:8000/embed

  # Required OpenSearch env:
  export OPENSEARCH_HOST=https://localhost:9200
  export OPENSEARCH_USER=admin
  export OPENSEARCH_PASS=admin
  # If you use self-signed certs, set:
  export OPENSEARCH_VERIFY_SSL=false

Run:
  python embed_chunks_v4.py --chunks-dir ./ChunksV3 --index code_chunks_v4 --batch-size 500
"""

import os
import json
import glob
import argparse
import time
from typing import Dict, Any, List, Optional

import urllib3
from tqdm import tqdm

from opensearchpy import OpenSearch, RequestsHttpConnection, helpers

# ---------------------------
# Config / Clients
# ---------------------------

def make_opensearch() -> OpenSearch:
    host = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
    user = os.getenv("OPENSEARCH_USER", "admin")
    pwd  = os.getenv("OPENSEARCH_PASS", "admin")
    verify = os.getenv("OPENSEARCH_VERIFY_SSL", "true").lower() == "true"

    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    client = OpenSearch(
        hosts=[host],
        http_auth=(user, pwd),
        use_ssl=host.startswith("https"),
        verify_certs=verify,
        ssl_assert_hostname=False if not verify else True,
        ssl_show_warn=verify,
        connection_class=RequestsHttpConnection,
        timeout=120,
        max_retries=5,
        retry_on_timeout=True,
    )
    return client


# ---------------------------
# Embedding (384-dim)
# ---------------------------

# Strategy:
# 1) Try sentence-transformers "all-MiniLM-L6-v2" (384-dim).
# 2) If not installed, fall back to HTTP endpoint in EMBEDDING_API_URL.

_EMBED_MODEL = None

def init_local_model():
    global _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: 384-dim
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Loaded local sentence-transformers model (all-MiniLM-L6-v2)")
    except Exception as e:
        print(f"⚠️  Could not load sentence-transformers: {e}")
        _EMBED_MODEL = None


def embed_text_384(text: str) -> List[float]:
    # Try local model
    if _EMBED_MODEL is not None:
        vec = _EMBED_MODEL.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()

    # Fallback to HTTP
    import requests
    url = os.getenv("EMBEDDING_API_URL")
    if not url:
        raise RuntimeError(
            "No local sentence-transformers model and EMBEDDING_API_URL not set. "
            "Install sentence-transformers or provide an HTTP endpoint."
        )
    r = requests.post(url, json={"text": text}, timeout=60)
    r.raise_for_status()
    data = r.json()
    vec = data.get("embedding")
    if not isinstance(vec, list) or len(vec) != 384:
        raise RuntimeError(f"Embedding service returned invalid vector length: {len(vec) if isinstance(vec, list) else 'N/A'}")
    return vec


# ---------------------------
# Embedding text composer
# ---------------------------

def first_declaration_line(doc_head: Optional[str], text: str) -> str:
    """
    Try to take the first declaration/signature line from doc_head; if not present,
    fall back to the first non-empty line from `text`.
    """
    if doc_head:
        for line in doc_head.splitlines():
            # pick a non-empty line that looks like code or a signature
            if line.strip():
                return line.strip()
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def trim_code(code: str, max_chars: int = 1200) -> str:
    if len(code) <= max_chars:
        return code
    # Heuristic: keep header + tail
    head = code[: int(max_chars * 0.7)]
    tail = code[-int(max_chars * 0.25):]
    return head + "\n...\n" + tail


def compose_embedding_text(chunk: Dict[str, Any]) -> str:
    repo_id   = chunk.get("repo_id", "")
    rel_path  = chunk.get("rel_path") or chunk.get("path", "")
    language  = chunk.get("language", "")
    primary_symbol = chunk.get("primary_symbol", "")
    primary_kind   = chunk.get("primary_kind", "")
    all_roles      = chunk.get("all_roles", [])
    summary_en     = chunk.get("summary_en", "")
    keywords_en    = chunk.get("keywords_en", [])
    doc_head       = chunk.get("doc_head", "")
    text           = chunk.get("text", "")

    signature_line = first_declaration_line(doc_head, text)
    code_block     = trim_code(text)

    # Compose labeled sections for stable embeddings
    lines = []
    lines.append(f"[repo]: {repo_id}")
    lines.append(f"[path]: {rel_path}")
    lines.append(f"[lang]: {language}")
    lines.append(f"[symbol]: {primary_symbol} ({primary_kind})")
    if all_roles:
        lines.append(f"[roles]: {', '.join(map(str, all_roles))}")

    if summary_en:
        lines.append(f"\n[summary]: {summary_en}")
    if keywords_en:
        lines.append(f"[keywords]: {', '.join(map(str, keywords_en))}")

    if signature_line:
        lines.append(f"\n[signature]:\n{signature_line}")

    lines.append(f"\n[code]:\n{code_block}")
    return "\n".join(lines)


# ---------------------------
# I/O helpers
# ---------------------------

def iter_chunk_files(chunks_dir: str):
    """Iterate over .json and .jsonl files in chunks_dir."""
    json_pattern = os.path.join(chunks_dir, "**", "*.json")
    jsonl_pattern = os.path.join(chunks_dir, "**", "*.jsonl")
    
    for path in glob.iglob(json_pattern, recursive=True):
        yield path
    for path in glob.iglob(jsonl_pattern, recursive=True):
        yield path


def load_chunks_from_file(path: str) -> List[Dict[str, Any]]:
    """
    Load chunks from either:
    - .json file (single object or array)
    - .jsonl file (one JSON object per line)
    """
    chunks = []
    
    if path.endswith('.jsonl'):
        # JSONL format: one JSON object per line
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"[WARN] {path}:{line_num}: Invalid JSON ({e})")
    else:
        # JSON format: single object or array
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            chunks = [data]
        elif isinstance(data, list):
            chunks = data
        else:
            raise ValueError(f"Unsupported JSON top-level in {path}: {type(data)}")
    
    return chunks


# ---------------------------
# Indexing
# ---------------------------

def make_action(chunk: Dict[str, Any], index_name: str) -> Dict[str, Any]:
    # Build embedding_text
    embedding_text = compose_embedding_text(chunk)
    # Embed
    vec = embed_text_384(embedding_text)
    if len(vec) != 384:
        raise ValueError(f"Embedding length must be 384, got {len(vec)}")

    # Use provided id, else derive a stable id
    doc_id = chunk.get("id") or f"{chunk.get('repo_id','')}-{chunk.get('rel_path','')}-{chunk.get('chunk_number','')}"

    # Construct the doc with important metadata + embedding + raw embedding_text
    body = dict(chunk)  # keep original fields for traceability
    body["embedding"] = vec
    body["embedding_text"] = embedding_text

    return {
        "_op_type": "index",       # overwrite/upsert behavior
        "_index": index_name,
        "_id": doc_id,
        "_source": body,
    }


def bulk_index(client: OpenSearch, actions: List[Dict[str, Any]], batch_size: int = 500):
    # Use opensearch-py's bulk helper with small batches to avoid payload bloat
    for i in range(0, len(actions), batch_size):
        batch = actions[i : i + batch_size]
        success, errors = helpers.bulk(client, batch, request_timeout=180, refresh=False)
        if errors:
            # helpers.bulk returns (success_count, [])
            # If it returns a list in errors (older versions), print it:
            print(f"[WARN] Bulk returned errors for batch {i//batch_size}: {errors}")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed V3 chunks into OpenSearch")
    parser.add_argument("--chunks-dir", default="./ChunksV3", help="Directory containing chunk files")
    parser.add_argument("--index", default="code_chunks_v4", help="OpenSearch index name")
    parser.add_argument("--batch-size", type=int, default=500, help="Bulk indexing batch size")
    args = parser.parse_args()

    print("=" * 80)
    print("EMBED CHUNKS V4 - OpenSearch Indexer")
    print("=" * 80)
    
    init_local_model()
    client = make_opensearch()
    
    # Test connection
    try:
        info = client.info()
        print(f"✅ Connected to OpenSearch: {info.get('version', {}).get('number', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to connect to OpenSearch: {e}")
        return

    actions: List[Dict[str, Any]] = []
    files = list(iter_chunk_files(args.chunks_dir))
    
    if not files:
        print(f"❌ No JSON/JSONL files found under {args.chunks_dir}")
        return

    print(f"📁 Discovered {len(files)} JSON/JSONL files under {args.chunks_dir}")
    print(f"🎯 Target index: {args.index}")
    print(f"📦 Batch size: {args.batch_size}")
    print()

    t0 = time.time()
    total_chunks = 0
    skipped = 0

    for file_path in tqdm(files, desc="Indexing files"):
        try:
            chunks = load_chunks_from_file(file_path)
        except Exception as e:
            print(f"\n[SKIP] {file_path}: failed to parse ({e})")
            continue

        for chunk in chunks:
            try:
                action = make_action(chunk, args.index)
                actions.append(action)
                total_chunks += 1
            except Exception as e:
                print(f"\n[SKIP] {file_path}: {e}")
                skipped += 1

        # Flush periodically to reduce memory
        if len(actions) >= args.batch_size:
            bulk_index(client, actions, batch_size=args.batch_size)
            actions.clear()

    if actions:
        bulk_index(client, actions, batch_size=args.batch_size)

    dt = time.time() - t0
    
    print()
    print("=" * 80)
    print("INDEXING COMPLETE")
    print("=" * 80)
    print(f"✅ Indexed: {total_chunks} chunks")
    print(f"⚠️  Skipped: {skipped} chunks")
    print(f"⏱️  Time: {dt:.1f}s ({total_chunks/dt:.1f} chunks/sec)")
    print(f"🔍 Index: {args.index}")
    print("=" * 80)


if __name__ == "__main__":
    main()

