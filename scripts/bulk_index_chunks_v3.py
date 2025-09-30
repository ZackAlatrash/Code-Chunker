#!/usr/bin/env python3
"""
Bulk indexer for code_chunks_v3 (keeps your old indexer intact).

Usage:
  python scripts/bulk_index_chunks_v3.py run/myrepo_chunks_v3.jsonl http://localhost:9200 \
    --index code_chunks_v3

Options:
  --index           Target index (default: code_chunks_v3)
  --id-field        Field to use as _id (default: id)
  --batch           Bulk batch size (default: 1000)
  --timeout         Request timeout seconds (default: 120)
  --max             Only index first N docs (for smoke tests)
  --dry-run         Parse and validate only; do not index
  --drop-missing-span  If primary_span is not list[int], drop it instead of failing
  --strip-empty-text   Drop docs where 'text' is empty/whitespace
"""

from __future__ import annotations
import argparse, json, sys
from typing import Iterator, Dict, Any, List
from opensearchpy import OpenSearch, helpers

def read_jsonl(path: str, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and n >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1

def _is_list_of_ints(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, int) for x in v)

def normalize_doc(doc: Dict[str, Any], drop_missing_span: bool, strip_empty_text: bool) -> Dict[str, Any]:
    # Ensure primary_span is either list[int] or absent
    if "primary_span" in doc and not _is_list_of_ints(doc["primary_span"]):
        if drop_missing_span:
            doc.pop("primary_span", None)
        else:
            raise ValueError(f"primary_span must be list[int], got {type(doc['primary_span'])}")

    # Ensure list fields are lists (some pipelines might serialize as set/tuple)
    for lf in ("def_symbols", "symbols"):
        if lf in doc and not isinstance(doc[lf], list):
            # try to coerce from string "a, b, c"
            v = doc[lf]
            if isinstance(v, str):
                doc[lf] = [s.strip() for s in v.split(",") if s.strip()]
            else:
                doc[lf] = list(v)

    if strip_empty_text:
        txt = doc.get("text", "")
        if not isinstance(txt, str) or not txt.strip():
            # mark as skip by setting a flag the generator will check
            doc["_skip"] = True

    return doc

def gen_actions(jsonl_path: str, index: str, id_field: str,
                max_docs: int | None, drop_missing_span: bool, strip_empty_text: bool) -> Iterator[Dict[str, Any]]:
    for doc in read_jsonl(jsonl_path, limit=max_docs):
        doc = normalize_doc(doc, drop_missing_span, strip_empty_text)
        if doc.get("_skip"):
            continue
        _id = doc.get(id_field)
        yield {
            "_op_type": "index",
            "_index": index,
            "_id": _id,
            "_source": doc,
        }

def main():
    ap = argparse.ArgumentParser(description="Bulk index chunks JSONL into OpenSearch (v3-safe).")
    ap.add_argument("jsonl", help="Path to chunks JSONL produced by build_chunks_v2.py")
    ap.add_argument("host", help="OpenSearch endpoint, e.g. http://localhost:9200")
    ap.add_argument("--index", default="code_chunks_v3", help="Target index name")
    ap.add_argument("--id-field", default="id", help="Field to use as _id")
    ap.add_argument("--batch", type=int, default=1000, help="Bulk batch size")
    ap.add_argument("--timeout", type=int, default=120, help="Request timeout (s)")
    ap.add_argument("--max", type=int, default=None, help="Index only first N docs (debug)")
    ap.add_argument("--dry-run", action="store_true", help="Validate but do not index")
    ap.add_argument("--drop-missing-span", action="store_true", help="Drop invalid primary_span instead of failing")
    ap.add_argument("--strip-empty-text", action="store_true", help="Skip docs with empty/whitespace 'text'")
    args = ap.parse_args()

    # Validate parse first to fail early (read a handful)
    try:
        for _ in read_jsonl(args.jsonl, limit=3):
            pass
    except Exception as e:
        print(f"[ERROR] Failed to read {args.jsonl}: {e}", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        # Do a normalization pass on up to 10 docs
        try:
            for i, doc in enumerate(read_jsonl(args.jsonl, limit=10), start=1):
                normalize_doc(doc, args.drop_missing_span, args.strip_empty_text)
            print(f"Dry-run OK: parsed and validated {min(i,10)} docs from {args.jsonl}")
        except Exception as e:
            print(f"[ERROR] Dry-run validation failed: {e}", file=sys.stderr)
            sys.exit(3)
        sys.exit(0)

    client = OpenSearch(hosts=[args.host])
    success, fail = helpers.bulk(
        client,
        gen_actions(args.jsonl, args.index, args.id_field, args.max, args.drop_missing_span, args.strip_empty_text),
        chunk_size=args.batch,
        request_timeout=args.timeout
    )
    print(f"✅ Bulk done → success={success}, failed={fail}, index={args.index}")

if __name__ == "__main__":
    main()