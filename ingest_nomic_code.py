#!/usr/bin/env python3
"""
Ingest code chunks into OpenSearch using Nomic Embed Code embeddings.

Input JSONL record shape (kept exactly as-is):
{
  "id": "/abs/path/file.py#1",
  "path": "/abs/path/file.py",
  "ext": "py",
  "chunk_number": 1,
  "text": "chunk text ...",
  "n_tokens": 197,
  "byte_len": 886,
  "start_line": 1,
  "end_line": 36
}

This script:
- streams the JSONL file line-by-line (memory efficient),
- encodes `text` with `nomic-ai/nomic-embed-code`,
- L2-normalizes vectors (good for cosine similarity),
- bulk-indexes into OpenSearch (adds "embedding" field).
"""

import argparse
import json
from typing import Dict, Any, Iterable, List, Tuple

from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer


def parse_args():
    ap = argparse.ArgumentParser(description="Ingest code chunks with Nomic Embed Code embeddings.")
    ap.add_argument("--index", required=True, help="OpenSearch index name (must exist with knn_vector mapping).")
    ap.add_argument("--jsonl", required=True, help="Path to chunks JSONL.")
    ap.add_argument("--host", default="http://localhost:9200", help="OpenSearch endpoint (default: http://localhost:9200)")
    ap.add_argument("--batch", type=int, default=256, help="Batch size for bulk ingest (default: 256)")
    ap.add_argument("--timeout", type=int, default=300, help="request_timeout seconds for bulk calls (default: 300)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                    help="Where to run the embedding model (default: cpu)")
    ap.add_argument("--model-batch", type=int, default=8,
                    help="Batch size for model.encode (default 8; lower if you OOM)")
    ap.add_argument("--max-seq-len", type=int, default=512,
                    help="Optional: cap model max sequence length to save memory")
    return ap.parse_args()


def stream_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def check_index_dimension(client: OpenSearch, index: str, expected_dim: int) -> None:
    """
    Sanity-check that index has an 'embedding' knn_vector with the right 'dimension'.
    If no mapping is present, we just warn and continue (user may be creating it separately).
    """
    try:
        mapping = client.indices.get_mapping(index=index)
        props = mapping.get(index, {}).get("mappings", {}).get("properties", {})
        emb = props.get("embedding")
        if emb and isinstance(emb, dict):
            dim = emb.get("dimension")
            if dim is not None and dim != expected_dim:
                raise SystemExit(
                    f"[ERROR] Index '{index}' has embedding.dimension={dim}, "
                    f"but model emits {expected_dim}. Recreate the index with the correct dimension."
                )
    except Exception as e:
        # Don’t hard-fail on mapping introspection; user may not want this check to block.
        print(f"[WARN] Could not verify index mapping dimension: {e}")


def build_actions(
    model: SentenceTransformer,
    batch: List[Dict[str, Any]],
    index: str,
    mb_size: int,
) -> Iterable[Dict[str, Any]]:
    """
    For a batch of docs, produce bulk index actions with normalized embeddings.
    - We encode ONLY the 'text' field (leaving your stored 'text' unchanged).
    - We do NOT modify any other fields.
    - We append 'embedding': [floats] to the source.
    """
    def encode_microbatch(texts: List[str], mb_size: int) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), mb_size):
            sub = texts[i:i + mb_size]
            out.extend(
                model.encode(
                    sub,
                    normalize_embeddings=True,
                    batch_size=mb_size,
                ).tolist()
            )
        return out

    texts = [d["text"] for d in batch]
    vecs = encode_microbatch(texts, mb_size)

    for d, v in zip(batch, vecs):
        src = dict(d)  # shallow copy to avoid mutating the input
        src["embedding"] = v
        # NOTE: we assume 'id' is unique and stable across re-ingests
        yield {
            "_op_type": "index",
            "_index": index,
            "_id": src["id"],
            "_source": src,
        }


def main():
    args = parse_args()

    # 1) Load model (runs locally; uses CPU/MPS/CUDA depending on your PyTorch install)
    #    No query prefix here: we are ingesting documents (code). Query-time will handle query prompts separately.
    print("[INFO] Loading model: nomic-ai/nomic-embed-code")
    model = SentenceTransformer("nomic-ai/nomic-embed-code", device=args.device)
    model.max_seq_length = args.max_seq_len
    model_dim = model.get_sentence_embedding_dimension()
    print(f"[INFO] Model embedding dimension: {model_dim}")

    # 2) Connect to OpenSearch
    client = OpenSearch(args.host)

    # 3) Sanity-check index mapping dimension if possible
    check_index_dimension(client, args.index, model_dim)

    # 4) Stream → batch → bulk
    buf: List[Dict[str, Any]] = []
    total_ok = 0
    total_seen = 0

    for d in stream_jsonl(args.jsonl):
        # quick validation
        if "id" not in d or "text" not in d:
            print(f"[WARN] Skipping record without 'id' or 'text': {d}")
            continue

        buf.append(d)
        total_seen += 1

        if len(buf) >= args.batch:
            ok, errors = helpers.bulk(
                client, build_actions(model, buf, args.index, args.model_batch),
                request_timeout=args.timeout, raise_on_error=False
            )
            total_ok += ok
            buf.clear()
            if errors:
                # Show one error for brevity; you can write all to a dead-letter file if needed
                print(f"[WARN] Some items failed in last batch (showing one): {errors[:1]}")

    # 5) Final flush
    if buf:
        ok, errors = helpers.bulk(
            client, build_actions(model, buf, args.index, args.model_batch),
            request_timeout=args.timeout, raise_on_error=False
        )
        total_ok += ok
        if errors:
            print(f"[WARN] Some items failed in final batch (showing one): {errors[:1]}")

    print(f"[DONE] Seen: {total_seen}  |  Indexed OK: {total_ok}")


if __name__ == "__main__":
    main()
