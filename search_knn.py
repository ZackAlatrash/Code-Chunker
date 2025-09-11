#!/usr/bin/env python3
import argparse
from typing import List, Dict, Any
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

INDEX = "code-chunks"


def build_knn_query(qvec: List[float], k: int) -> Dict[str, Any]:
    """Pure vector search query."""
    return {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": qvec,
                    "k": k
                }
            }
        },
        "_source": ["id", "path", "ext", "chunk_number", "text"]
    }


def build_hybrid_query(query_text: str, qvec: List[float], k: int) -> Dict[str, Any]:
    """Hybrid = BM25 (keyword) + kNN (semantic)."""
    return {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    {"multi_match": {"query": query_text, "fields": ["text^2", "path"]}},
                    {"knn": {"embedding": {"vector": qvec, "k": max(k, 50)}}}
                ]
            }
        },
        "_source": ["id", "path", "ext", "chunk_number", "text"]
    }


def pretty_print(res: Dict[str, Any]) -> None:
    hits = res.get("hits", {}).get("hits", [])
    if not hits:
        print("No results found.")
        return

    print("\n=== Search Results ===\n")
    for i, h in enumerate(hits, 1):
        src = h["_source"]
        print(f"[{i}] File: {src.get('path')}  (ext: {src.get('ext')}, chunk: {src.get('chunk_number')})")
        print(f"    ID: {src.get('id')}")
        print("    Code:")
        # show first ~6 lines of the chunk, trimmed
        lines = src.get("text", "").strip().splitlines()
        preview = "\n           ".join(lines[:6])
        if len(lines) > 6:
            preview += "\n           ..."
        print(f"           {preview}")
        print()


def main():
    p = argparse.ArgumentParser(description="Semantic (kNN) and hybrid search over code chunks.")
    p.add_argument("query", help="Natural-language query or keyword")
    p.add_argument("--host", default="http://localhost:9200", help="OpenSearch endpoint")
    p.add_argument("--index", default=INDEX, help="Index name")
    p.add_argument("--k", type=int, default=5, help="Top-k results")
    p.add_argument("--hybrid", action="store_true", help="Use hybrid (BM25 + vector) instead of pure kNN")
    args = p.parse_args()

    # Load model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qvec = model.encode([args.query], normalize_embeddings=True)[0].tolist()

    # Build query
    if args.hybrid:
        body = build_hybrid_query(args.query, qvec, args.k)
    else:
        body = build_knn_query(qvec, args.k)

    # Connect & search
    client = OpenSearch(args.host)
    res = client.search(index=args.index, body=body)

    # Pretty output
    pretty_print(res)


if __name__ == "__main__":
    main()