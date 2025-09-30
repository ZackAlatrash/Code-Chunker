#!/usr/bin/env python3
import argparse
from typing import Any, Dict, List

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer


INDEX_DEFAULT = "code-chunks"

# Nomic recommends a special prompt for queries.
QUERY_PREFIX = "Represent this query for searching relevant code: "


def embed_query(model: SentenceTransformer, query: str) -> List[float]:
    qtext = QUERY_PREFIX + query.strip()
    vec = model.encode([qtext], normalize_embeddings=True)[0].tolist()
    return vec


def build_filters(ext: str | None, path_substr: str | None, project_id: str | None) -> List[Dict[str, Any]]:
    f: List[Dict[str, Any]] = []
    if project_id:
        f.append({"term": {"project_id": project_id}})
    if ext:
        f.append({"term": {"ext": ext}})
    if path_substr:
        f.append({"wildcard": {"path": {"value": f"*{path_substr}*"}}})
    return f


def build_knn_query(qvec: List[float], k: int, filters: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "size": k,
        "query": {
            "bool": {
                "filter": filters,
                "should": [
                    {"knn": {"embedding": {"vector": qvec, "k": max(k, 50)}}}
                ]
            }
        },
        "_source": ["id", "path", "ext", "chunk_number", "text", "start_line", "end_line", "symbols"]
    }


def build_hybrid_query(query_text: str, qvec: List[float], k: int, filters: List[Dict[str, Any]]) -> Dict[str, Any]:
    should = [
        {"match_phrase": {"text": {"query": query_text, "boost": 6}}},
        {"multi_match": {"query": query_text, "fields": ["symbols^6", "text^2", "path"]}},
        {"knn": {"embedding": {"vector": qvec, "k": max(k, 50)}}},
    ]
    return {
        "size": k,
        "query": {"bool": {"should": should, "filter": filters}},
        "_source": ["id", "path", "ext", "chunk_number", "text", "start_line", "end_line", "symbols"]
    }


def build_exact_bm25_query(query_text: str, k: int, filters: List[Dict[str, Any]]) -> Dict[str, Any]:
    should = [
        {"match_phrase": {"text": {"query": query_text, "boost": 6}}},
        {"multi_match": {"query": query_text, "fields": ["symbols^6", "text^2", "path"]}},
    ]
    return {
        "size": k,
        "query": {"bool": {"should": should, "filter": filters}},
        "_source": ["id", "path", "ext", "chunk_number", "text", "start_line", "end_line", "symbols"]
    }


def pretty_print(res: Dict[str, Any]) -> None:
    hits = res.get("hits", {}).get("hits", [])
    if not hits:
        print("No results found.")
        return
    print("\n=== Search Results ===\n")
    for i, h in enumerate(hits, 1):
        score = h.get("_score")
        s = h["_source"]
        path = s.get("path")
        ext = s.get("ext")
        chunk = s.get("chunk_number")
        ls, le = s.get("start_line"), s.get("end_line")
        print(f"[{i}] score={score:.4f}  file={path}  (ext:{ext}, chunk:{chunk}, lines:{ls}–{le})")
        print(f"    id={s.get('id')}")
        code_lines = (s.get("text") or "").rstrip().splitlines()
        preview = "\n           ".join(code_lines[:12])
        if len(code_lines) > 12:
            preview += "\n           ..."
        print(f"    code:\n           {preview}\n")


def main():
    ap = argparse.ArgumentParser(description="Search code chunks with Nomic Embed Code (kNN / hybrid / exact).")
    ap.add_argument("query", help="Your search query.")
    ap.add_argument("--host", default="http://localhost:9200", help="OpenSearch endpoint")
    ap.add_argument("--index", default=INDEX_DEFAULT, help="Index name")
    ap.add_argument("--k", type=int, default=5, help="Top-k results to return")
    ap.add_argument("--hybrid", action="store_true", help="Use BM25 + kNN hybrid retrieval")
    ap.add_argument("--exact", action="store_true", help="BM25 only (great for identifiers)")
    ap.add_argument("--ext", help="Filter by language extension (e.g., py, go)")
    ap.add_argument("--path-substr", help="Filter by substring in file path")
    ap.add_argument("--project-id", help="Restrict to a project_id (if you indexed it)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"], help="Device for embedding model")
    args = ap.parse_args()

    client = OpenSearch(args.host)
    filters = build_filters(args.ext, args.path_substr, args.project_id)

    if args.exact:
        body = build_exact_bm25_query(args.query, args.k, filters)
        res = client.search(index=args.index, body=body)
        pretty_print(res)
        return

    print("[INFO] Loading model: nomic-ai/nomic-embed-code")
    model = SentenceTransformer("nomic-ai/nomic-embed-code", device=args.device)
    qvec = embed_query(model, args.query)

    if args.hybrid:
        body = build_hybrid_query(args.query, qvec, args.k, filters)
    else:
        body = build_knn_query(qvec, args.k, filters)

    res = client.search(index=args.index, body=body)
    pretty_print(res)


if __name__ == "__main__":
    main()