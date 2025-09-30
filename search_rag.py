#!/usr/bin/env python3
import argparse
from typing import List, Dict, Any, Optional
from collections import defaultdict
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

DEFAULT_CHUNKS_INDEX = "code_chunks_v2"   # your main chunks index
DEFAULT_ROUTER_INDEX = "repo_router_v1"   # one doc per repo (router)

# -------------------- kNN helpers (version tolerant) --------------------

def knn_search_filtered(client: OpenSearch, index: str, field: str, qvec: List[float],
                        repo_ids: List[str], k: int, num_candidates: int,
                        source_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Try filtered kNN in several forms (newer -> older). If all fail,
    run plain kNN and filter client-side.
    """
    _src = source_fields or True

    # 1) bool.filter + top-level knn (newer OpenSearch)
    body1 = {
        "size": k,
        "query": {"bool": {"filter": [{"terms": {"repo_id": repo_ids}}]}},
        "knn": {"field": field, "query_vector": qvec, "k": k, "num_candidates": num_candidates},
        "_source": _src
    }
    try:
        return client.search(index=index, body=body1)["hits"]["hits"]
    except Exception:
        pass

    # 2) filter inside knn block (some versions)
    body2 = {
        "size": k,
        "knn": {
            "field": field, "query_vector": qvec, "k": k, "num_candidates": num_candidates,
            "filter": {"terms": {"repo_id": repo_ids}}
        },
        "_source": _src
    }
    try:
        return client.search(index=index, body=body2)["hits"]["hits"]
    except Exception:
        pass

    # 3) Older syntax: query.bool.should with knn nested by field name
    #    {"query": {"bool": {"filter": ..., "should": [{"knn": {"<field>": {"vector": qvec, "k": k}}]}}}
    body3 = {
        "size": k,
        "query": {
            "bool": {
                "filter": [{"terms": {"repo_id": repo_ids}}],
                "should": [{"knn": {field: {"vector": qvec, "k": k}}}]
            }
        },
        "_source": _src
    }
    try:
        return client.search(index=index, body=body3)["hits"]["hits"]
    except Exception:
        pass

    # 4) Older syntax: query.knn without filter, then filter client-side
    body4 = {
        "size": max(k, num_candidates),
        "query": {"knn": {field: {"vector": qvec, "k": max(k, num_candidates)}}},
        "_source": _src
    }
    try:
        res = client.search(index=index, body=body4)["hits"]["hits"]
        allowed = set(repo_ids)
        return [h for h in res if h.get("_source", {}).get("repo_id") in allowed][:k]
    except Exception:
        pass

    # 5) Last-resort fallback: script_score with cosineSimilarity (Elasticsearch-style dense_vector)
    #    If the backend doesn't support this either, return empty list rather than crash.
    body5 = {
        "size": k,
        "query": {
            "script_score": {
                "query": {"bool": {"filter": [{"terms": {"repo_id": repo_ids}}]}},
                "script": {
                    "source": f"cosineSimilarity(params.qvec, '{field}') + 1.0",
                    "params": {"qvec": qvec}
                }
            }
        },
        "_source": _src
    }
    try:
        return client.search(index=index, body=body5)["hits"]["hits"]
    except Exception:
        # Can't do vector similarity on this cluster; give up gracefully
        return []

# -------------------- Query builders --------------------

def router_query(query_text: str, topn: int) -> Dict[str, Any]:
    return {
        "size": topn,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["short_title^3", "summary^2", "domains^2", "key_modules", "key_symbols", "tech_stack"]
            }
        }
    }

def bm25_filtered_query(query_text: str, repo_ids: List[str], size: int,
                        source_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "size": size,
        "query": {
            "bool": {
                "filter": [{"terms": {"repo_id": repo_ids}}],
                "should": [
                    {"multi_match": {"query": query_text, "fields": ["text^3", "symbols^4", "module", "rel_path"]}}
                ]
            }
        },
        "_source": source_fields or True
    }

def rrf_fuse(lists: List[List[Dict[str, Any]]], K: int = 30, k_const: int = 60) -> List[Dict[str, Any]]:
    scores = defaultdict(float)
    payload: Dict[str, Dict[str, Any]] = {}
    for L in lists:
        for rank, h in enumerate(L, start=1):
            _id = h["_id"]
            scores[_id] += 1.0 / (k_const + rank)
            payload[_id] = h
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:K]
    return [payload[_id] for _id, _ in ranked]

# -------------------- Pretty print --------------------

def pretty_print(hits: List[Dict[str, Any]]) -> None:
    if not hits:
        print("No results.")
        return
    print("\n=== Search Results ===\n")
    for i, h in enumerate(hits, 1):
        src = h.get("_source", {})
        rel = src.get("rel_path") or src.get("path")
        sl, el = src.get("start_line"), src.get("end_line")
        lines_info = f", lines {sl}-{el}" if sl is not None and el is not None else ""
        print(f"[{i}] Repo: {src.get('repo_id')}  File: {rel}  (chunk {src.get('chunk_number')}{lines_info})")
        print(f"    Score: {h.get('_score')}  Lang: {src.get('language')}  Ext: {src.get('ext')}")
        print(f"    ID: {src.get('id')}")
        # print full chunk (you asked for all lines previously)
        code = src.get("text", "").rstrip()
        if code:
            print("    Code:")
            print("           " + "\n           ".join(code.splitlines()))
        print()

# -------------------- Main search flow --------------------

def main():
    ap = argparse.ArgumentParser(description="Router → filtered hybrid (BM25 + kNN) → RRF over code chunks.")
    ap.add_argument("query", help="Natural-language query")
    ap.add_argument("--host", default="http://localhost:9200", help="OpenSearch endpoint")
    ap.add_argument("--chunks-index", default=DEFAULT_CHUNKS_INDEX, help="Chunks index name")
    ap.add_argument("--router-index", default=DEFAULT_ROUTER_INDEX, help="Router index name")
    ap.add_argument("--explicit-repo", default=None, help="Skip router and search only this repo_id")
    ap.add_argument("--top-repos", type=int, default=2, help="Router: how many repos to consider")
    ap.add_argument("--bm25-size", type=int, default=200)
    ap.add_argument("--knn-size", type=int, default=200)
    ap.add_argument("--knn-candidates", type=int, default=400)
    ap.add_argument("--final-k", type=int, default=10)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--vector-field", default="vector", help="knn_vector field name in chunks index")
    args = ap.parse_args()

    client = OpenSearch(hosts=[args.host])
    model = SentenceTransformer(args.model)

    # 1) Router (or explicit repo)
    if args.explicit_repo:
        repo_ids = [args.explicit_repo]
    else:
        rbody = router_query(args.query, args.top_repos)
        rres = client.search(index=args.router_index, body=rbody)
        repo_ids = [h["_source"]["repo_id"] for h in rres["hits"]["hits"]]
        if not repo_ids:
            print("Router found no matching repo. Try --explicit-repo or check router index content.")
            return

    # 2) Embed query
    qvec = model.encode(args.query, normalize_embeddings=True).tolist()

    # 3) Candidate generation (BM25 + kNN), both filtered by repo_ids
    source_fields = [
        "id","repo_id","language","path","rel_path","module","ext",
        "chunk_number","start_line","end_line","symbols","text","n_tokens"
    ]
    bbody = bm25_filtered_query(args.query, repo_ids, args.bm25_size, source_fields)
    bhits = client.search(index=args.chunks_index, body=bbody)["hits"]["hits"]

    khits = knn_search_filtered(
        client=client,
        index=args.chunks_index,
        field=args.vector_field,
        qvec=qvec,
        repo_ids=repo_ids,
        k=args.knn_size,
        num_candidates=args.knn_candidates,
        source_fields=source_fields
    )

    # 4) Fuse with RRF and take final_k
    fused = rrf_fuse([bhits, khits], K=args.final_k)

    # 5) Print results
    print("Routed repos:", repo_ids)
    pretty_print(fused)

if __name__ == "__main__":
    main()
