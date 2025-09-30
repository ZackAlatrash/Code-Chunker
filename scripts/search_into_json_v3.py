#!/usr/bin/env python3
"""
search_into_json_v3.py
----------------------
Enhanced search script for the v3 stack:
- code_chunks_v3: Symbol-aware chunking with primary_symbol, def_symbols, symbols, doc_head
- repo_router_v2: LLM-enriched router with enhanced metadata
- repo_guide_v1: LLM-generated repo guides

Key improvements over search_into_json.py:
- Leverages symbol metadata for better relevance
- Enhanced router queries using new router v2 fields
- Symbol-aware search scoring
- Enhanced LLM bundle with symbol information

Usage:
  python scripts/search_into_json_v3.py "how does authentication work?"
  python scripts/search_into_json_v3.py "find JWT token validation" --explicit-repo myapp
"""

import argparse, json, time
from typing import List, Dict, Any, Optional
import re
from collections import defaultdict
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

DEFAULT_CHUNKS_INDEX = "code_chunks_v3"    # Enhanced symbol-aware chunks
DEFAULT_ROUTER_INDEX = "repo_router_v2"    # LLM-enriched router
DEFAULT_REPO_GUIDE_INDEX = "repo_guide_v1" # LLM-enriched repo guides

# -------------------- Helpers for compact LLM bundle --------------------

def _truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    t = text.strip()
    return t if len(t) <= max_chars else t[:max_chars].rstrip() + "…"


def fetch_router_context(client, router_index: str, repo_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch compact repo context for the routed repos using repo_router_v2 fields."""
    if not repo_ids:
        return []

    # Prefer mget for exact IDs (fast & precise)
    docs = [{"_index": router_index, "_id": rid} for rid in repo_ids]
    res = client.mget(body={"docs": docs})
    out = []
    for d in res.get("docs", []):
        if not d.get("found"):
            continue
        s = d.get("_source", {})
        out.append({
            "repo_id": s.get("repo_id"),
            "short_title": _truncate(s.get("short_title", ""), 200),
            "summary": _truncate(s.get("summary", ""), 1000),
            "languages": s.get("languages", [])[:5],
            "tech_stack": s.get("tech_stack", [])[:10],
            "modules": s.get("modules", [])[:15], 
            "important_files": s.get("important_files", [])[:12],
            "key_symbols": _truncate(s.get("key_symbols", ""), 400),
            "keywords": _truncate(s.get("keywords", ""), 300),
        })
    return out


def _clip_by_lines(text: str, max_lines: int = 120) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + "\n# …(trimmed)…"


def build_llm_bundle_v3(query: str, repo_ids, fused_hits, repo_context: List[Dict[str, Any]],
                       max_lines_per_chunk: int = 120,
                       repo_guides: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Enhanced bundle builder that includes symbol metadata from code_chunks_v3."""
    sources = []
    for i, h in enumerate(fused_hits, 1):
        src = h.get("_source", {})
        code = _clip_by_lines(src.get("text", ""), max_lines=max_lines_per_chunk)
        sources.append({
            "idx": i,
            "repo_id": src.get("repo_id"),
            "rel_path": src.get("rel_path") or src.get("path"),
            "start_line": src.get("start_line"),
            "end_line": src.get("end_line"),
            "chunk_number": src.get("chunk_number"),
            "score": h.get("_score"),
            "code": code,
            # New v3 symbol fields
            "primary_symbol": src.get("primary_symbol", ""),
            "primary_kind": src.get("primary_kind", ""),
            "primary_span": src.get("primary_span", []),
            "def_symbols": src.get("def_symbols", [])[:10],  # Limit for readability
            "symbols": src.get("symbols", [])[:20],  # Top symbols for context
            "doc_head": _truncate(src.get("doc_head", ""), 500),
        })
    return {
        "version": "1.3",  # Incremented for v3 enhancements
        "query": query,
        "routed_repo_ids": repo_ids,
        "repos": repo_context,
        "repo_guides": repo_guides or [],
        "sources": sources
    }


def fetch_repo_guides(client: OpenSearch, index: str, repo_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Return a list of compact repo guides for the routed repos.
    Each item: {repo_id, overview, key_flows, entrypoints, languages, modules}
    Missing repos are silently skipped.
    """
    if not repo_ids:
        return []
    docs = [{"_index": index, "_id": rid} for rid in repo_ids]
    try:
        res = client.mget(body={"docs": docs})
    except Exception:
        return []
    out = []
    for d in res.get("docs", []):
        if not d.get("found"):
            continue
        s = d.get("_source", {})
        out.append({
            "repo_id": s.get("repo_id"),
            "overview": s.get("overview", ""),
            "key_flows": s.get("key_flows", ""),
            "entrypoints": s.get("entrypoints", ""),
            "languages": s.get("languages", []),
            "modules": s.get("modules", ""),
        })
    return out

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

    # 5) Last-resort fallback: script_score with cosineSimilarity
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
        return []

# -------------------- Enhanced Query builders --------------------

def router_query_v2(query_text: str, topn: int) -> Dict[str, Any]:
    """Enhanced router query leveraging repo_router_v2 fields."""
    return {
        "size": topn,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "short_title^4",        # Enhanced title
                    "summary^3",            # LLM-generated summary
                    "key_symbols^5",        # Important symbols (highest weight)
                    "keywords^4",           # Curated keywords
                    "synonyms^3",           # Alternative terms
                    "tech_stack^2",         # Technology stack
                    "modules^2",            # Key modules
                    "important_files^1.5",  # Important files
                    "sample_queries^2"      # Example queries
                ]
            }
        }
    }

def bm25_filtered_query_v3(query_text: str, repo_ids: List[str], size: int,
                          source_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Enhanced BM25 query leveraging symbol metadata from code_chunks_v3."""
    return {
        "size": size,
        "query": {
            "bool": {
                "filter": [{"terms": {"repo_id": repo_ids}}],
                "should": [
                    {"multi_match": {
                        "query": query_text, 
                        "fields": [
                            "text^3",              # Code content
                            "primary_symbol^6",    # Main symbol (highest weight)
                            "def_symbols^5",       # Defined symbols
                            "symbols^4",           # All symbols
                            "doc_head^3",          # Documentation
                            "rel_path^2",          # File path
                            "primary_kind^2"       # Symbol type (function, class, etc.)
                        ]
                    }}
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

# -------------------- Pretty print with enhanced info --------------------

def pretty_print_v3(hits: List[Dict[str, Any]]) -> None:
    if not hits:
        print("No results.")
        return
    print("\n=== Search Results (v3 Enhanced) ===\n")
    for i, h in enumerate(hits, 1):
        src = h.get("_source", {})
        rel = src.get("rel_path") or src.get("path")
        sl, el = src.get("start_line"), src.get("end_line")
        lines_info = f", lines {sl}-{el}" if sl is not None and el is not None else ""
        
        # Enhanced display with symbol info
        primary_sym = src.get("primary_symbol", "")
        primary_kind = src.get("primary_kind", "")
        symbol_info = f" ({primary_kind}: {primary_sym})" if primary_sym else ""
        
        def_symbols = src.get("def_symbols", [])[:3]  # Show first 3
        def_info = f" [defines: {', '.join(def_symbols)}]" if def_symbols else ""
        
        print(f"[{i}] Repo: {src.get('repo_id')}  File: {rel}{symbol_info}")
        print(f"    Score: {h.get('_score'):.3f}  Lang: {src.get('language')}  Chunk: {src.get('chunk_number')}{lines_info}")
        print(f"    ID: {src.get('id')}{def_info}")
        
        # Show doc_head if available
        doc_head = src.get("doc_head", "").strip()
        if doc_head:
            doc_preview = _truncate(doc_head, 100)
            print(f"    Doc: {doc_preview}")
        print()

# -------------------- Main search flow --------------------

def main():
    ap = argparse.ArgumentParser(description="Enhanced Router → Symbol-aware hybrid search → RRF over code_chunks_v3.")
    ap.add_argument("query", help="Natural-language query")
    ap.add_argument("--host", default="http://localhost:9200", help="OpenSearch endpoint")
    ap.add_argument("--chunks-index", default=DEFAULT_CHUNKS_INDEX, help="Chunks index name (v3)")
    ap.add_argument("--router-index", default=DEFAULT_ROUTER_INDEX, help="Router index name (v2)")
    ap.add_argument("--repo-guide-index", default=DEFAULT_REPO_GUIDE_INDEX,
                    help="Index with LLM-enriched repo guides")
    ap.add_argument("--explicit-repo", default=None, help="Skip router and search only this repo_id")
    ap.add_argument("--top-repos", type=int, default=2, help="Router: how many repos to consider")
    ap.add_argument("--bm25-size", type=int, default=200)
    ap.add_argument("--knn-size", type=int, default=200)
    ap.add_argument("--knn-candidates", type=int, default=400)
    ap.add_argument("--final-k", type=int, default=10)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--vector-field", default="vector", help="knn_vector/dense_vector field name in chunks index")
    ap.add_argument("--out", default="retrieval_v3.json", help="Write retrieval bundle here")
    ap.add_argument("--quiet", action="store_true", help="Do not pretty print to stdout")
    args = ap.parse_args()

    client = OpenSearch(hosts=[args.host])
    model = SentenceTransformer(args.model)

    # 1) Router (or explicit repo)
    if args.explicit_repo:
        repo_ids = [args.explicit_repo]
    else:
        rbody = router_query_v2(args.query, args.top_repos)
        rres = client.search(index=args.router_index, body=rbody)
        repo_ids = [h["_source"]["repo_id"] for h in rres["hits"]["hits"]]
        if not repo_ids:
            print("Router found no matching repo. Try --explicit-repo or check router index content.")
            # Still write a minimal bundle for debugging
            bundle = {
                "version": "1.3",
                "created_at": int(time.time()),
                "query": args.query,
                "router": {"router_index": args.router_index, "routed_repo_ids": [], "top_repos": args.top_repos},
                "retrieval": {
                    "chunks_index": args.chunks_index, "bm25_size": args.bm25_size,
                    "knn_size": args.knn_size, "knn_candidates": args.knn_candidates,
                    "final_k": args.final_k, "vector_field": args.vector_field,
                    "embedding_model": args.model
                },
                "hits": []
            }
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(bundle, f, ensure_ascii=False, indent=2)
            return

    # Fetch repo guides (optional enrichment for orientation)
    repo_guides = fetch_repo_guides(client, args.repo_guide_index, repo_ids)
    if len(repo_guides) < len(repo_ids):
        missing = [r for r in repo_ids if r not in {g.get("repo_id") for g in repo_guides}]
        if missing:
            print(f"[warn] No repo_guide found for: {missing}")

    # 2) Embed query
    qvec = model.encode(args.query, normalize_embeddings=True).tolist()

    # 3) Candidate generation (BM25 + kNN), both filtered by repo_ids
    # Enhanced source fields for code_chunks_v3
    source_fields = [
        "id", "repo_id", "language", "path", "rel_path", "ext",
        "chunk_number", "start_line", "end_line", "text", "n_tokens",
        # New v3 symbol fields
        "primary_symbol", "primary_kind", "primary_span", 
        "def_symbols", "symbols", "doc_head"
    ]
    
    bbody = bm25_filtered_query_v3(args.query, repo_ids, args.bm25_size, source_fields)
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

    # 5) Build enhanced LLM bundle and write JSON
    repo_ctx = fetch_router_context(client, args.router_index, repo_ids)

    llm_bundle = build_llm_bundle_v3(
        query=args.query,
        repo_ids=repo_ids,
        fused_hits=fused,
        repo_context=repo_ctx,
        max_lines_per_chunk=120,  # adjust as you like
        repo_guides=repo_guides
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(llm_bundle, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print("Routed repos:", repo_ids)
        pretty_print_v3(fused)
        print(f"Wrote enhanced LLM bundle → {args.out}  (repos: {len(repo_ctx)}, sources: {len(llm_bundle['sources'])})")

if __name__ == "__main__":
    main()
