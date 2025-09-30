#!/usr/bin/env python3
import argparse, json, time
from typing import List, Dict, Any, Optional
import re
from collections import defaultdict
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

DEFAULT_CHUNKS_INDEX = "code_chunks_v2"   # your main chunks index
DEFAULT_ROUTER_INDEX = "repo_router_v1"   # one doc per repo (router)
DEFAULT_REPO_GUIDE_INDEX = "repo_guide_v1"  # LLM-enriched repo guides

# ---- Topic lexicon for intent-guarded retrieval ----
TOPIC_LEX = {
    "auth": [
        "auth","authenticate","authentication","authorization","authorise","authorize",
        "login","logout","password","session","cookie","csrf","jwt","oauth","openid",
        "api key","apikey","basic auth","bearer","token","refresh","idp","sso","rbac","acl"
    ],
    "rate-limit": ["rate limit","ratelimit","throttle","qps","quota","limiter","burst"],
    "cache": ["cache","caching","ttl","lru","memcache","memcached","redis","expiry","expires","expiration"],
    "logging": ["log","logging","logger","trace","debug","info","warn","error","sentry","otel","opentelemetry"],
    # extend as needed…
}

# -------- Intent & path heuristics for "fetch current forecast" --------
INTENTS = {
    "fetch": ["fetch", "get", "retrieve", "request", "call", "query", "load", "pull"],
}

DOMAINS = {
    "forecast": ["forecast", "forecasts", "current", "now", "conditions", "currentcondition", "hourly", "daily", "foreca"],
}

# Paths to promote/demote. Adjust as needed for your repos.
PRIORITIZE_PATHS = ["internal/forecasts/", "adapters/", "clients/", "service/", "proxy/foreca"]
DEPRIORITIZE_PATHS = ["cmd/", "serve/", "server/", "grpc", ".pb.go", ".proto", "main.go", "bootstrap", "init"]

def detect_topic_terms(q: str) -> List[str]:
    ql = q.lower()
    for _, terms in TOPIC_LEX.items():
        if any(t in ql for t in terms):
            return terms
    return []

def passes_relevance_gate(hits: List[Dict[str, Any]], topic_terms: List[str],
                          k_check: int = 8, min_hits: int = 2) -> bool:
    if not topic_terms:
        return True
    cnt = 0
    for h in hits[:k_check]:
        src = h.get("_source", {})
        txt = (src.get("text") or "").lower()
        syms = " ".join(src.get("symbols", [])).lower() if isinstance(src.get("symbols"), list) else str(src.get("symbols") or "").lower()
        if any(t in txt or t in syms for t in topic_terms):
            cnt += 1
    return cnt >= min_hits

# -------------------- Helpers for compact LLM bundle --------------------

def _truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    t = text.strip()
    return t if len(t) <= max_chars else t[:max_chars].rstrip() + "…"


def fetch_router_context(client, router_index: str, repo_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch compact repo context for the routed repos."""
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
            "short_title": _truncate(s.get("short_title", ""), 120),
            "summary": _truncate(s.get("summary", ""), 600),
            "languages": s.get("languages", [])[:3],
            "key_modules": _truncate(s.get("key_modules", ""), 120),
            "key_symbols": _truncate(s.get("key_symbols", ""), 300),
            "tech_stack": _truncate(", ".join(s.get("tech_stack", [])) if isinstance(s.get("tech_stack"), list) else s.get("tech_stack", ""), 160),
            "entrypoints": _truncate(s.get("entrypoints", ""), 160),
            "domains": _truncate(s.get("domains", ""), 160),
        })
    return out


def _clip_by_lines(text: str, max_lines: int = 120) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + "\n# …(trimmed)…"


def build_llm_bundle(query: str, repo_ids, fused_hits, repo_context: List[Dict[str, Any]],
                     max_lines_per_chunk: int = 120,
                     repo_guides: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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
            "code": code
        })
    return {
        "version": "1.2",
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

# -------------------- Query builders --------------------

def _contains_any(hay: str, terms: list[str]) -> bool:
    hay = hay.lower()
    return any(t in hay for t in terms)

def detect_fetch_forecast_intent(q: str) -> bool:
    ql = q.lower()
    return _contains_any(ql, INTENTS["fetch"]) and _contains_any(ql, DOMAINS["forecast"])

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

def bm25_v3_query(query_text: str, repo_ids: List[str], size: int,
                  source_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "size": size,
        "_source": source_fields or True,
        "query": {
            "bool": {
                "filter": [{"terms": {"repo_id": repo_ids}}],
                "should": [{
                    "multi_match": {
                        "query": query_text,
                        "fields": [
                            "primary_symbol^8",
                            "def_symbols^6",
                            "symbols^4",
                            "rel_path^3",
                            "doc_head^2",
                            "text"
                        ],
                        "type": "best_fields",
                        "operator": "or"
                    }
                }]
            }
        }
    }

def bm25_v3_guarded(query_text: str, repo_ids: List[str], size: int,
                    topic_terms: List[str],
                    source_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    must_terms = []
    for t in topic_terms:
        must_terms.append({"match_phrase": {"text": t}})
        must_terms.append({"match_phrase": {"symbols": t}})
    return {
        "size": size,
        "_source": source_fields or True,
        "query": {
            "bool": {
                "filter": [{"terms": {"repo_id": repo_ids}}],
                "must":   [{"bool": {"should": must_terms, "minimum_should_match": 1}}],
                "should": [{
                    "multi_match": {
                        "query": query_text,
                        "fields": [
                            "primary_symbol^8",
                            "def_symbols^6",
                            "symbols^4",
                            "rel_path^3",
                            "doc_head^2",
                            "text"
                        ],
                        "type": "best_fields",
                        "operator": "or"
                    }
                }]
            }
        }
    }

def bm25_fetch_forecast(query_text: str, repo_ids: list[str], size: int,
                        source_fields: list[str] | None = None) -> dict:
    verbs = " ".join(INTENTS["fetch"])
    nouns = " ".join(DOMAINS["forecast"])

    must_clauses = [
        {  # at least one fetch verb must appear
            "bool": {
                "should": [{
                    "multi_match": {
                        "query": verbs,
                        "fields": ["text", "symbols", "rel_path"],
                        "type": "best_fields"
                    }
                }],
                "minimum_should_match": 1
            }
        },
        {  # at least one forecast noun must appear
            "bool": {
                "should": [{
                    "multi_match": {
                        "query": nouns,
                        "fields": ["text", "symbols", "rel_path"],
                        "type": "best_fields"
                    }
                }],
                "minimum_should_match": 1
            }
        }
    ]

    path_shoulds = []
    for p in PRIORITIZE_PATHS:
        path_shoulds.append({"wildcard": {"rel_path": {"value": f"*{p}*", "boost": 3.0}}})
    for p in DEPRIORITIZE_PATHS:
        path_shoulds.append({"wildcard": {"rel_path": {"value": f"*{p}*", "boost": 0.3}}})

    return {
        "size": size,
        "_source": source_fields or True,
        "query": {
            "bool": {
                "filter": [{"terms": {"repo_id": repo_ids}}],
                "must": must_clauses,
                "should": [
                    {   # keep your v3 field boosts to favor definitions/symbols
                        "multi_match": {
                            "query": query_text,
                            "fields": [
                                "primary_symbol^8",
                                "def_symbols^6",
                                "symbols^4",
                                "rel_path^3",
                                "doc_head^2",
                                "text"
                            ],
                            "type": "best_fields",
                            "operator": "or"
                        }
                    },
                    *path_shoulds
                ]
            }
        }
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

def _chunk_text_for_rerank(src: dict, max_chars: int = 2000) -> str:
    # Prefer 'text' (code chunk). Fall back to doc_head if needed.
    t = (src.get("text") or src.get("doc_head") or "").strip()
    if len(t) > max_chars:
        t = t[:max_chars]
    return t

def rerank_with_cross_encoder(query_text: str,
                              fused_hits: list[dict],
                              model_name: str,
                              topn: int,
                              pool_size: int) -> list[dict]:
    """
    Rerank the top 'pool_size' fused hits using a cross-encoder, return top 'topn'.
    Falls back to the original fused order if CrossEncoder is unavailable.
    """
    if CrossEncoder is None:
        # sentence-transformers not available; keep fused ordering
        return fused_hits[:topn]

    pool = fused_hits[:max(1, min(pool_size, len(fused_hits)))]
    # Build (query, text) pairs
    pairs = []
    for h in pool:
        src = h.get("_source", h)
        txt = _chunk_text_for_rerank(src)
        if not txt:
            txt = (src.get("rel_path") or "")  # tiny fallback
        pairs.append((query_text, txt))

    # Load model once per run
    model = CrossEncoder(model_name)

    scores = model.predict(pairs)  # higher = more relevant
    # Attach scores and sort
    for s, h in zip(scores, pool):
        h["_rerank_score"] = float(s)

    pool_sorted = sorted(pool, key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
    return pool_sorted[:topn]

# -------------------- Pretty print --------------------

STOPWORDS = {
    "the","and","or","a","an","to","of","in","for","on","with","by","as","is","it","are","be","was","were",
    "can","you","me","what","about","do","know","tell","how","please","explain","show"
}

def tokenize_meaningful(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[A-Za-z][A-Za-z0-9_]+", (text or "").lower())
        if len(t) >= 3 and t not in STOPWORDS
    }

def _bag_from_source(src: dict) -> str:
    parts = [
        src.get("text") or "",
        " ".join(src.get("symbols") or []) if isinstance(src.get("symbols"), list) else str(src.get("symbols") or ""),
        src.get("rel_path") or "",
        src.get("primary_symbol") or ""
    ]
    return " ".join(parts).lower()

def evidence_overlap_ok(question: str, fused_hits: list[dict], min_distinct: int = 2, top_n: int = 10) -> bool:
    qtok = tokenize_meaningful(question)
    if not qtok:
        return True
    seen: set[str] = set()
    for h in fused_hits[:top_n]:
        src = h.get("_source", h)  # handle both shapes
        bag = _bag_from_source(src)
        for t in qtok:
            if t in bag:
                seen.add(t)
    return len(seen) >= min_distinct

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
        primary_symbol = src.get("primary_symbol")
        symbol_info = f" ({src.get('primary_kind', 'symbol')}: {primary_symbol})" if primary_symbol else ""
        print(f"[{i}] Repo: {src.get('repo_id')}  File: {rel}{symbol_info}  (chunk {src.get('chunk_number')}{lines_info})")
        print(f"    Score: {h.get('_score')}  Lang: {src.get('language')}  Ext: {src.get('ext')}")
        print(f"    ID: {src.get('id')}")
        print()

# -------------------- Main search flow --------------------

def main():
    ap = argparse.ArgumentParser(description="Router → filtered hybrid (BM25 + kNN) → RRF over code chunks.")
    ap.add_argument("query", help="Natural-language query")
    ap.add_argument("--host", default="http://localhost:9200", help="OpenSearch endpoint")
    ap.add_argument("--chunks-index", default=DEFAULT_CHUNKS_INDEX, help="Chunks index name")
    ap.add_argument("--router-index", default=DEFAULT_ROUTER_INDEX, help="Router index name")
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
    ap.add_argument("--out", default="retrieval.json", help="Write retrieval bundle here")
    ap.add_argument("--quiet", action="store_true", help="Do not pretty print to stdout")
    ap.add_argument("--bm25-v3", action="store_true", help="Use boosted BM25 fields for code_chunks_v3")
    ap.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking on candidate pool")
    ap.add_argument("--rerank-size", type=int, default=60, help="Pool size to rerank (top-N from fused candidates)")
    ap.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L6-v2", help="Cross-encoder model name")
    args = ap.parse_args()

    def _looks_like_v3(index_name: str) -> bool:
        return "code_chunks_v3" in (index_name or "")

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
            # Still write a minimal bundle for debugging
            bundle = {
                "version": "1.0",
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
    # make sure source_fields includes v3 fields; harmless if missing
    source_fields = [
        "id","repo_id","language","path","rel_path","module","ext",
        "chunk_number","start_line","end_line","n_tokens","text",
        # v3 metadata (present if index is v3; harmless if missing)
        "primary_symbol","primary_kind","primary_span","def_symbols","symbols","doc_head"
    ]
    
    # Decide which BM25 builder to use
    use_fetch_forecast = detect_fetch_forecast_intent(args.query)

    if use_fetch_forecast:
        bbody = bm25_fetch_forecast(args.query, repo_ids, args.bm25_size, source_fields)
    else:
        # Topic detection for guarded retrieval
        topic_terms = detect_topic_terms(args.query)
        
        if topic_terms:
            bbody = bm25_v3_guarded(args.query, repo_ids, args.bm25_size, topic_terms, source_fields)
        else:
            # Prefer your v3 BM25 if available, else fall back to existing bm25_filtered_query
            try:
                bbody = bm25_v3_query(args.query, repo_ids, args.bm25_size, source_fields)
            except NameError:
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

    # 4) Fuse larger, then optionally rerank to final_k
    # Build a larger fused pool first (e.g., max of bm25_size/knn_size or rerank_size)
    pool_K = max(args.final_k, args.rerank_size, args.bm25_size, args.knn_size)
    fused_pool = rrf_fuse([bhits, khits], K=pool_K)

    # Optional rerank
    if args.rerank:
        final_hits = rerank_with_cross_encoder(
            query_text=args.query,
            fused_hits=fused_pool,
            model_name=args.rerank_model,
            topn=args.final_k,
            pool_size=args.rerank_size
        )
    else:
        final_hits = fused_pool[:args.final_k]

    diagnostics = {
        "router_repos": repo_ids,
        "intent_fetch_forecast": use_fetch_forecast,
        "rerank_enabled": bool(args.rerank),
        "rerank_model": args.rerank_model if args.rerank else "",
        "rerank_pool_size": args.rerank_size if args.rerank else 0
    }

    gate_reason = ""

    # If we used the fetch/forecast intent path, require minimal token overlap in top results
    if use_fetch_forecast and not evidence_overlap_ok(args.query, final_hits, min_distinct=2, top_n=10):
        print("[WARN] Evidence overlap gate failed: top results do not reflect query tokens.")
        gate_reason = "no_topic_evidence"
    else:
        # Gate: if topic terms present but top results don't contain them, mark bundle as no-evidence
        topic_terms = detect_topic_terms(args.query) if not use_fetch_forecast else []
        if topic_terms and not passes_relevance_gate(final_hits, topic_terms, k_check=8, min_hits=2):
            print("[WARN] Relevance gate failed: insufficient topic evidence in top results.")
            gate_reason = "no_topic_evidence"

    # 5) Build compact LLM bundle and write JSON
    # Optional: backfill_missing_texts(client, args.chunks_index, fused, text_field="text")

    repo_ctx = fetch_router_context(client, args.router_index, repo_ids)

    llm_bundle = build_llm_bundle(
        query=args.query,
        repo_ids=repo_ids,
        fused_hits=final_hits,
        repo_context=repo_ctx,
        max_lines_per_chunk=120,  # adjust as you like
        repo_guides=repo_guides
    )

    # Add diagnostics and reason to the bundle
    llm_bundle["diagnostics"] = diagnostics
    llm_bundle["reason"] = gate_reason

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(llm_bundle, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print("Routed repos:", repo_ids)
        pretty_print(final_hits)
        print(f"Wrote LLM bundle → {args.out}  (repos: {len(repo_ctx)}, sources: {len(llm_bundle['sources'])})")

if __name__ == "__main__":
    main()
