#!/usr/bin/env python3
"""
FastAPI backend for RAG Code Search System
Provides REST API endpoints for the code search functionality
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from collections import defaultdict
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import uvicorn
import requests
import json
import textwrap
import re

# Configuration
DEFAULT_CHUNKS_INDEX = "code_chunks_v2"
DEFAULT_CHUNKS_INDEX_V3 = "code_chunks_v3"
DEFAULT_ROUTER_INDEX_V1 = "repo_router_v1"
DEFAULT_ROUTER_INDEX_V2 = "repo_router_v2"
DEFAULT_REPO_GUIDE_INDEX = "repo_guide_v1"
DEFAULT_HOST = "http://localhost:9200"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "qwen2.5-coder:7b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

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
    "database": ["database","db","sql","nosql","postgres","mysql","mongodb","sqlite","query","schema","migration"],
    "api": ["api","endpoint","rest","graphql","route","handler","controller","middleware"],
    "security": ["security","encrypt","decrypt","hash","salt","vulnerability","xss","csrf","injection"],
    "testing": ["test","testing","spec","mock","stub","unit test","integration test","e2e"],
    # extend as needed…
}

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
        
        # Enhanced relevance check: topic terms should appear in meaningful context
        # Avoid false positives from gRPC/infrastructure code
        content = txt + " " + syms
        topic_matches = 0
        for term in topic_terms:
            if term in content:
                # Topic-specific context validation
                meaningful_context = False
                
                # Auth context
                if any(auth_term in term for auth_term in ["auth", "login", "password", "session", "token", "jwt", "oauth"]):
                    if any(keyword in content for keyword in ["handler", "middleware", "login", "password", "session", "jwt", "oauth", "token", "user", "verify", "validate"]):
                        meaningful_context = True
                
                # API context
                elif any(api_term in term for api_term in ["api", "endpoint", "route"]):
                    if any(keyword in content for keyword in ["handler", "router", "controller", "endpoint", "route", "get", "post", "put", "delete"]):
                        meaningful_context = True
                
                # Database context  
                elif any(db_term in term for db_term in ["database", "db", "sql", "query"]):
                    if any(keyword in content for keyword in ["query", "select", "insert", "update", "delete", "table", "schema", "connection"]):
                        meaningful_context = True
                
                # Default context for other topics
                else:
                    meaningful_context = True
                
                if meaningful_context:
                    topic_matches += 1
                    break
        
        if topic_matches > 0:
            cnt += 1
    return cnt >= min_hits

# Initialize FastAPI app
app = FastAPI(
    title="RAG Code Search API",
    description="API for semantic code search using RAG",
    version="1.0.0"
)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for reuse
client = None
model = None

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    explicit_repo: Optional[str] = None
    top_repos: int = 2
    bm25_size: int = 200
    knn_size: int = 200
    knn_candidates: int = 400
    final_k: int = 10
    chunks_index: str = DEFAULT_CHUNKS_INDEX
    router_version: str = "v1"  # "v1" or "v2"
    chunks_version: str = "v2"  # "v2" or "v3"

class QARequest(BaseModel):
    question: str
    explicit_repo: Optional[str] = None
    top_repos: int = 2
    final_k: int = 10
    llm_model: str = DEFAULT_LLM_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    temperature: float = 0.0
    max_lines: int = 120
    spotlight_chunks: int = 3
    router_version: str = "v1"  # "v1" or "v2"
    chunks_version: str = "v2"  # "v2" or "v3"

class SearchResult(BaseModel):
    id: str
    repo_id: str
    language: Optional[str] = None
    path: Optional[str] = None
    rel_path: Optional[str] = None
    module: Optional[str] = None
    ext: Optional[str] = None
    chunk_number: Optional[int] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    symbols: Optional[List[str]] = None
    text: str
    n_tokens: Optional[int] = None
    score: float
    # New v3 symbol fields
    primary_symbol: Optional[str] = None
    primary_kind: Optional[str] = None
    primary_span: Optional[List[int]] = None
    def_symbols: Optional[List[str]] = None
    doc_head: Optional[str] = None

class RepoGuide(BaseModel):
    repo_id: str
    overview: Optional[str] = None
    key_flows: Optional[str] = None
    entrypoints: Optional[str] = None
    languages: List[str] = []
    modules: Optional[str] = None

class RouterInfo(BaseModel):
    repo_id: str
    score: float
    short_title: Optional[str] = None
    summary: Optional[str] = None
    domains: Optional[str] = None
    tech_stack: Optional[str] = None
    languages: List[str] = []
    key_modules: Optional[str] = None
    key_symbols: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    routed_repos: List[str]  # Keep for backward compatibility
    router_info: List[RouterInfo] = []  # Detailed router information
    total_results: int
    query: str
    repo_guides: List[RepoGuide] = []
    message: Optional[str] = None  # For topic guard notifications

class QAResponse(BaseModel):
    question: str
    answer: str
    routed_repos: List[str]
    sources_used: int
    model_used: str

# Helper functions (copied from search_rag.py)
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

    # Last resort: return empty list
    return []

def router_query_v1(query_text: str, topn: int) -> Dict[str, Any]:
    return {
        "size": topn,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["short_title^3", "summary^2", "domains^2", "key_modules", "key_symbols", "tech_stack"]
            }
        }
    }

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

def router_query(query_text: str, topn: int, router_version: str = "v1") -> Dict[str, Any]:
    """Router query dispatcher based on version"""
    if router_version == "v2":
        return router_query_v2(query_text, topn)
    else:
        return router_query_v1(query_text, topn)

def bm25_filtered_query_v2(query_text: str, repo_ids: List[str], size: int,
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

def bm25_v3_guarded(query_text: str, repo_ids: List[str], size: int,
                    topic_terms: List[str],
                    source_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Topic-constrained BM25 query that requires chunks to contain topic terms"""
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

def bm25_filtered_query(query_text: str, repo_ids: List[str], size: int,
                        source_fields: Optional[List[str]] = None, chunks_version: str = "v2") -> Dict[str, Any]:
    """BM25 query dispatcher based on chunks version"""
    if chunks_version == "v3":
        return bm25_filtered_query_v3(query_text, repo_ids, size, source_fields)
    else:
        return bm25_filtered_query_v2(query_text, repo_ids, size, source_fields)

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

# Q&A Helper Functions
def _clip_lines(text: str, max_lines: int) -> str:
    """Clip text to max_lines, adding truncation notice if needed"""
    if not text: return ""
    lines = text.splitlines()
    if len(lines) <= max_lines: return text
    return "\n".join(lines[:max_lines]) + "\n# …(trimmed)…"

def _ext_from_path(path: str) -> str:
    """Get file extension from path"""
    if not path or "." not in path: return ""
    return path.rsplit(".", 1)[-1].lower()

def build_qa_prompt(question: str, search_results: List[Dict[str, Any]], repo_guides: List[Dict[str, Any]] = None, max_lines: int = 120, spotlight_count: int = 3, topic_terms: List[str] = None) -> str:
    """Build prompt for Q&A with repository context and code sources"""
    
    # Build repository context section
    repo_context = ""
    if repo_guides:
        repo_context = "\n## REPOSITORY CONTEXT:\n"
        for guide in repo_guides:
            repo_id = guide.get("repo_id", "")
            if repo_id:
                repo_context += f"\n### {repo_id}:\n"
                
                if guide.get("overview"):
                    repo_context += f"**Overview**: {guide['overview']}\n\n"
                
                if guide.get("key_flows"):
                    repo_context += f"**Key Flows**: {guide['key_flows']}\n\n"
                
                if guide.get("entrypoints"):
                    repo_context += f"**Entrypoints**: {guide['entrypoints']}\n\n"
                
                if guide.get("languages"):
                    langs = guide["languages"] if isinstance(guide["languages"], list) else [guide["languages"]]
                    repo_context += f"**Languages**: {', '.join(langs)}\n\n"
                
                if guide.get("modules"):
                    repo_context += f"**Key Modules**: {guide['modules']}\n\n"
    
    system_context = f"""You are a READ-ONLY code assistant for professional developers.
Your job is to EXPLAIN, LOCATE, CROSS-REFERENCE, and QUOTE code from the provided sources.
You MUST NOT propose code edits, write new code, or modify existing code.

Evidence policy:
- Use ONLY the provided code sources for evidence.
- If the answer is not in them, say: "Not found in provided sources."
- Every substantive bullet/paragraph MUST end with at least one citation tag: [repo_id|rel_path:start-end].

Style (developer-grade):
- Be comprehensive and precise. Prefer depth over brevity.
- Use tight technical language, avoid fluff. Use bullets and short paragraphs.
{repo_context}
"""

    # Build sources list
    sources_header = []
    for i, result in enumerate(search_results, 1):
        src = result.get("_source", result)  # Handle both formats
        sl = src.get("start_line", "?")
        el = src.get("end_line", "?")
        sources_header.append(f"[#{i}] {src.get('repo_id')} | {src.get('rel_path')} | lines {sl}-{el}")
    
    sources_list = "\n".join(sources_header)
    
    # Build code blocks
    code_blocks = []
    for i, result in enumerate(search_results, 1):
        src = result.get("_source", result)
        sl = src.get("start_line", "?")
        el = src.get("end_line", "?")
        header = f"===== Source #{i}: {src.get('repo_id')} | {src.get('rel_path')} | lines {sl}-{el}"
        code = _clip_lines(src.get("text", ""), max_lines)
        lang = _ext_from_path(src.get("rel_path") or "")
        
        if lang:
            block = f"{header}\n```{lang}\n{code}\n```"
        else:
            block = f"{header}\n```\n{code}\n```"
        code_blocks.append(block)
    
    # Example format (few-shot learning) 
    example_format = """
Answer Structure (MANDATORY):
- Organize the answer into these sections (skip a section if not applicable):
  1) Summary
  2) Relevant files & roles
  3) How it works (step-by-step)
  4) Key functions/classes (name, purpose, location)
  5) **Spotlight** - Show the most critical code excerpts (≤15 lines each) with citations
  6) Caveats / edge cases
  7) Pointers for further reading (paths)

Example style:

Summary
- The HTTP server is started by the CLI entrypoint and wired to route handlers. [weather-api|src/cli/main.py:12-40]

Relevant files & roles
- src/cli/main.py — starts the server and mounts routes. [weather-api|src/cli/main.py:1-80]
- src/routes/status.py — health endpoint. [weather-api|src/routes/status.py:1-60]

How it works
- The CLI parses flags, builds a config, then calls create_app(). [weather-api|src/cli/main.py:12-40]
- create_app() registers /status and /metrics handlers. [weather-api|src/app.py:35-78]

Spotlight
```python
def create_app(config: Config) -> FastAPI:
    app = FastAPI()
    app.include_router(status.router)
    return app
```
[weather-api|src/app.py:35-45]
"""

    # Topic check section
    topic_section = ""
    if topic_terms:
        topic_section = f"""

Topic Check:
- Asked topic: contains any of: {', '.join(topic_terms)}
- If none of the snippets include these terms, reply exactly: Not found in provided sources.
"""

    # Combine everything
    prompt = f"""Question: {question}

{system_context}

{example_format}

Sources:
{sources_list}

{chr(10).join(code_blocks)}
{topic_section}

Please answer the question using the provided code sources. Follow the answer structure above, including a mandatory Spotlight section with {spotlight_count} most critical code blocks (≤15 lines each) that demonstrate the key concepts. Include citations in the format [repo_id|rel_path:start-end] for your evidence."""

    return prompt

def reorder_sources_for_quality(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Heuristic: implementation files before tests, similar to answer.py"""
    def is_test(s):
        src = s.get("_source", s)
        p = (src.get("rel_path") or src.get("path") or "").lower()
        return "tests" in p or "/test" in p or p.startswith("test_") or p.endswith("_test.py")
    
    impl = [s for s in sources if not is_test(s)]
    tests = [s for s in sources if is_test(s)]
    # Keep original ordering inside buckets (already by retrieval score)
    return impl + tests

def pick_spotlight_sources(sources: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Pick top-k sources by score with basic diversity (prefer distinct rel_path).
    Assumes sources roughly sorted by descending score from retrieval.
    """
    seen_paths = set()
    selected = []
    for s in sources:
        src = s.get("_source", s)
        path = src.get("rel_path") or src.get("path")
        if path in seen_paths:
            continue
        selected.append(s)
        seen_paths.add(path)
        if len(selected) >= k:
            break
    return selected

def add_auto_spotlight_fallback(answer: str, sources: List[Dict[str, Any]], spotlight_count: int = 3) -> str:
    """Add automatic spotlight section if the model didn't include one"""
    if "Spotlight" in answer or "spotlight" in answer:
        return answer  # Already has spotlight section
    
    # Get the best sources for spotlight
    ordered_sources = reorder_sources_for_quality(sources)
    spotlight_sources = pick_spotlight_sources(ordered_sources, spotlight_count)
    
    if not spotlight_sources:
        return answer
    
    spotlight_section = "\n\n## Spotlight\n\n"
    for s in spotlight_sources:
        src = s.get("_source", s)
        sl = src.get("start_line", "?")
        el = src.get("end_line", "?")
        lang = _ext_from_path(src.get("rel_path") or "")
        code = _clip_lines(src.get("text", ""), max_lines=15)
        
        spotlight_section += f"**{src.get('repo_id')} | {src.get('rel_path')} | lines {sl}-{el}**\n"
        if lang:
            spotlight_section += f"```{lang}\n{code}\n```\n"
        else:
            spotlight_section += f"```\n{code}\n```\n"
        spotlight_section += f"[{src.get('repo_id')}|{src.get('rel_path')}:{sl}-{el}]\n\n"
    
    return answer + spotlight_section

def get_router_index(version: str) -> str:
    """Get the appropriate router index based on version"""
    if version == "v2":
        return DEFAULT_ROUTER_INDEX_V2
    else:
        return DEFAULT_ROUTER_INDEX_V1

def get_chunks_index(version: str) -> str:
    """Get the appropriate chunks index based on version"""
    if version == "v3":
        return DEFAULT_CHUNKS_INDEX_V3
    else:
        return DEFAULT_CHUNKS_INDEX

def fetch_repo_guides(client: OpenSearch, repo_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch repo guide information for the routed repositories.
    Returns a list of repo guide dictionaries with overview, key_flows, etc.
    """
    if not repo_ids:
        return []
    
    try:
        print(f"DEBUG: Fetching repo guides for: {repo_ids}")
        
        # Try mget first
        docs = [{"_index": DEFAULT_REPO_GUIDE_INDEX, "_id": rid} for rid in repo_ids]
        res = client.mget(body={"docs": docs})
        
        guides = []
        for d in res.get("docs", []):
            if not d.get("found"):
                print(f"DEBUG: Repo guide not found for: {d.get('_id')}")
                continue
            s = d.get("_source", {})
            guides.append({
                "repo_id": s.get("repo_id"),
                "overview": s.get("overview", ""),
                "key_flows": s.get("key_flows", ""),
                "entrypoints": s.get("entrypoints", ""),
                "languages": s.get("languages", []),
                "modules": s.get("modules", ""),
            })
        
        # If mget didn't work, try search instead
        if not guides:
            print(f"DEBUG: mget failed, trying search for repo guides")
            search_body = {
                "query": {
                    "terms": {
                        "repo_id": repo_ids
                    }
                },
                "size": len(repo_ids)
            }
            res = client.search(index=DEFAULT_REPO_GUIDE_INDEX, body=search_body)
            
            for hit in res.get("hits", {}).get("hits", []):
                s = hit.get("_source", {})
                guides.append({
                    "repo_id": s.get("repo_id"),
                    "overview": s.get("overview", ""),
                    "key_flows": s.get("key_flows", ""),
                    "entrypoints": s.get("entrypoints", ""),
                    "languages": s.get("languages", []),
                    "modules": s.get("modules", ""),
                })
        
        print(f"DEBUG: Found {len(guides)} repo guides")
        return guides
    except Exception as e:
        print(f"ERROR: Could not fetch repo guides: {e}")
        import traceback
        traceback.print_exc()
        return []

def call_ollama_chat(model: str, prompt: str, ollama_url: str = DEFAULT_OLLAMA_URL, 
                    temperature: float = 0.0, timeout: int = 300) -> str:
    """Call Ollama API for chat completion"""
    try:
        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "options": {"temperature": temperature},
            "stream": False
        }
        
        response = requests.post(ollama_url, json=body, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        return (data.get("message") or {}).get("content", "")
        
    except requests.RequestException as e:
        return f"Error calling Ollama: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize OpenSearch client and embedding model on startup"""
    global client, model
    try:
        print("🔄 Connecting to OpenSearch...")
        client = OpenSearch(hosts=[DEFAULT_HOST])
        
        # Test connection
        health = client.cluster.health()
        print(f"✅ Connected to OpenSearch at {DEFAULT_HOST}")
        
        print("🔄 Loading embedding model...")
        model = SentenceTransformer(DEFAULT_MODEL)
        print(f"✅ Loaded embedding model: {DEFAULT_MODEL}")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise - let the server start anyway
        # raise

@app.get("/", response_class=HTMLResponse)
async def root():
    """API documentation and health check"""
    return """
    <html>
        <head><title>RAG Code Search & Q&A API</title></head>
        <body>
            <h1>RAG Code Search & Q&A API</h1>
            <p>🚀 API is running!</p>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                <li><a href="/health">/health</a> - Health check</li>
                <li><code>POST /search</code> - Search for code chunks</li>
                <li><code>POST /qa</code> - Ask questions and get LLM answers</li>
                <li><code>GET /repositories</code> - List available repositories</li>
                <li><code>GET /qa/simple?q=question</code> - Quick Q&A test</li>
            </ul>
            <p><strong>New:</strong> Q&A functionality powered by Ollama LLM!</p>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test OpenSearch connection
        client.cluster.health()
        return {
            "status": "healthy",
            "opensearch": "connected",
            "model": "loaded",
            "timestamp": "2024-01-01T00:00:00Z"  # Could use datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.post("/search", response_model=SearchResponse)
async def search_code(request: SearchRequest):
    """
    Main search endpoint for RAG-based code search
    
    Uses router → filtered hybrid (BM25 + kNN) → RRF pipeline
    """
    if not client or not model:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    try:
        # 1) Router (or explicit repo)
        router_index = get_router_index(request.router_version)
        chunks_index = get_chunks_index(request.chunks_version)
        print(f"DEBUG: Using router index: {router_index}, chunks index: {chunks_index}")
        
        if request.explicit_repo:
            repo_ids = [request.explicit_repo]
            router_info = []  # No router info for explicit repos
        else:
            rbody = router_query(request.query, request.top_repos, request.router_version)
            rres = client.search(index=router_index, body=rbody)
            router_hits = rres["hits"]["hits"]
            
            if not router_hits:
                return SearchResponse(
                    results=[],
                    routed_repos=[],
                    router_info=[],
                    total_results=0,
                    query=request.query,
                    repo_guides=[]
                )
            
            repo_ids = [h["_source"]["repo_id"] for h in router_hits]
            
            # Extract detailed router information
            router_info = []
            for hit in router_hits:
                src = hit["_source"]
                # Handle differences between v1 and v2 router schemas
                tech_stack = src.get("tech_stack")
                if isinstance(tech_stack, list):
                    # v2 format - join array to string
                    tech_stack = ", ".join(tech_stack)
                
                modules = src.get("key_modules") or src.get("modules")  # v1 uses key_modules, v2 uses modules
                if isinstance(modules, list):
                    # v2 format - join array to string
                    modules = ", ".join(modules)
                
                router_info.append(RouterInfo(
                    repo_id=src.get("repo_id", ""),
                    score=hit["_score"],
                    short_title=src.get("short_title"),
                    summary=src.get("summary"),
                    domains=src.get("domains"),  # Only in v1
                    tech_stack=tech_stack,
                    languages=src.get("languages", []),
                    key_modules=modules,
                    key_symbols=src.get("key_symbols")
                ))

        # 2) Embed query
        qvec = model.encode(request.query, normalize_embeddings=True).tolist()

        # 3) Candidate generation (BM25 + kNN), both filtered by repo_ids
        # Enhanced source fields for code_chunks_v3
        if request.chunks_version == "v3":
            source_fields = [
                "id", "repo_id", "language", "path", "rel_path", "ext",
                "chunk_number", "start_line", "end_line", "text", "n_tokens",
                # New v3 symbol fields
                "primary_symbol", "primary_kind", "primary_span", 
                "def_symbols", "symbols", "doc_head"
            ]
        else:
            source_fields = [
                "id","repo_id","language","path","rel_path","module","ext",
                "chunk_number","start_line","end_line","symbols","text","n_tokens"
            ]
        
        # Topic detection for guarded retrieval
        topic_terms = detect_topic_terms(request.query)
        
        if topic_terms:
            # Use guarded BM25 query when topic terms are detected
            bbody = bm25_v3_guarded(request.query, repo_ids, request.bm25_size, topic_terms, source_fields)
            print(f"DEBUG: 🛡️ Topic guard ACTIVATED for query '{request.query}' - constraining to terms: {topic_terms[:5]}...")
        else:
            bbody = bm25_filtered_query(request.query, repo_ids, request.bm25_size, source_fields, request.chunks_version)
        
        bhits = client.search(index=chunks_index, body=bbody)["hits"]["hits"]

        khits = knn_search_filtered(
            client=client,
            index=chunks_index,
            field="vector",  # Default vector field name
            qvec=qvec,
            repo_ids=repo_ids,
            k=request.knn_size,
            num_candidates=request.knn_candidates,
            source_fields=source_fields
        )

        # 4) Fuse with RRF and take final_k
        fused = rrf_fuse([bhits, khits], K=request.final_k)

        # Relevance gate: check if topic terms are sufficiently represented in results
        if topic_terms and not passes_relevance_gate(fused, topic_terms, k_check=8, min_hits=2):
            print(f"DEBUG: 🚫 Relevance gate BLOCKED query '{request.query}' - insufficient evidence for topic terms: {topic_terms[:3]}...")
            return SearchResponse(
                results=[],
                routed_repos=repo_ids,
                router_info=router_info if not request.explicit_repo else [],
                total_results=0,
                query=request.query,
                repo_guides=[],
                message="Not found in provided sources - insufficient topic evidence in search results."
            )

        # 5) Fetch repo guides for context
        repo_guides_data = fetch_repo_guides(client, repo_ids)
        repo_guides = []
        for guide_data in repo_guides_data:
            repo_guide = RepoGuide(
                repo_id=guide_data.get("repo_id", ""),
                overview=guide_data.get("overview"),
                key_flows=guide_data.get("key_flows"),
                entrypoints=guide_data.get("entrypoints"),
                languages=guide_data.get("languages", []),
                modules=guide_data.get("modules")
            )
            repo_guides.append(repo_guide)

        # 6) Convert to response format
        results = []
        for hit in fused:
            src = hit.get("_source", {})
            result = SearchResult(
                id=src.get("id", ""),
                repo_id=src.get("repo_id", ""),
                language=src.get("language"),
                path=src.get("path"),
                rel_path=src.get("rel_path"),
                module=src.get("module"),
                ext=src.get("ext"),
                chunk_number=src.get("chunk_number"),
                start_line=src.get("start_line"),
                end_line=src.get("end_line"),
                symbols=src.get("symbols"),
                text=src.get("text", ""),
                n_tokens=src.get("n_tokens"),
                score=hit.get("_score", 0.0),
                # New v3 symbol fields (only populated if available)
                primary_symbol=src.get("primary_symbol"),
                primary_kind=src.get("primary_kind"),
                primary_span=src.get("primary_span"),
                def_symbols=src.get("def_symbols"),
                doc_head=src.get("doc_head")
            )
            results.append(result)

        return SearchResponse(
            results=results,
            routed_repos=repo_ids,
            router_info=router_info,
            total_results=len(results),
            query=request.query,
            repo_guides=repo_guides
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

@app.get("/repositories")
async def list_repositories(router_version: str = "v1"):
    """List available repositories in the specified router index"""
    try:
        router_index = get_router_index(router_version)
        body = {
            "size": 100,
            "query": {"match_all": {}},
            "_source": ["repo_id", "short_title", "summary", "domains", "tech_stack"]
        }
        res = client.search(index=router_index, body=body)
        repos = []
        for hit in res["hits"]["hits"]:
            src = hit["_source"]
            repos.append({
                "repo_id": src.get("repo_id"),
                "title": src.get("short_title"),
                "summary": src.get("summary"),
                "domains": src.get("domains"),
                "tech_stack": src.get("tech_stack")
            })
        return {"repositories": repos, "total": len(repos), "router_version": router_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch repositories from {router_version}: {e}")

# Simple search endpoint for testing
@app.get("/search/simple")
async def simple_search(
    q: str = Query(..., description="Search query"),
    k: int = Query(10, description="Number of results"),
    repo: Optional[str] = Query(None, description="Specific repository ID"),
    router_version: str = Query("v1", description="Router version (v1 or v2)"),
    chunks_version: str = Query("v2", description="Chunks version (v2 or v3)")
):
    """Simplified search endpoint for quick testing"""
    request = SearchRequest(
        query=q,
        final_k=k,
        explicit_repo=repo,
        router_version=router_version,
        chunks_version=chunks_version
    )
    return await search_code(request)

@app.post("/qa", response_model=QAResponse)
async def ask_question(request: QARequest):
    """
    Q&A endpoint: Retrieve relevant code and generate LLM answer
    
    Combines RAG search with Ollama LLM to answer questions about your codebase
    """
    if not client or not model:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    try:
        # 1) Router (or explicit repo) - same as search
        router_index = get_router_index(request.router_version)
        chunks_index = get_chunks_index(request.chunks_version)
        print(f"DEBUG: Q&A using router index: {router_index}, chunks index: {chunks_index}")
        
        if request.explicit_repo:
            repo_ids = [request.explicit_repo]
        else:
            rbody = router_query(request.question, request.top_repos, request.router_version)
            rres = client.search(index=router_index, body=rbody)
            repo_ids = [h["_source"]["repo_id"] for h in rres["hits"]["hits"]]
            if not repo_ids:
                return QAResponse(
                    question=request.question,
                    answer="No relevant repositories found for your question. Please check if your repositories are indexed.",
                    routed_repos=[],
                    sources_used=0,
                    model_used=request.llm_model
                )

        # 2) Embed query
        qvec = model.encode(request.question, normalize_embeddings=True).tolist()

        # 3) Candidate generation (BM25 + kNN), both filtered by repo_ids
        if request.chunks_version == "v3":
            source_fields = [
                "id", "repo_id", "language", "path", "rel_path", "ext",
                "chunk_number", "start_line", "end_line", "text", "n_tokens",
                # New v3 symbol fields
                "primary_symbol", "primary_kind", "primary_span", 
                "def_symbols", "symbols", "doc_head"
            ]
        else:
            source_fields = [
                "id","repo_id","language","path","rel_path","module","ext",
                "chunk_number","start_line","end_line","symbols","text","n_tokens"
            ]
        
        # Topic detection for guarded retrieval in Q&A
        topic_terms = detect_topic_terms(request.question)
        
        if topic_terms:
            # Use guarded BM25 query when topic terms are detected
            bbody = bm25_v3_guarded(request.question, repo_ids, 200, topic_terms, source_fields)
            print(f"DEBUG: 🛡️ Q&A topic guard ACTIVATED for question '{request.question}' - constraining to terms: {topic_terms[:5]}...")
        else:
            bbody = bm25_filtered_query(request.question, repo_ids, 200, source_fields, request.chunks_version)
        
        bhits = client.search(index=chunks_index, body=bbody)["hits"]["hits"]

        khits = knn_search_filtered(
            client=client,
            index=chunks_index,
            field="vector",  # Default vector field name
            qvec=qvec,
            repo_ids=repo_ids,
            k=200,
            num_candidates=400,
            source_fields=source_fields
        )

        # 4) Fuse with RRF and take final_k
        fused = rrf_fuse([bhits, khits], K=request.final_k)
        
        # Relevance gate: check if topic terms are sufficiently represented in Q&A results
        if topic_terms and not passes_relevance_gate(fused, topic_terms, k_check=8, min_hits=2):
            print(f"DEBUG: 🚫 Q&A relevance gate BLOCKED question '{request.question}' - insufficient evidence for topic terms: {topic_terms[:3]}...")
            return QAResponse(
                question=request.question,
                answer="Not found in provided sources.",
                routed_repos=repo_ids,
                sources_used=0,
                model_used=request.llm_model
            )
        
        if not fused:
            return QAResponse(
                question=request.question,
                answer="No relevant code found for your question. Try rephrasing or check if your code is indexed.",
                routed_repos=repo_ids,
                sources_used=0,
                model_used=request.llm_model
            )

        # 5) Fetch repo guides for context
        repo_guides_data = fetch_repo_guides(client, repo_ids)
        print(f"DEBUG: Q&A fetched {len(repo_guides_data)} repo guides")

        # 6) Reorder sources for better quality (implementation files before tests)
        reordered_sources = reorder_sources_for_quality(fused)

        # 7) Build prompt with repo guides context and call Ollama with spotlight count
        spotlight_count = min(request.spotlight_chunks, len(reordered_sources))
        prompt = build_qa_prompt(
            question=request.question, 
            search_results=reordered_sources, 
            repo_guides=repo_guides_data,
            max_lines=request.max_lines,
            spotlight_count=spotlight_count,
            topic_terms=topic_terms
        )
        answer = call_ollama_chat(
            model=request.llm_model,
            prompt=prompt,
            ollama_url=request.ollama_url,
            temperature=request.temperature
        )

        # 7) Add automatic spotlight fallback if model didn't include one
        answer = add_auto_spotlight_fallback(answer, reordered_sources, spotlight_count)

        return QAResponse(
            question=request.question,
            answer=answer,
            routed_repos=repo_ids,
            sources_used=len(fused),
            model_used=request.llm_model
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A failed: {e}")

# Simple Q&A endpoint for testing
@app.get("/qa/simple")
async def simple_qa(
    q: str = Query(..., description="Question to ask"),
    repo: Optional[str] = Query(None, description="Specific repository ID"),
    model: str = Query(DEFAULT_LLM_MODEL, description="Ollama model to use"),
    router_version: str = Query("v1", description="Router version (v1 or v2)"),
    chunks_version: str = Query("v2", description="Chunks version (v2 or v3)")
):
    """Simplified Q&A endpoint for quick testing"""
    request = QARequest(
        question=q,
        explicit_repo=repo,
        llm_model=model,
        router_version=router_version,
        chunks_version=chunks_version
    )
    return await ask_question(request)

if __name__ == "__main__":
    print("🚀 Starting RAG Code Search & Q&A API...")
    print(f"📝 API docs will be available at: http://localhost:8000/docs")
    print(f"🔍 Simple search test: http://localhost:8000/search/simple?q=authentication")
    print(f"🤖 Simple Q&A test: http://localhost:8000/qa/simple?q=how%20does%20this%20work")
    print(f"🌐 Frontend: Open frontend/index.html and frontend/qa.html")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
