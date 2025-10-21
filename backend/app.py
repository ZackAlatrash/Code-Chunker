#!/usr/bin/env python3
"""
Lean FastAPI backend for RAG Code Search System
Uses modular services for clean separation of concerns
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import uvicorn

# Import our modular services
from services import SearchService, LLMService, SearchV4Service, RAGv4Service

# Configuration
DEFAULT_CHUNKS_INDEX = "code_chunks_v2"
DEFAULT_CHUNKS_INDEX_V3 = "code_chunks_v3"
DEFAULT_CHUNKS_INDEX_V4 = "code_chunks_v4"
DEFAULT_ROUTER_INDEX_V1 = "repo_router_v1"
DEFAULT_ROUTER_INDEX_V2 = "repo_router_v2"
DEFAULT_REPO_GUIDE_INDEX = "repo_guide_v1"
DEFAULT_HOST = "http://localhost:9200"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "qwen2.5-coder:7b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

# Initialize FastAPI app
app = FastAPI(
    title="RAG Code Search API",
    description="Modular API for semantic code search using RAG",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
search_service: Optional[SearchService] = None
search_v4_service: Optional[SearchV4Service] = None
llm_service: Optional[LLMService] = None
rag_v4_service: Optional[RAGv4Service] = None

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
    chunks_version: str = "v2"  # "v2", "v3", or "v4"
    rerank: bool = False  # Enable cross-encoder reranking (legacy, deprecated)
    rerank_size: int = 60  # Pool size to rerank
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"  # Cross-encoder model name
    use_filter_rerank: bool = False  # Enable LLM filter + hosted reranking (new system)
    retriever_index: Optional[str] = None  # Override retriever index (e.g., code_chunks_v5_qodo)
    semantic_only: bool = False  # If true, disable BM25 and use kNN-only

class QARequest(BaseModel):
    question: str
    explicit_repo: Optional[str] = None
    top_repos: int = 2
    final_k: int = 8
    max_lines: int = 120
    spotlight_chunks: int = 3
    llm_model: str = DEFAULT_LLM_MODEL
    temperature: float = 0.0
    ollama_url: str = DEFAULT_OLLAMA_URL
    router_version: str = "v1"
    chunks_version: str = "v2"  # "v2", "v3", or "v4"
    rag_mode: str = "legacy"  # "legacy" or "v4" (Evidence-based with citations)
    use_filter_rerank: bool = False  # Enable local LLM filter + reranking (new)
    rerank: bool = False  # Enable cross-encoder reranking (legacy, deprecated)
    rerank_size: int = 60  # Pool size to rerank
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"  # Cross-encoder model name
    include_repo_context: bool = False  # Include repo guides for orientation (NEVER cited)
    debug: bool = False  # Enable debug recording
    retriever_index: Optional[str] = None  # Override retriever index (e.g., code_chunks_v5_qodo)

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
    # Filter-rerank fields
    filter_score: Optional[int] = None
    rerank_score: Optional[float] = None
    filter_why: Optional[str] = None

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
    routed_repos: List[str]
    router_info: List[RouterInfo] = []
    total_results: int
    query: str
    repo_guides: List[RepoGuide] = []
    message: Optional[str] = None

class DebugInfo(BaseModel):
    llm_prompt: str
    raw_llm_response: str
    topic_terms: List[str] = []
    search_results_count: int
    router_repos: List[str] = []
    bundle_sources: int
    processing_time_ms: Optional[float] = None

class QAResponse(BaseModel):
    question: str
    answer: str
    routed_repos: List[str]
    sources_used: int
    model_used: str
    debug_info: Optional[DebugInfo] = None

# Helper functions
def get_router_index(version: str) -> str:
    """Get the appropriate router index based on version"""
    if version == "v2":
        return DEFAULT_ROUTER_INDEX_V2
    else:
        return DEFAULT_ROUTER_INDEX_V1

def get_chunks_index(version: str) -> str:
    """Get the appropriate chunks index based on version"""
    if version == "v4":
        return DEFAULT_CHUNKS_INDEX_V4
    elif version == "v3":
        return DEFAULT_CHUNKS_INDEX_V3
    else:
        return DEFAULT_CHUNKS_INDEX

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global search_service, search_v4_service, llm_service, rag_v4_service
    try:
        print("🚀 Starting Modular RAG Code Search & Q&A API...")
        print("🔄 Connecting to OpenSearch...")
        client = OpenSearch(hosts=[DEFAULT_HOST])
        
        # Test connection
        health = client.cluster.health()
        print(f"✅ Connected to OpenSearch at {DEFAULT_HOST}")
        
        print("🔄 Loading embedding model...")
        model = SentenceTransformer(DEFAULT_MODEL)
        print(f"✅ Loaded embedding model: {DEFAULT_MODEL}")
        
        # Initialize services
        search_service = SearchService(client, model)
        search_v4_service = SearchV4Service(client)
        llm_service = LLMService(DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL)
        rag_v4_service = RAGv4Service(DEFAULT_OLLAMA_URL, DEFAULT_LLM_MODEL)
        
        print("✅ Services initialized successfully")
        print(f"📝 API docs available at: http://localhost:8000/docs")
        print(f"🌐 Frontend: Open frontend/index.html and frontend/qa.html")
        
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "opensearch": "connected" if search_service else "disconnected",
            "model": "loaded" if search_service else "not loaded",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.get("/repositories")
async def list_repositories(router_version: str = Query("v1")):
    """List available repositories"""
    if not search_service:
        raise HTTPException(status_code=500, detail="Search service not initialized")
    
    try:
        router_index = get_router_index(router_version)
        repositories = search_service.list_repositories(router_index)
        return {"repositories": repositories, "router_version": router_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list repositories: {e}")

@app.post("/search", response_model=SearchResponse)
async def search_code(request: SearchRequest):
    """
    Main search endpoint for RAG-based code search
    Uses modular services for clean separation of concerns
    """
    if not search_service:
        raise HTTPException(status_code=500, detail="Search service not initialized")
    
    try:
        # 1) Router: find relevant repositories
        router_index = get_router_index(request.router_version)
        chunks_index = get_chunks_index(request.chunks_version)
        
        if request.explicit_repo:
            repo_ids = [request.explicit_repo]
            router_info = []
        else:
            # Use router to find relevant repos
            router_response = search_service.execute_router_search(request.query, router_index, request.top_repos)
            # Extract router hits and convert to repo_ids
            router_hits = router_response.get("hits", {}).get("hits", [])
            repo_ids = [h["_source"]["repo_id"] for h in router_hits if "_source" in h and "repo_id" in h["_source"]]
            
            # Convert router hits to RouterInfo
            router_info = []
            for hit in router_hits:
                src = hit.get("_source", {})
                router_info.append(RouterInfo(
                    repo_id=src.get("repo_id", ""),
                    score=hit.get("_score", 0.0),
                    short_title=src.get("short_title"),
                    summary=src.get("summary"),
                    domains=src.get("domains"),
                    tech_stack=", ".join(src.get("tech_stack", [])) if isinstance(src.get("tech_stack"), list) else src.get("tech_stack"),
                    languages=src.get("languages", []),
                    key_modules=src.get("key_modules"),
                    key_symbols=src.get("key_symbols")
                ))

        if not repo_ids:
            return SearchResponse(
                results=[],
                routed_repos=[],
                router_info=[],
                total_results=0,
                query=request.query,
                repo_guides=[],
                message="No repositories found for query"
            )
        
        # Use search_v4 if chunks_version is "v4"
        if request.chunks_version == "v4":
            if not search_v4_service:
                raise HTTPException(status_code=500, detail="Search V4 service not initialized")
            
            # Log retriever index selection
            print(f"[Search] Using retriever_index: {request.retriever_index or 'default (code_chunks_v4)'}")
            
            # Apply LLM filter + rerank if enabled
            if request.use_filter_rerank:
                try:
                    from rag_filter_rerank import FilterRerankPipeline, get_settings as get_fr_settings
                    from rag_filter_rerank.cache import get_cache
                    from rag_filter_rerank.filter_engine import FilterEngine
                    from rag_filter_rerank.reranker import Reranker, build_items_for_rerank
                    from rag_filter_rerank.schemas import Chunk, FilterResult
                    
                    # Get initial candidates (more than final_k for filtering)
                    recall_n = 100  # Get 100 candidates for filtering
                    candidates = search_v4_service.simple_search(
                        request.query,
                        repo_ids,
                        final_k=recall_n,
                        fetch_all_texts=True,
                        retriever_index=request.retriever_index
                    )
                    
                    # Convert to Chunk objects
                    chunks = []
                    for hit in candidates:
                        # Extract text from _source if present (OpenSearch format)
                        text = hit.get("text", "")
                        if not text and "_source" in hit:
                            text = hit["_source"].get("text", "")
                        
                        # Helper to get field from either top-level or _source
                        def get_field(key, default=""):
                            val = hit.get(key)
                            if val is None and "_source" in hit:
                                val = hit["_source"].get(key)
                            return val if val is not None else default
                        
                        chunks.append(Chunk(
                            id=get_field("id"),
                            repo_id=get_field("repo_id"),
                            rel_path=get_field("rel_path"),
                            start_line=int(get_field("start_line", 0)),
                            end_line=int(get_field("end_line", 0)),
                            language=get_field("language"),
                            text=text,
                            summary_en=get_field("summary_en", None),  # Allow None for optional field
                            chunk_hash=get_field("chunk_hash", None),  # Allow None for optional field
                            retriever_score=hit.get("_score", 0.0)
                        ))
                    
                    # Stage 1: LLM Filter
                    fr_settings = get_fr_settings()
                    print(f"[Filter-Rerank] Using model: {fr_settings.OLLAMA_MODEL}, timeout: {fr_settings.FILTER_TIMEOUT_S}s")
                    cache = get_cache(fr_settings)
                    filter_engine = FilterEngine(cache, fr_settings)
                    
                    print(f"[Filter-Rerank] Filtering {len(chunks)} chunks...")
                    filtered = filter_engine.filter_chunks(request.query, chunks)
                    print(f"[Filter-Rerank] Filtered: {len(filtered)}/{len(chunks)} survivors")
                    
                    # Stage 2: Rerank
                    if not fr_settings.DISABLE_RERANK and filtered:
                        reranker = Reranker(cache, fr_settings)
                        items = build_items_for_rerank(filtered)
                        scores = reranker.rerank(request.query, items)
                        
                        # Map scores back and sort
                        score_map = {s.id: s.score for s in scores}
                        for f in filtered:
                            f.chunk.rerank_score = score_map.get(f.chunk.id, 0.0)
                        
                        filtered.sort(key=lambda f: (-f.chunk.rerank_score or 0, -f.score, f.chunk.provenance_id))
                    
                    # Convert back to search result format (match simple_search structure with _source wrapper)
                    fused = []
                    for f in filtered[:request.final_k]:
                        chunk_dict = {
                            "_id": f.chunk.id,
                            "_score": f.chunk.rerank_score or f.score,
                            "_source": {
                                "id": f.chunk.id,
                                "repo_id": f.chunk.repo_id,
                                "rel_path": f.chunk.rel_path,
                                "start_line": f.chunk.start_line,
                                "end_line": f.chunk.end_line,
                                "language": f.chunk.language,
                                "text": f.chunk.text,
                                "summary_en": f.chunk.summary_en,
                                "filter_score": f.score,
                                "rerank_score": f.chunk.rerank_score,
                                "filter_why": f.why
                            }
                        }
                        fused.append(chunk_dict)
                    
                    print(f"[Filter-Rerank] Filtered {len(filtered)}/{len(chunks)} chunks, returning top {len(fused)}")
                    
                except Exception as e:
                    print(f"[Filter-Rerank] Error: {e}, falling back to standard search")
                    # Fallback to standard search
                    fused = search_v4_service.simple_search(
                        request.query,
                        repo_ids,
                        final_k=request.final_k,
                        fetch_all_texts=True
                    )
            else:
                # Standard search without filter-rerank
                fused = search_v4_service.simple_search(
                    request.query,
                    repo_ids,
                    final_k=request.final_k,
                    fetch_all_texts=True,  # Fetch text for all chunks for display
                    retriever_index=request.retriever_index
                )
        else:
            # Use legacy search (v2/v3)
            # 3) Encode query for kNN search
            qvec = search_service.encode_query(request.query)
            
            # 4) Define source fields based on chunk version
            if request.chunks_version == "v3":
                source_fields = [
                    "id", "repo_id", "language", "path", "rel_path", "ext",
                    "chunk_number", "start_line", "end_line", "text", "n_tokens", "symbols",
                    "primary_symbol", "primary_kind", "primary_span", 
                    "def_symbols", "doc_head"
                ]
            else:
                source_fields = [
                    "id", "repo_id", "language", "path", "rel_path", "ext",
                    "chunk_number", "start_line", "end_line", "text", "n_tokens", "symbols"
                ]
            
            # 5) Execute BM25 and kNN searches
            # Use standard BM25 search
            bbody = search_service.search_bm25(
                request.query, repo_ids, request.bm25_size, source_fields, request.chunks_version
            )
            use_fetch_forecast = False
            
            bhits = search_service.client.search(index=chunks_index, body=bbody)["hits"]["hits"]
            
            # kNN search
            khits = search_service.search_knn(
                qvec, repo_ids, chunks_index, request.knn_size, request.knn_candidates, source_fields
            )
            
            # 6) Fuse results with RRF and optionally rerank
            fused = search_service.fuse_and_rerank(
                [bhits, khits], 
                query=request.query,
                final_k=request.final_k,
                rerank=request.rerank,
                rerank_size=request.rerank_size,
                rerank_model=request.rerank_model,
                bm25_size=request.bm25_size,
                knn_size=request.knn_size
            )
        
        # 6) Get repo guides
        repo_guides_data = search_service.get_repo_guides(repo_ids, DEFAULT_REPO_GUIDE_INDEX)
        repo_guides = []
        for guide in repo_guides_data:
            repo_guides.append(RepoGuide(
                repo_id=guide.get("repo_id", ""),
                overview=guide.get("overview"),
                key_flows=guide.get("key_flows"), 
                entrypoints=guide.get("entrypoints"),
                languages=guide.get("languages", []),
                modules=guide.get("modules")
            ))
        
        # 7) Convert search results
        results = []
        for hit in fused:
            src = hit.get("_source", {})
            results.append(SearchResult(
                id=src.get("id", ""),
                repo_id=src.get("repo_id", ""),
                language=src.get("language"),
                path=src.get("path"),
                rel_path=src.get("rel_path"),
                ext=src.get("ext"),
                chunk_number=src.get("chunk_number"),
                start_line=src.get("start_line"),
                end_line=src.get("end_line"),
                symbols=src.get("symbols", []),
                text=src.get("text", ""),
                n_tokens=src.get("n_tokens"),
                score=hit.get("_score", 0.0),
                # v3 fields
                primary_symbol=src.get("primary_symbol"),
                primary_kind=src.get("primary_kind"),
                primary_span=src.get("primary_span"),
                def_symbols=src.get("def_symbols"),
                doc_head=src.get("doc_head"),
                # Filter-rerank fields
                filter_score=src.get("filter_score"),
                rerank_score=src.get("rerank_score"),
                filter_why=src.get("filter_why")
            ))

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

@app.post("/qa", response_model=QAResponse)
async def ask_question(request: QARequest):
    """
    Q&A endpoint using modular LLM service
    """
    if not search_service or not llm_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        # 1) Router: find relevant repositories
        router_index = get_router_index(request.router_version)
        chunks_index = get_chunks_index(request.chunks_version)
        
        if request.explicit_repo:
            repo_ids = [request.explicit_repo]
            print(f"[QA] Using explicit repo: {request.explicit_repo}")
        else:
            # Use router to find relevant repos
            router_response = search_service.execute_router_search(request.question, router_index, request.top_repos)
            router_hits = router_response.get("hits", {}).get("hits", [])
            repo_ids = [h["_source"]["repo_id"] for h in router_hits if "_source" in h and "repo_id" in h["_source"]]
            print(f"[QA] Router returned repos: {repo_ids}")
        
            if not repo_ids:
                # Include debug info even for early returns
                debug_info = None
                if request.debug:
                    debug_info = DebugInfo(
                        llm_prompt="[No LLM call - No repositories found]",
                        raw_llm_response="No relevant repositories found for your question.",
                        topic_terms=[],
                        search_results_count=0,
                        router_repos=[],
                        bundle_sources=0,
                        processing_time_ms=0.0
                    )
                
                return QAResponse(
                    question=request.question,
                answer="No relevant repositories found for your question.",
                    routed_repos=[],
                    sources_used=0,
                model_used=request.llm_model,
                debug_info=debug_info
            )
        
        # 3) Search for relevant code
        # Use search_v4 if chunks_version is "v4"
        if request.chunks_version == "v4":
            if not search_v4_service:
                raise HTTPException(status_code=500, detail="Search V4 service not initialized")
            
            # Log retriever index selection
            print(f"[QA] Using retriever_index: {request.retriever_index or 'default (code_chunks_v4)'}")
            
            # Apply LLM filter + rerank if enabled
            if request.use_filter_rerank:
                try:
                    from rag_filter_rerank import FilterRerankPipeline, get_settings as get_fr_settings
                    from rag_filter_rerank.cache import get_cache
                    from rag_filter_rerank.filter_engine import FilterEngine
                    from rag_filter_rerank.reranker import Reranker, build_items_for_rerank
                    from rag_filter_rerank.schemas import Chunk, FilterResult
                    
                    # Get initial candidates (more than final_k for filtering)
                    recall_n = 100  # Get 100 candidates for filtering
                    candidates = search_v4_service.simple_search(
                        request.question,
                        repo_ids,
                        final_k=recall_n,
                        fetch_all_texts=True,
                        retriever_index=request.retriever_index
                    )
                    
                    print(f"[QA Filter-Rerank] Got {len(candidates)} candidates from search_v4")
                    
                    # Convert to Chunk objects
                    chunks = []
                    for idx, hit in enumerate(candidates):
                        # Helper to get field from either top-level or _source
                        def get_field(key, default=""):
                            val = hit.get(key)
                            if val is None and "_source" in hit:
                                val = hit["_source"].get(key)
                            return val if val is not None else default
                        
                        # Extract text from _source if present
                        text = hit.get("text", "")
                        if not text and "_source" in hit:
                            text = hit["_source"].get("text", "")
                        
                        # Debug first and last chunks
                        if idx == 0 or idx == len(candidates) - 1:
                            print(f"[QA Filter-Rerank] Chunk {idx} structure:")
                            print(f"  - Hit keys: {list(hit.keys())}")
                            print(f"  - Has _source: {'_source' in hit}")
                            print(f"  - Top-level text length: {len(hit.get('text', ''))}")
                            if "_source" in hit:
                                print(f"  - _source keys: {list(hit['_source'].keys())}")
                                print(f"  - _source text length: {len(hit['_source'].get('text', ''))}")
                            print(f"  - Final text length: {len(text)}")
                            print(f"  - id: {get_field('id')}")
                            print(f"  - rel_path: {get_field('rel_path')}")
                        
                        chunk = Chunk(
                            id=get_field("id"),
                            repo_id=get_field("repo_id"),
                            rel_path=get_field("rel_path"),
                            start_line=int(get_field("start_line", 0)),
                            end_line=int(get_field("end_line", 0)),
                            language=get_field("language"),
                            text=text,
                            summary_en=get_field("summary_en", None),
                            chunk_hash=get_field("chunk_hash", None),
                            retriever_score=hit.get("_score", 0.0)
                        )
                        
                        # Verify chunk has text
                        if idx == 0:
                            print(f"[QA Filter-Rerank] Created Chunk object with text length: {len(chunk.text)}")
                        
                        chunks.append(chunk)
                    
                    # Stage 1: LLM Filter
                    fr_settings = get_fr_settings()
                    print(f"[QA Filter-Rerank] Using model: {fr_settings.OLLAMA_MODEL}, timeout: {fr_settings.FILTER_TIMEOUT_S}s")
                    cache = get_cache(fr_settings)
                    filter_engine = FilterEngine(cache, fr_settings)
                    
                    print(f"[QA Filter-Rerank] Starting filter on {len(chunks)} chunks")
                    filtered = filter_engine.filter_chunks(request.question, chunks)
                    print(f"[QA Filter-Rerank] Filter complete: {len(filtered)} survivors")
                    
                    # Stage 2: Rerank
                    if not fr_settings.DISABLE_RERANK and filtered:
                        reranker = Reranker(cache, fr_settings)
                        items = build_items_for_rerank(filtered)
                        scores = reranker.rerank(request.question, items)
                        
                        # Map scores back and sort
                        score_map = {s.id: s.score for s in scores}
                        for f in filtered:
                            f.chunk.rerank_score = score_map.get(f.chunk.id, 0.0)
                        
                        filtered.sort(key=lambda f: (-f.chunk.rerank_score or 0, -f.score, f.chunk.provenance_id))
                    
                    # Convert back to search result format
                    fused = []
                    for idx, f in enumerate(filtered[:request.final_k]):
                        chunk_dict = {
                            "id": f.chunk.id,
                            "repo_id": f.chunk.repo_id,
                            "rel_path": f.chunk.rel_path,
                            "start_line": f.chunk.start_line,
                            "end_line": f.chunk.end_line,
                            "language": f.chunk.language,
                            "text": f.chunk.text,
                            "summary_en": f.chunk.summary_en,
                            "_score": f.chunk.rerank_score or f.score,
                            "filter_score": f.score,
                            "rerank_score": f.chunk.rerank_score,
                            "filter_why": f.why
                        }
                        
                        # Debug first chunk after conversion
                        if idx == 0:
                            print(f"[QA Filter-Rerank] Converted chunk 0:")
                            print(f"  - f.chunk.text length: {len(f.chunk.text)}")
                            print(f"  - chunk_dict['text'] length: {len(chunk_dict['text'])}")
                            print(f"  - chunk_dict keys: {list(chunk_dict.keys())}")
                        
                        fused.append(chunk_dict)
                    
                    print(f"[QA Filter-Rerank] Filtered {len(filtered)}/{len(chunks)} chunks, returning top {len(fused)}")
                    print(f"[QA Filter-Rerank] Sample fused[0] text length: {len(fused[0]['text']) if fused else 0}")
                    
                except Exception as e:
                    print(f"[QA Filter-Rerank] Error: {e}, falling back to standard search")
                    import traceback
                    traceback.print_exc()
                    # Fallback to standard search
                    fused = search_v4_service.simple_search(
                        request.question,
                        repo_ids,
                        final_k=request.final_k,
                        fetch_all_texts=True,
                        retriever_index=request.retriever_index
                    )
            else:
                # Standard search without filter-rerank
                fused = search_v4_service.simple_search(
                    request.question,
                    repo_ids,
                    final_k=request.final_k,
                    fetch_all_texts=True,  # Fetch text for all chunks for QA
                    retriever_index=request.retriever_index,
                    semantic_only=True if request.retriever_index else False
                )
        else:
            # Use legacy search (v2/v3)
            qvec = search_service.encode_query(request.question)
            source_fields = [
                "id", "repo_id", "language", "path", "rel_path", "ext",
                "chunk_number", "start_line", "end_line", "text", "n_tokens", "symbols",
                "primary_symbol", "primary_kind", "def_symbols", "doc_head"
            ]
            
            # Execute searches
            # Use standard BM25 search
            bbody = search_service.search_bm25(
                request.question, repo_ids, 200, source_fields, request.chunks_version
            )
            use_fetch_forecast = False
            
            bhits = search_service.client.search(index=chunks_index, body=bbody)["hits"]["hits"]
            
            khits = search_service.search_knn(
                qvec, repo_ids, chunks_index, 200, 400, source_fields
            )
            
            # Fuse and validate with optional reranking
            fused = search_service.fuse_and_rerank(
                [bhits, khits], 
                query=request.question,
                final_k=30,
                rerank=request.rerank,
                rerank_size=request.rerank_size,
                rerank_model=request.rerank_model,
                bm25_size=200,
                knn_size=200
            )
        
        # Check if RAG v4 mode is enabled
        if request.rag_mode == "v4":
            if not rag_v4_service:
                raise HTTPException(status_code=500, detail="RAG v4 service not initialized")
            
            # Use RAG v4 Evidence-based answering
            # Convert fused results to chunks format
            chunks = []
            for hit in fused[:request.final_k]:
                # Handle both formats: with _source (standard search) and without (filter-rerank)
                if "_source" in hit:
                    src = hit["_source"]
                else:
                    # Filter-rerank returns flat dicts
                    src = hit
                chunks.append(src)
            
            print(f"[QA RAGv4] Prepared {len(chunks)} chunks, sample chunk keys: {list(chunks[0].keys()) if chunks else []}")
            print(f"[QA RAGv4] Sample chunk text length: {len(chunks[0].get('text', '')) if chunks else 0}")
            
            # Call RAG v4 service with extended timeout for detailed answers
            rag_result = rag_v4_service.answer_question(
                question=request.question,
                chunks=chunks,
                max_items=request.final_k,
                max_code_chars=10000,
                timeout=240  # Increased for longer, detailed answers
            )
            
            # Extract citations
            citations = rag_v4_service.extract_citations(rag_result["answer"])
            
            # Build debug info if requested
            debug_info = None
            if request.debug:
                debug_info = DebugInfo(
                    llm_prompt=rag_result["evidence"],
                    raw_llm_response=rag_result["answer"],
                    topic_terms=[],
                    search_results_count=len(fused),
                    router_repos=repo_ids,
                    bundle_sources=rag_result["chunks_used"],
                    processing_time_ms=0.0
                )
            
            return QAResponse(
                question=request.question,
                answer=rag_result["answer"],
                routed_repos=repo_ids,
                sources_used=rag_result["chunks_used"],
                model_used=rag_result["model"],
                debug_info=debug_info
            )
        
        # 4) Get repo guides and build LLM bundle (Legacy mode)
        repo_guides_data = search_service.get_repo_guides(repo_ids)
        
        # Get top results for LLM
        top_results = fused[:request.final_k]
        
        # Build bundle using search service
        bundle = search_service.create_llm_bundle(
            query=request.question,
            repo_ids=repo_ids,
            search_results=top_results,
            repo_context={"hits": [], "guides": repo_guides_data},
            repo_guides=repo_guides_data,
            max_lines=request.max_lines
        )
        
        # 5) Use LLM service to generate answer
        debug_info = None
        if request.debug:
            import time
            start_time = time.time()
            
            # Build the prompt to capture for debugging
            prompt = llm_service.build_qa_prompt(
                bundle=bundle,
                max_lines_per_chunk=request.max_lines,
                spotlight_n=request.spotlight_chunks,
                topic_terms=[]
            )
            
            # Call LLM directly to capture raw response
            raw_response = llm_service.call_llm(
            prompt=prompt,
            temperature=request.temperature
        )

            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            debug_info = DebugInfo(
                llm_prompt=prompt,
                raw_llm_response=raw_response,
                topic_terms=[],
                search_results_count=len(fused),
                router_repos=repo_ids,
                bundle_sources=len(top_results),
                processing_time_ms=processing_time
            )
            
            answer = raw_response
        else:
            # Normal flow without debug capture
            answer = llm_service.answer_question(
                bundle=bundle,
                temperature=request.temperature,
                max_lines=request.max_lines,
                spotlight_count=request.spotlight_chunks,
                topic_terms=[],
                max_sources=request.final_k,
                include_repo_context=request.include_repo_context,
                disable_guards=True
            )

        return QAResponse(
            question=request.question,
            answer=answer,
            routed_repos=repo_ids,
            sources_used=len(top_results),
            model_used=request.llm_model,
            debug_info=debug_info
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A failed: {e}")

# Simple endpoints for testing
@app.get("/search/simple")
async def simple_search(
    q: str = Query(..., description="Search query"),
    repo: Optional[str] = Query(None, description="Specific repository ID"),
    router_version: str = Query("v1"),
    chunks_version: str = Query("v2")
):
    """Simplified search endpoint for quick testing"""
    request = SearchRequest(
        query=q,
        explicit_repo=repo,
        router_version=router_version,
        chunks_version=chunks_version
    )
    return await search_code(request)

@app.get("/qa/simple")
async def simple_qa(
    q: str = Query(..., description="Question to ask"),
    repo: Optional[str] = Query(None, description="Specific repository ID"),
    model: str = Query(DEFAULT_LLM_MODEL, description="Ollama model to use"),
    router_version: str = Query("v1"),
    chunks_version: str = Query("v2"),
    debug: bool = Query(False, description="Enable debug recording")
):
    """Simplified Q&A endpoint for quick testing"""
    request = QARequest(
        question=q,
        explicit_repo=repo,
        llm_model=model,
        router_version=router_version,
        chunks_version=chunks_version,
        debug=debug
    )
    return await ask_question(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)