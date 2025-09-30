"""
Search Service

Provides clean interface to search functionality using existing scripts.
"""

import sys
import os
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

# Import search functionality from existing scripts
from search_into_json import (
    detect_topic_terms,
    passes_relevance_gate, 
    bm25_v3_guarded,
    bm25_filtered_query,
    bm25_v3_query,
    bm25_fetch_forecast,
    detect_fetch_forecast_intent,
    evidence_overlap_ok,
    knn_search_filtered,
    rrf_fuse,
    rerank_with_cross_encoder,
    _chunk_text_for_rerank,
    fetch_router_context,
    fetch_repo_guides,
    build_llm_bundle,
    router_query,
    TOPIC_LEX
)

class SearchService:
    """Service for handling search operations"""
    
    def __init__(self, opensearch_client: OpenSearch, embedding_model: SentenceTransformer):
        self.client = opensearch_client
        self.model = embedding_model
    
    def detect_query_topics(self, query: str) -> List[str]:
        """Detect topic terms in a query"""
        return detect_topic_terms(query)
    
    def validate_topic_evidence(self, hits: List[Dict], topic_terms: List[str], 
                               k_check: int = 8, min_hits: int = 2) -> bool:
        """Check if search results contain sufficient topic evidence"""
        return passes_relevance_gate(hits, topic_terms, k_check, min_hits)
    
    def validate_evidence_overlap(self, question: str, fused_hits: List[Dict], 
                                 min_distinct: int = 2, top_n: int = 10) -> bool:
        """Validate that search results have sufficient token overlap with query"""
        return evidence_overlap_ok(question, fused_hits, min_distinct, top_n)
    
    def search_with_topic_guard(self, query: str, repo_ids: List[str], 
                               size: int, topic_terms: List[str],
                               source_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform topic-guarded BM25 search"""
        return bm25_v3_guarded(query, repo_ids, size, topic_terms, source_fields)
    
    def search_bm25(self, query: str, repo_ids: List[str], size: int,
                    source_fields: Optional[List[str]] = None, 
                    version: str = "v2") -> Dict[str, Any]:
        """Perform standard BM25 search with intent detection"""
        # Check for fetch forecast intent first
        if detect_fetch_forecast_intent(query):
            return bm25_fetch_forecast(query, repo_ids, size, source_fields)
        
        # Fall back to existing logic
        if version == "v3":
            return bm25_v3_query(query, repo_ids, size, source_fields)
        else:
            return bm25_filtered_query(query, repo_ids, size, source_fields)
    
    def search_bm25_with_intent(self, query: str, repo_ids: List[str], size: int,
                               source_fields: Optional[List[str]] = None) -> tuple[Dict[str, Any], bool]:
        """Perform BM25 search with intent detection, returns (query_body, used_fetch_intent)"""
        use_fetch_forecast = detect_fetch_forecast_intent(query)
        
        if use_fetch_forecast:
            query_body = bm25_fetch_forecast(query, repo_ids, size, source_fields)
        else:
            # Topic detection for guarded retrieval
            topic_terms = detect_topic_terms(query)
            
            if topic_terms:
                query_body = bm25_v3_guarded(query, repo_ids, size, topic_terms, source_fields)
            else:
                # Prefer v3 BM25 if available
                try:
                    query_body = bm25_v3_query(query, repo_ids, size, source_fields)
                except NameError:
                    query_body = bm25_filtered_query(query, repo_ids, size, source_fields)
        
        return query_body, use_fetch_forecast
    
    def search_knn(self, query_vector: List[float], repo_ids: List[str],
                   index: str, k: int, num_candidates: int,
                   source_fields: Optional[List[str]] = None) -> List[Dict]:
        """Perform kNN vector search"""
        return knn_search_filtered(
            self.client, index, "vector", query_vector, repo_ids,
            k, num_candidates, source_fields
        )
    
    def fuse_results(self, result_lists: List[List[Dict]], final_k: int) -> List[Dict]:
        """Combine multiple search results using RRF"""
        return rrf_fuse(result_lists, K=final_k)
    
    def rerank_results(self, query: str, fused_hits: List[Dict], 
                      model_name: str, topn: int, pool_size: int) -> List[Dict]:
        """Rerank search results using cross-encoder"""
        return rerank_with_cross_encoder(query, fused_hits, model_name, topn, pool_size)
    
    def fuse_and_rerank(self, result_lists: List[List[Dict]], query: str,
                       final_k: int, rerank: bool = False, 
                       rerank_size: int = 60, rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
                       bm25_size: int = 200, knn_size: int = 200) -> List[Dict]:
        """Fuse results and optionally rerank them"""
        # Build a larger fused pool first
        pool_K = max(final_k, rerank_size, bm25_size, knn_size)
        fused_pool = rrf_fuse(result_lists, K=pool_K)
        
        # Optional rerank
        if rerank:
            return rerank_with_cross_encoder(query, fused_pool, rerank_model, final_k, rerank_size)
        else:
            return fused_pool[:final_k]
    
    def search_router(self, query: str, topn: int = 5) -> Dict[str, Any]:
        """Search router for relevant repositories"""
        return router_query(query, topn)
    
    def get_router_context(self, index: str, repo_ids: List[str]) -> Dict[str, Any]:
        """Fetch repository router context"""
        return fetch_router_context(self.client, index, repo_ids)
    
    def execute_router_search(self, query: str, index: str, topn: int = 5) -> Dict[str, Any]:
        """Execute router search and return results"""
        body = self.search_router(query, topn)
        return self.client.search(index=index, body=body)
    
    def get_repo_guides(self, repo_ids: List[str], index: str = "repo_guide_v1") -> List[Dict[str, Any]]:
        """Fetch repository guides"""
        return fetch_repo_guides(self.client, index, repo_ids)
    
    def create_llm_bundle(self, query: str, repo_ids: List[str], 
                         search_results: List[Dict], repo_context: Dict[str, Any],
                         repo_guides: List[Dict] = None, max_lines: int = 120) -> Dict[str, Any]:
        """Create LLM-ready bundle from search results"""
        return build_llm_bundle(
            query=query,
            repo_ids=repo_ids, 
            fused_hits=search_results,
            repo_context=repo_context,
            max_lines_per_chunk=max_lines,
            repo_guides=repo_guides
        )
    
    def encode_query(self, query: str) -> List[float]:
        """Encode query text to embedding vector"""
        return self.model.encode(query, normalize_embeddings=True).tolist()
    
    def list_repositories(self, router_index: str) -> List[Dict[str, Any]]:
        """List all repositories from router index"""
        try:
            body = {
                "size": 100,  # Get up to 100 repositories
                "query": {"match_all": {}},
                "_source": ["repo_id", "short_title", "summary", "languages", "domains", "tech_stack"]
            }
            
            response = self.client.search(index=router_index, body=body)
            repositories = []
            
            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                repositories.append({
                    "repo_id": source.get("repo_id", ""),
                    "short_title": source.get("short_title", ""),
                    "summary": source.get("summary", ""),
                    "languages": source.get("languages", []),
                    "domains": source.get("domains", "unknown"),
                    "tech_stack": source.get("tech_stack", "unknown")
                })
            
            return repositories
            
        except Exception as e:
            print(f"Error listing repositories from {router_index}: {e}")
            return []
