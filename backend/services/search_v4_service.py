"""
Search V4 Service Wrapper

Wraps the new search_v4 hybrid search system for use in the backend API.
"""
import sys
import os
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from search_v4.service import search_v4
from opensearchpy import OpenSearch

class SearchV4Service:
    """Service wrapper for search_v4 hybrid search"""
    
    def __init__(self, opensearch_client: OpenSearch):
        """
        Initialize the search_v4 service.
        
        Args:
            opensearch_client: OpenSearch client instance
        """
        self.client = opensearch_client
    
    def search(
        self,
        query: str,
        router_repo_ids: List[str],
        query_plan: Optional[Dict[str, Any]] = None,
        final_k: int = 10,
        fetch_all_texts: bool = False,
        retriever_index: Optional[str] = None,
        semantic_only: bool = False
    ) -> Dict[str, Any]:
        """
        Execute hybrid search using search_v4.
        
        Args:
            query: User query
            router_repo_ids: List of repo IDs from router
            query_plan: Optional query plan from query_planner.py
            final_k: Number of final results to return
            fetch_all_texts: If True, fetch text for all results (needed for QA)
        
        Returns:
            Search results dict with OpenSearch-compatible format
        """
        # Build a minimal query plan if not provided
        if query_plan is None:
            query_plan = {
                "clarified_query": query,
                "identifiers": [],
                "file_hints": [],
                "language": None,
                "bm25_should": []
            }
        
        # Call search_v4
        try:
            print(f"[SearchV4Service] Starting search: query='{query}', repos={router_repo_ids}, final_k={final_k}, fetch_all_texts={fetch_all_texts}")
            
            # Update TOP_K temporarily for this search
            from search_v4 import config
            original_top_k = config.TOP_K
            config.TOP_K = final_k
            
            results = search_v4(
                query,
                router_repo_ids,
                query_plan,
                fetch_all_texts=fetch_all_texts,
                retriever_index=retriever_index,
                semantic_only=semantic_only
            )
            
            print(f"[SearchV4Service] Got {len(results.get('results', []))} results from search_v4")
            
            # Restore original
            config.TOP_K = original_top_k
            
            # Convert to OpenSearch-compatible format
            hits = []
            for result in results.get("results", []):
                hit = {
                    "_id": result.get("id"),
                    "_score": 1.0,  # search_v4 doesn't return scores yet
                    "_source": {
                        "id": result.get("id"),
                        "repo_id": result.get("repo_id"),
                        "rel_path": result.get("rel_path"),
                        "language": result.get("language"),
                        "primary_symbol": result.get("primary_symbol"),
                        "all_symbols": result.get("all_symbols"),
                        "all_roles": result.get("all_roles"),
                        "start_line": result.get("start_line"),
                        "end_line": result.get("end_line"),
                        "summary_en": result.get("summary_en"),
                        "text": result.get("text", ""),
                        # Include all other fields from v3 chunks
                        **{k: v for k, v in result.items() if k not in [
                            "id", "repo_id", "rel_path", "language", "primary_symbol",
                            "all_symbols", "all_roles", "start_line", "end_line",
                            "summary_en", "text"
                        ]}
                    }
                }
                hits.append(hit)
            
            return hits
            
        except Exception as e:
            # Fall back to empty results on error
            print(f"[SearchV4Service] Error during search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def simple_search(
        self,
        query: str,
        repo_ids: List[str],
        final_k: int = 10,
        fetch_all_texts: bool = False,
        retriever_index: Optional[str] = None,
        semantic_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Simplified search interface that just needs query and repos.
        
        Args:
            query: User query
            repo_ids: List of repo IDs to search
            final_k: Number of results to return
            fetch_all_texts: If True, fetch text for all results (needed for QA)
        
        Returns:
            List of search result hits
        """
        return self.search(query, repo_ids, query_plan=None, final_k=final_k, fetch_all_texts=fetch_all_texts, retriever_index=retriever_index, semantic_only=semantic_only)

