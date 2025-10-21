"""
OpenSearch retrieval: kNN (semantic) and BM25 (lexical) queries.
"""
from typing import Any, Dict, List, Optional

from .config import INDEX_NAME, RETRIEVER_INDEX, KNN_K, KNN_NUM_CANDIDATES, KNN_SIZE, BM25_SIZE
from .opensearch_client import search as os_search


# Minimal fields to retrieve for ranking (before fetching full text)
FIELDS_MIN = [
    "id", "repo_id", "rel_path", "summary_en", "primary_symbol",
    "all_symbols", "all_roles", "start_line", "end_line", "language"
]


def _parse_hits(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse OpenSearch hits into normalized minimal payloads.
    
    Args:
        res: OpenSearch response dict
    
    Returns:
        List of hit dicts with minimal metadata
    """
    out = []
    for i, hit in enumerate(res.get("hits", {}).get("hits", []), start=1):
        src = hit.get("_source") or {}
        fields = hit.get("fields") or {}
        
        # Merge _source and fields, normalize lists
        merged = {**src}
        for k, v in fields.items():
            # OpenSearch fields API returns lists for all values
            merged[k] = v[0] if isinstance(v, list) and len(v) == 1 else v
        
        # Build minimal hit
        hit_data = {
            "id": hit.get("_id") or merged.get("id"),
            "rank": i,
            "score": hit.get("_score"),
        }
        
        # Add requested fields
        for field in FIELDS_MIN:
            if field in merged:
                hit_data[field] = merged[field]
        
        out.append(hit_data)
    
    return out


def knn_candidates(
    query_vec: List[float],
    repo_ids: List[str],
    language: Optional[str] = None,
    size: int = KNN_SIZE,
    index_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve candidates using kNN semantic search.
    
    Args:
        query_vec: 384-dim embedding vector
        repo_ids: List of repo IDs to filter by
        language: Optional language filter (e.g., "go")
        size: Number of results to retrieve
    
    Returns:
        List of candidate hits with minimal metadata
    """
    filters = []
    
    if repo_ids:
        filters.append({"terms": {"repo_id": repo_ids}})
    
    if language:
        filters.append({"term": {"language": language}})
    
    # Build k-NN query (OpenSearch 3.x format: query.knn)
    knn_query: Dict[str, Any] = {
        "knn": {
            "embedding": {
                "vector": query_vec,
                "k": size
            }
        }
    }
    
    # Wrap with filter if needed
    if filters:
        query_clause = {
            "bool": {
                "must": [knn_query],
                "filter": filters
            }
        }
    else:
        query_clause = knn_query
    
    body: Dict[str, Any] = {
        "size": size,
        "_source": FIELDS_MIN,
        "query": query_clause
    }
    
    index = index_name or RETRIEVER_INDEX
    res = os_search(index, body)
    return _parse_hits(res)


def bm25_candidates(
    plan: Dict[str, Any],
    repo_ids: List[str],
    size: int = BM25_SIZE,
    index_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve candidates using BM25 lexical search.
    
    Args:
        plan: Query plan from query_planner.py
        repo_ids: List of repo IDs to filter by
        size: Number of results to retrieve
    
    Returns:
        List of candidate hits with minimal metadata
    """
    should = []
    
    # Add planner-provided should clauses (up to 12)
    for item in (plan.get("bm25_should") or [])[:12]:
        field = item.get("field")
        term = item.get("term")
        boost = item.get("boost", 1.0)
        
        if not field or not term:
            continue
        
        # Use match_phrase for primary_symbol (exact symbol matching)
        # Use match for other fields (fuzzy/stemmed)
        if field == "primary_symbol":
            should.append({
                "match_phrase": {
                    field: {
                        "query": term,
                        "boost": boost
                    }
                }
            })
        else:
            should.append({
                "match": {
                    field: {
                        "query": term,
                        "boost": boost
                    }
                }
            })
    
    # Add helpful defaults from clarified query
    clarified_query = (plan.get("clarified_query") or "").strip()
    if clarified_query:
        should.append({
            "match": {
                "summary_en": {
                    "query": clarified_query,
                    "boost": 2.0
                }
            }
        })
        should.append({
            "match": {
                "text": {
                    "query": clarified_query,
                    "boost": 1.0
                }
            }
        })
    
    # Build must clauses (hard filters)
    must = []
    language = (plan.get("language") or "").strip()
    if language:
        must.append({"term": {"language": language}})
    
    # Build filter clauses
    filters = []
    if repo_ids:
        filters.append({"terms": {"repo_id": repo_ids}})
    
    # Construct final query
    body: Dict[str, Any] = {
        "size": size,
        "_source": FIELDS_MIN,
        "query": {
            "bool": {
                "filter": filters,
                "must": must,
                "should": should,
                "minimum_should_match": 0  # Should clauses are optional (boost only)
            }
        }
    }
    
    index = index_name or RETRIEVER_INDEX
    res = os_search(index, body)
    return _parse_hits(res)

