"""
Search orchestration: planner → retrieval → rerank → LLM payload.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from .config import TOP_K, WITH_TEXT_TOP, INDEX_NAME, RETRIEVER_INDEX, RRF_K, MAX_PER_FILE, KNN_SIZE, BM25_SIZE
from .opensearch_client import mget
from .retrieval import knn_candidates, bm25_candidates
from .rerank import rrf_fuse, apply_symbol_boosts, diversify
from .embeddings import get_embedding_for_index


def _fetch_texts(doc_ids: List[str], index_name: str = None) -> Dict[str, str]:
    """
    Fetch full text for specified document IDs.
    
    Args:
        doc_ids: List of document IDs
    
    Returns:
        Dict mapping doc_id -> text
    """
    if not doc_ids:
        return {}
    
    index = index_name or RETRIEVER_INDEX
    res = mget(index, doc_ids, _source=["text"])
    out = {}
    
    for doc in res.get("docs", []):
        if doc.get("found"):
            doc_id = doc["_id"]
            text = (doc.get("_source") or {}).get("text", "")
            out[doc_id] = text
    
    return out


def search_v4(
    query: str,
    router_repo_ids: List[str],
    plan: Dict[str, Any],
    fetch_all_texts: bool = False,
    retriever_index: str = None,
    semantic_only: bool = False
) -> Dict[str, Any]:
    """
    Execute hybrid search with RRF fusion and symbol-aware reranking.
    
    Args:
        query: Original user query
        router_repo_ids: Repo IDs from repo router (1-3 repos)
        plan: Query plan from query_planner.py
        fetch_all_texts: If True, fetch text for all results (for QA). If False, only top WITH_TEXT_TOP
    
    Returns:
        Search results with metadata + text for top results
    """
    # Determine semantic text for embedding
    semantic_text = (
        plan.get("hyde_passage") or
        plan.get("clarified_query") or
        query
    ).strip()
    
    # Extract language filter if present
    language = (plan.get("language") or "").strip() or None
    
    # Get query embedding
    query_vec = get_embedding_for_index(semantic_text, retriever_index)
    
    # Run kNN and BM25 in parallel
    if semantic_only:
        knn_hits = knn_candidates(
            query_vec,
            router_repo_ids,
            language,
            KNN_SIZE,
            retriever_index
        )
        bm25_hits = []
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_knn = executor.submit(
                knn_candidates,
                query_vec,
                router_repo_ids,
                language,
                KNN_SIZE,
                retriever_index
            )
            future_bm25 = executor.submit(
                bm25_candidates,
                plan,
                router_repo_ids,
                BM25_SIZE,
                retriever_index
            )
            
            knn_hits = future_knn.result()
            bm25_hits = future_bm25.result()
    
    print(f"[search_v4] kNN returned {len(knn_hits)} hits, BM25 returned {len(bm25_hits)} hits")
    
    # RRF fusion
    scores, hits_by_id = rrf_fuse(knn_hits, bm25_hits, rrf_k=RRF_K)
    print(f"[search_v4] After RRF fusion: {len(scores)} unique docs")
    
    # Apply symbol-aware boosts
    apply_symbol_boosts(
        scores,
        hits_by_id,
        plan.get("identifiers", []),
        plan.get("file_hints", [])
    )
    
    # Sort by final score
    sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    
    # Diversify (limit per file)
    diversified = diversify(sorted_ids, hits_by_id, max_per_file=MAX_PER_FILE)
    print(f"[search_v4] After diversification: {len(diversified)} docs")
    
    # Take top K
    final_ids = diversified[:TOP_K]
    print(f"[search_v4] Returning top {len(final_ids)} docs (TOP_K={TOP_K})")
    
    # Fetch full text for top WITH_TEXT_TOP only (or all if fetch_all_texts=True)
    if fetch_all_texts:
        texts = _fetch_texts(final_ids, index_name=retriever_index)
    else:
        texts = _fetch_texts(final_ids[:WITH_TEXT_TOP], index_name=retriever_index)
    
    # Build results
    results = []
    for doc_id in final_ids:
        hit = hits_by_id[doc_id]
        
        item = {
            "id": doc_id,
            "repo_id": hit.get("repo_id"),
            "rel_path": hit.get("rel_path"),
            "primary_symbol": hit.get("primary_symbol"),
            "all_symbols": hit.get("all_symbols"),
            "all_roles": hit.get("all_roles"),
            "start_line": hit.get("start_line"),
            "end_line": hit.get("end_line"),
            "summary_en": hit.get("summary_en"),
        }
        
        # Add text only for top WITH_TEXT_TOP
        if doc_id in texts:
            item["text"] = texts[doc_id]
        
        results.append(item)
    
    return {
        "query": query,
        "router_repo_ids": router_repo_ids,
        "plan": plan,
        "results": results
    }

