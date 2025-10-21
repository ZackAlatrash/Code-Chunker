"""
Reranking and diversification: RRF fusion + symbol-aware boosts + per-file limits.
"""
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def rrf_fuse(
    knn_hits: List[Dict[str, Any]],
    bm25_hits: List[Dict[str, Any]],
    rrf_k: int = 60
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Reciprocal Rank Fusion: combine kNN and BM25 rankings.
    
    RRF score for doc d = sum over rankings of: 1 / (k + rank_i)
    
    Args:
        knn_hits: Results from kNN search
        bm25_hits: Results from BM25 search
        rrf_k: RRF constant (typically 60)
    
    Returns:
        Tuple of (scores dict, hits_by_id dict)
    """
    scores: Dict[str, float] = defaultdict(float)
    hits_by_id: Dict[str, Dict[str, Any]] = {}
    
    # Process kNN hits
    for i, hit in enumerate(knn_hits, start=1):
        doc_id = hit["id"]
        hits_by_id[doc_id] = {**hits_by_id.get(doc_id, {}), **hit}
        scores[doc_id] += 1.0 / (rrf_k + i)
    
    # Process BM25 hits
    for i, hit in enumerate(bm25_hits, start=1):
        doc_id = hit["id"]
        hits_by_id[doc_id] = {**hits_by_id.get(doc_id, {}), **hit}
        scores[doc_id] += 1.0 / (rrf_k + i)
    
    return scores, hits_by_id


def apply_symbol_boosts(
    scores: Dict[str, float],
    hits_by_id: Dict[str, Dict[str, Any]],
    identifiers: List[str],
    file_hints: List[str]
) -> None:
    """
    Apply symbol-aware boosts to RRF scores (in-place).
    
    Boosts:
        +0.25: primary_symbol in identifiers
        +0.15: any all_symbols in identifiers
        +0.10: role is "complete"
        +0.05: role is "declaration"
        +0.10: file_hints match rel_path
        -0.05: role is "mixed" and a "complete" version exists for same symbol
    
    Args:
        scores: Score dict (modified in-place)
        hits_by_id: Hits metadata by doc ID
        identifiers: Extracted identifiers from query
        file_hints: File path hints from query
    """
    # Normalize inputs
    ids_set = set(s for s in identifiers if s)
    hints = [h.lower() for h in (file_hints or []) if h]
    
    # Precompute best role by primary_symbol (for mixed penalty)
    best_role_by_sym: Dict[str, int] = {}
    for hit in hits_by_id.values():
        sym = hit.get("primary_symbol") or ""
        roles = set(hit.get("all_roles") or [])
        
        if not sym:
            continue
        
        # Rank: complete=2, declaration=1, else=0
        rank = 0
        if "complete" in roles:
            rank = 2
        elif "declaration" in roles:
            rank = 1
        
        # Track best rank seen for this symbol
        prev_rank = best_role_by_sym.get(sym, 0)
        if rank > prev_rank:
            best_role_by_sym[sym] = rank
    
    # Apply boosts per document
    for doc_id, hit in hits_by_id.items():
        boost = 0.0
        
        # Extract metadata
        primary_symbol = hit.get("primary_symbol") or ""
        all_symbols = set(hit.get("all_symbols") or [])
        all_roles = set(hit.get("all_roles") or [])
        rel_path = (hit.get("rel_path") or "").lower()
        
        # +0.25: Primary symbol match
        if primary_symbol and primary_symbol in ids_set:
            boost += 0.25
        
        # +0.15: Any symbol match
        if ids_set.intersection(all_symbols):
            boost += 0.15
        
        # +0.10: Complete role
        if "complete" in all_roles:
            boost += 0.10
        # +0.05: Declaration role (only if not complete)
        elif "declaration" in all_roles:
            boost += 0.05
        
        # +0.10: File path hint match
        if any(hint in rel_path for hint in hints):
            boost += 0.10
        
        # -0.05: Mixed penalty if better version exists
        if "mixed" in all_roles and primary_symbol:
            best_rank = best_role_by_sym.get(primary_symbol, 0)
            if best_rank >= 2:  # A "complete" version exists
                boost -= 0.05
        
        # Apply boost
        scores[doc_id] += boost


def diversify(
    sorted_ids: List[str],
    hits_by_id: Dict[str, Dict[str, Any]],
    max_per_file: int = 3
) -> List[str]:
    """
    Diversify results by limiting chunks per file.
    
    Args:
        sorted_ids: Document IDs sorted by score
        hits_by_id: Hits metadata by doc ID
        max_per_file: Maximum chunks per file path
    
    Returns:
        Diversified list of document IDs
    """
    per_file: Dict[str, int] = {}
    out: List[str] = []
    
    for doc_id in sorted_ids:
        hit = hits_by_id[doc_id]
        rel_path = hit.get("rel_path") or ""
        
        # Check current count for this file
        count = per_file.get(rel_path, 0)
        if count >= max_per_file:
            continue
        
        # Add to results
        out.append(doc_id)
        per_file[rel_path] = count + 1
    
    return out

