"""
Local Reranker with fallback logic.

Uses local cross-encoder models (HuggingFace transformers) for reranking.
Gracefully falls back to order-preserving scores if model loading fails.
"""
import logging
from typing import List, Optional

from .schemas import RerankItem, RerankScore, FilterResult
from .cache import Cache
from .truncation import build_rerank_snippet
from .reranker_local import create_local_reranker, LocalCrossEncoderReranker

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranker with local cross-encoder support and fallback logic.
    """
    
    def __init__(self, cache: Cache, settings):
        """
        Initialize reranker.
        
        Args:
            cache: Cache instance
            settings: Configuration settings
        """
        self.cache = cache
        self.settings = settings
        self.local_reranker: Optional[LocalCrossEncoderReranker] = None
        
        # Try to load local reranker
        if not settings.DISABLE_RERANK:
            logger.info("Initializing local cross-encoder reranker...")
            self.local_reranker = create_local_reranker(cache, settings)
            if self.local_reranker:
                logger.info("✅ Local reranker ready")
            else:
                logger.warning("❌ Local reranker not available, will use fallback")
        else:
            logger.info("Reranking disabled by config")
    
    def rerank(self, query: str, items: List[RerankItem]) -> List[RerankScore]:
        """
        Rerank items using local cross-encoder or fallback.
        
        Args:
            query: Query string
            items: Items to rerank
        
        Returns:
            List of RerankScore ordered by relevance
        """
        if not items:
            return []
        
        # Check if reranking is disabled
        if self.settings.DISABLE_RERANK:
            logger.info("Reranking disabled, using fallback scores")
            return self._fallback_scores(items)
        
        # Try local reranker
        if self.local_reranker:
            try:
                return self.local_reranker.rerank(query, items)
            except Exception as e:
                logger.error(f"Local reranking failed: {e}, falling back")
                return self._fallback_scores(items)
        else:
            logger.warning("No reranker available, using fallback scores")
            return self._fallback_scores(items)
    
    def _fallback_scores(self, items: List[RerankItem]) -> List[RerankScore]:
        """
        Generate fallback scores (pass-through order).
        
        Args:
            items: Items in original order
        
        Returns:
            Scores proportional to input order
        """
        logger.info("Using fallback scores (order-preserving)")
        scores = [
            RerankScore(id=item.id, score=1.0 - (i / len(items)), cached=False)
            for i, item in enumerate(items)
        ]
        return scores


def build_items_for_rerank(filtered: List[FilterResult]) -> List[RerankItem]:
    """
    Build rerank items from filtered results.
    
    Args:
        filtered: Filtered results with chunks
    
    Returns:
        List of RerankItem with text snippets
    """
    items = []
    for result in filtered:
        snippet = build_rerank_snippet(result.chunk, max_chars=800)
        items.append(RerankItem(id=result.chunk.id, text=snippet))
    return items
