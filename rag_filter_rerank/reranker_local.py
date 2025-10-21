"""
Local Cross-Encoder Reranker using HuggingFace transformers.

Uses BAAI/bge-reranker-v2-m3 by default for quality and speed.
Supports GPU acceleration when available, with CPU fallback.
"""
import hashlib
import logging
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .schemas import RerankItem, RerankScore
from .cache import Cache

logger = logging.getLogger(__name__)


class LocalCrossEncoderReranker:
    """Local cross-encoder reranker using transformers."""
    
    def __init__(self, cache: Cache, settings):
        """
        Initialize local reranker.
        
        Args:
            cache: Cache instance for storing scores
            settings: Configuration settings
        """
        self.cache = cache
        self.settings = settings
        self.model_name = settings.RERANK_MODEL
        
        # Determine device
        device = settings.RERANK_DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model and tokenizer
        logger.info(f"Loading local reranker: {self.model_name} on {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device).eval()
            logger.info(f"✅ Local reranker loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise
    
    @torch.no_grad()
    def rerank(self, query: str, items: List[RerankItem]) -> List[RerankScore]:
        """
        Rerank items using local cross-encoder model.
        
        Args:
            query: Query string
            items: Items to rerank
        
        Returns:
            List of RerankScore ordered by relevance
        """
        if not items:
            return []
        
        # Enforce max docs limit
        max_docs = self.settings.MAX_DOCS_RERANKED
        if len(items) > max_docs:
            logger.warning(f"Truncating {len(items)} items to {max_docs}")
            items = items[:max_docs]
        
        # Check cache and prepare items to score
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        to_run = []
        mapping = []  # [(item, cache_key, original_index)]
        cached_scores = {}
        
        for idx, item in enumerate(items):
            text_hash = hashlib.sha256(item.text.encode()).hexdigest()[:16]
            cache_key = f"rerank:local:{self.model_name}:{query_hash}:{text_hash}"
            
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached_scores[item.id] = float(cached)
            else:
                to_run.append(item)
                mapping.append((item, cache_key, idx))
        
        logger.info(f"Rerank: {len(cached_scores)} cached, {len(to_run)} need scoring")
        
        # Score uncached items in batches
        if to_run:
            batch_size = self.settings.RERANK_BATCH_SIZE
            for batch_start in range(0, len(to_run), batch_size):
                batch_end = min(batch_start + batch_size, len(to_run))
                batch = to_run[batch_start:batch_end]
                
                # Prepare pairs for cross-encoder
                pairs = [(query, item.text) for item in batch]
                
                # Tokenize
                encoded = self.tokenizer(
                    [q for q, _ in pairs],
                    [t for _, t in pairs],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get scores
                logits = self.model(**encoded).logits.squeeze(-1)
                scores = logits.detach().float().cpu().tolist()
                
                # Handle single item case
                if not isinstance(scores, list):
                    scores = [scores]
                
                # Cache and store scores
                for (item, cache_key, _), score in zip(mapping[batch_start:batch_end], scores):
                    score_float = float(score)
                    self.cache.set(cache_key, score_float, ttl_s=self.settings.CACHE_TTL_S)
                    cached_scores[item.id] = score_float
        
        # Build final scored list
        final_scores = []
        for item in items:
            score = cached_scores.get(item.id, 0.0)
            final_scores.append(RerankScore(id=item.id, score=score, cached=item.id in cached_scores and item not in to_run))
        
        # Sort by score descending
        final_scores.sort(key=lambda s: -s.score)
        
        return final_scores


def create_local_reranker(cache: Cache, settings) -> Optional[LocalCrossEncoderReranker]:
    """
    Factory function to create local reranker with error handling.
    
    Args:
        cache: Cache instance
        settings: Configuration settings
    
    Returns:
        LocalCrossEncoderReranker instance or None if model loading fails
    """
    try:
        return LocalCrossEncoderReranker(cache, settings)
    except Exception as e:
        logger.error(f"Could not create local reranker: {e}")
        logger.warning("Reranking will be skipped")
        return None

