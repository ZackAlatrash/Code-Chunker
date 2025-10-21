"""
Qodo Embedder using SentenceTransformers.

Provides local-only embedding using Qodo/Qodo-Embed-1-1.5B model
with query prefixing and unit normalization for code retrieval.
"""
import logging
import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import torch

from .config import Settings
from .truncation import truncate_text

logger = logging.getLogger(__name__)

# Global singleton instance
_global_embedder: Optional['QodoEmbedder'] = None


def get_embedder(settings: Optional[Settings] = None) -> 'QodoEmbedder':
    """
    Get or create the global Qodo embedder instance.
    
    Args:
        settings: Configuration settings (uses default if None)
        
    Returns:
        Singleton QodoEmbedder instance
    """
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = QodoEmbedder(settings)
    return _global_embedder


class QodoEmbedder:
    """Qodo embedding wrapper with query prefixing and normalization."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Qodo embedder.
        
        Args:
            settings: Configuration settings (uses default if None)
        """
        self.settings = settings or Settings()
        self.model: Optional[SentenceTransformer] = None
        self._model_loaded = False
        
    def _load_model(self) -> None:
        """Lazy load the SentenceTransformer model."""
        if self._model_loaded:
            return
            
        logger.info(f"Loading Qodo model: {self.settings.MODEL_ID}")
        try:
            # Set cache directory if specified
            cache_dir = self.settings.MODEL_CACHE_DIR
            
            self.model = SentenceTransformer(
                self.settings.MODEL_ID,
                cache_folder=cache_dir
            )
            
            # Set to eval mode for inference
            self.model.eval()
            
            # Log model info
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   - Model: {self.settings.MODEL_ID}")
            logger.info(f"   - Device: {self.model.device}")
            logger.info(f"   - Max sequence length: {self.model.max_seq_length}")
            
            self._model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load model {self.settings.MODEL_ID}: {e}")
            raise
    
    def embed_docs(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of documents.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            Normalized embeddings as numpy array (n_docs, 1536)
        """
        if not texts:
            return np.array([])
        
        self._load_model()
        
        # Truncate texts if needed
        truncated_texts = [
            truncate_text(text, self.settings.TRUNCATE_CHARS)
            for text in texts
        ]
        
        logger.info(f"Embedding {len(texts)} documents (batch_size={self.settings.BATCH_SIZE})")
        
        # Encode in batches
        embeddings = self.model.encode(
            truncated_texts,
            batch_size=self.settings.BATCH_SIZE,
            normalize_embeddings=False,  # We'll normalize manually
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Manual L2 normalization for explicit control
        if self.settings.USE_NORMALIZE:
            embeddings = self._normalize_embeddings(embeddings)

        # Warn but do not drop rows to preserve 1:1 alignment
        row_norms = np.linalg.norm(embeddings, axis=1) if embeddings.ndim == 2 else np.array([np.linalg.norm(embeddings)])
        near_zero = int((row_norms < 1e-8).sum())
        if near_zero:
            logger.warning(f"{near_zero} embeddings are near-zero; keeping them to preserve alignment")
        
        logger.info(f"✅ Generated embeddings: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query with prefix.
        
        Args:
            query: Query string
            
        Returns:
            Normalized query embedding as numpy array (1536,)
        """
        self._load_model()
        
        # Apply query prefix for better retrieval alignment
        prefixed_query = f"{self.settings.QUERY_PREFIX}{query}"
        
        logger.debug(f"Embedding query: '{prefixed_query[:100]}...'")
        
        # Encode single query
        embedding = self.model.encode(
            [prefixed_query],
            normalize_embeddings=False,
            convert_to_numpy=True
        )[0]  # Get single embedding
        
        # Manual L2 normalization
        if self.settings.USE_NORMALIZE:
            embedding = self._normalize_embeddings(embedding.reshape(1, -1))[0]

        # Warn if near-zero
        norm = float(np.linalg.norm(embedding))
        if norm < 1e-8:
            logger.warning("Query embedding is near-zero; keeping to preserve alignment")
        
        logger.debug(f"✅ Generated query embedding: {embedding.shape}")
        return embedding
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings to unit vectors.
        
        Args:
            embeddings: Raw embeddings (n_docs, dim) or (dim,)
            
        Returns:
            L2 normalized embeddings
        """
        # Handle both 1D and 2D arrays
        if embeddings.ndim == 1:
            norm = np.linalg.norm(embeddings)
            if norm > 0:
                return embeddings / norm
            return embeddings
        
        # 2D array: normalize each row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
        return embeddings / norms
    
    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            Embedding dimension (1536 for Qodo-Embed-1-1.5B)
        """
        self._load_model()
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        self._load_model()
        return {
            "model_id": self.settings.MODEL_ID,
            "embedding_dim": self.get_embedding_dim(),
            "max_seq_length": self.model.max_seq_length,
            "device": str(self.model.device),
            "normalize": self.settings.USE_NORMALIZE,
            "query_prefix": self.settings.QUERY_PREFIX,
            "truncate_chars": self.settings.TRUNCATE_CHARS
        }
