"""
Embedding model interface - integrated with existing repo embedder.

Uses the same sentence-transformers model (all-MiniLM-L6-v2) as embed_chunks_v4.py
Default: 384-dim, L2-normalized vectors
"""
from __future__ import annotations
import importlib
import json
import logging
import math
import os
import threading
from functools import lru_cache
from typing import List, Callable, Any, Optional

import requests

from .config import (
    EMBED_DIM, EMBED_BACKEND, EMBED_MODEL_NAME,
    EMBED_HTTP_URL, EMBED_HTTP_AUTH, REQUEST_TIMEOUT
)

logger = logging.getLogger(__name__)

# --- Utilities ---------------------------------------------------------------

def _l2_normalize(vec: List[float]) -> List[float]:
    """L2 normalize a vector to unit length."""
    s = math.sqrt(sum((x * x) for x in vec)) or 1.0
    return [x / s for x in vec]


def _assert_dim(vec: List[float], where: str) -> None:
    """Validate vector dimension matches expected EMBED_DIM."""
    if len(vec) != EMBED_DIM:
        raise ValueError(
            f"[embeddings] {where} returned dim={len(vec)} but EMBED_DIM={EMBED_DIM}. "
            f"Please fix the embedder or set EMBED_DIM to match."
        )


# --- Scripts backend (sentence-transformers, same as embed_chunks_v4.py) -----

_scripts_lock = threading.Lock()
_scripts_model = None


def _init_scripts_backend() -> None:
    """
    Initialize the sentence-transformers model.
    Uses the same model as embed_chunks_v4.py for consistency.
    """
    global _scripts_model
    if _scripts_model is not None:
        return

    with _scripts_lock:
        if _scripts_model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            
            # Use same model as embed_chunks_v4.py
            model_name = EMBED_MODEL_NAME or "sentence-transformers/all-MiniLM-L6-v2"
            _scripts_model = SentenceTransformer(model_name)
            
            print(f"✅ Loaded embedding model: {model_name} (dim={EMBED_DIM})")
        
        except ImportError as e:
            raise ImportError(
                "[embeddings] sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e
        
        except Exception as e:
            raise RuntimeError(
                f"[embeddings] Failed to load model {EMBED_MODEL_NAME}: {e}"
            ) from e


def _scripts_embed(text: str) -> List[float]:
    """
    Embed text using sentence-transformers model.
    Returns 384-dim L2-normalized vector.
    """
    if _scripts_model is None:
        _init_scripts_backend()
    
    assert _scripts_model is not None
    
    # Encode with normalization (same as embed_chunks_v4.py)
    vec = _scripts_model.encode([text], normalize_embeddings=True)[0]
    vec_list = vec.tolist()
    
    # Validate dimension
    _assert_dim(vec_list, "scripts")
    
    return vec_list


# --- HTTP backend (optional) -------------------------------------------------

def _http_embed(text: str) -> List[float]:
    """
    Embed text via HTTP API endpoint.
    """
    if not EMBED_HTTP_URL:
        raise RuntimeError("EMBED_HTTP_URL is not set for http backend")

    headers = {"Content-Type": "application/json"}
    if EMBED_HTTP_AUTH:
        headers["Authorization"] = EMBED_HTTP_AUTH

    body = {"text": text}
    
    try:
        r = requests.post(
            EMBED_HTTP_URL,
            headers=headers,
            json=body,
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        data = r.json()
        
        # Be permissive about response format
        vec = data.get("embedding") or data.get("vector") or data
        
        if not isinstance(vec, list):
            raise RuntimeError("HTTP embedder returned invalid payload")
        
        vec_list = [float(x) for x in vec]
        _assert_dim(vec_list, "http")
        
        return _l2_normalize(vec_list)
    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP embedding request failed: {e}") from e


# --- Cloud backend (optional) -----------------------------------------------

def _cloud_embed(text: str) -> List[float]:
    """
    Placeholder for cloud embedding services (OpenAI, Cohere, etc.).
    """
    raise NotImplementedError(
        "Cloud embedding not wired. "
        "Set EMBED_BACKEND=scripts (default) or EMBED_BACKEND=http"
    )


# --- Public API --------------------------------------------------------------

@lru_cache(maxsize=256)
def get_embedding(text: str) -> List[float]:
    """
    Return a 384-dim, L2-normalized embedding for text.
    
    Default backend: sentence-transformers (same as embed_chunks_v4.py)
    Model: all-MiniLM-L6-v2
    
    Args:
        text: Input text to embed
    
    Returns:
        384-dim L2-normalized embedding vector
    
    Environment Variables:
        EMBED_BACKEND: "scripts" (default), "http", or "cloud"
        EMBED_MODEL_NAME: Model name (default: "sentence-transformers/all-MiniLM-L6-v2")
        EMBED_HTTP_URL: HTTP API endpoint (if EMBED_BACKEND=http)
        EMBED_HTTP_AUTH: HTTP auth header (optional)
    
    Examples:
        >>> vec = get_embedding("hello world")
        >>> len(vec)
        384
        >>> abs(sum(x**2 for x in vec) - 1.0) < 0.01  # Normalized
        True
    """
    backend = (EMBED_BACKEND or "scripts").lower()
    
    if backend == "scripts":
        return _scripts_embed(text)
    elif backend == "http":
        return _http_embed(text)
    elif backend == "cloud":
        return _cloud_embed(text)
    else:
        raise ValueError(
            f"Unknown EMBED_BACKEND={EMBED_BACKEND}. "
            f"Valid options: scripts, http, cloud"
        )


def get_embedding_for_index(text: str, index_name: Optional[str]) -> List[float]:
    """
    Return an embedding for the given text appropriate for the specified index.
    - If index_name indicates Qodo (e.g., "code_chunks_v5_qodo"), use the local Qodo embedder (1536-dim).
    - Otherwise, fall back to the default 384-dim MiniLM embedding.
    """
    try:
        if index_name and "v5_qodo" in index_name:
            logger.info(f"Using Qodo embedder for index: {index_name}")
            print(f"🔵 [embeddings] Using Qodo embedder for index: {index_name}", flush=True)
            # Defer import to avoid heavy load unless needed
            from rag_qodo_embed.embedder import get_embedder as get_qodo_embedder
            embedder = get_qodo_embedder()
            emb = embedder.embed_query(text)
            # Ensure list[float]
            result = [float(x) for x in emb.tolist()] if hasattr(emb, "tolist") else [float(x) for x in emb]
            logger.info(f"Generated Qodo embedding: dim={len(result)}")
            print(f"✅ [embeddings] Generated Qodo embedding: dim={len(result)}", flush=True)
            return result
    except Exception as e:
        # Fallback to default if Qodo embed fails
        logger.error(f"Qodo embed failed, falling back to default: {e}")
        print(f"❌ [embeddings] Qodo embed failed, falling back to default: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    logger.info(f"Using default MiniLM embedder for index: {index_name or 'default'}")
    print(f"🟡 [embeddings] Using default MiniLM embedder for index: {index_name or 'default'}", flush=True)
    return get_embedding(text)


# --- Self-test ---------------------------------------------------------------

if __name__ == "__main__":
    # Quick self-test
    print("Testing embedding model...")
    print(f"Backend: {EMBED_BACKEND}")
    print(f"Model: {EMBED_MODEL_NAME}")
    print(f"Expected dim: {EMBED_DIM}")
    print()
    
    v = get_embedding("hello world")
    norm = math.sqrt(sum(x*x for x in v))
    
    print(f"✅ Embedding successful!")
    print(f"   Dimension: {len(v)}")
    print(f"   L2 norm: {norm:.6f} (should be ~1.0)")
    print(f"   Sample values: {v[:5]}")
