"""
RAG Filter-Rerank Package

A two-stage relevance filtering system for RAG:
1. Stage 1: LLM Filter (Ollama local) - score and filter chunks
2. Stage 2: Hosted Reranker (Cohere) - cross-encoder rerank
3. Evidence generation and answering

All components are pluggable and cache-backed for performance.
"""

__version__ = "1.0.0"

from .config import get_settings
from .schemas import Chunk, FilterResult, RerankScore
from .pipeline import FilterRerankPipeline

__all__ = [
    "get_settings",
    "Chunk",
    "FilterResult",
    "RerankScore",
    "FilterRerankPipeline",
]

