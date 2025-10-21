"""
Qodo Embedding Pipeline for Code Retrieval

A complete local-only embedding pipeline using Qodo/Qodo-Embed-1-1.5B
for code chunk retrieval with OpenSearch integration.

Components:
- embedder: SentenceTransformers wrapper with query prefixing
- opensearch_client: Bulk indexing and KNN search
- indexer: CLI for indexing chunk JSONs
- search_cli: CLI for querying the index
- evaluate: A/B testing between embedding models
"""

__version__ = "1.0.0"
__author__ = "Code Chunker Team"

from .config import get_settings, Settings
from .embedder import QodoEmbedder
from .opensearch_client import QodoOpenSearchClient
from .truncation import truncate_text

__all__ = [
    "get_settings",
    "Settings", 
    "QodoEmbedder",
    "QodoOpenSearchClient",
    "truncate_text"
]
