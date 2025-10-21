"""
Configuration for search_v4.

All settings can be overridden via environment variables.
"""
import os

# OpenSearch connection
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS", "")

# Index configuration
INDEX_NAME = os.getenv("INDEX_NAME", "code_chunks_v4")
RETRIEVER_INDEX = os.getenv("RETRIEVER_INDEX", INDEX_NAME)  # Allows switching to Qodo index

# kNN parameters
KNN_K = int(os.getenv("KNN_K", "50"))
KNN_NUM_CANDIDATES = int(os.getenv("KNN_NUM_CANDIDATES", "1000"))
KNN_SIZE = int(os.getenv("KNN_SIZE", "50"))

# BM25 parameters
BM25_SIZE = int(os.getenv("BM25_SIZE", "300"))

# Fusion parameters
RRF_K = int(os.getenv("RRF_K", "60"))

# Result diversification
MAX_PER_FILE = int(os.getenv("MAX_PER_FILE", "3"))
TOP_K = int(os.getenv("TOP_K", "16"))
WITH_TEXT_TOP = int(os.getenv("WITH_TEXT_TOP", "8"))

# Request timeout (seconds)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Embedding configuration
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "scripts")  # scripts | http | cloud
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_HTTP_URL = os.getenv("EMBED_HTTP_URL", "")       # optional
EMBED_HTTP_AUTH = os.getenv("EMBED_HTTP_AUTH", "")     # optional, e.g. "Bearer xxx"

