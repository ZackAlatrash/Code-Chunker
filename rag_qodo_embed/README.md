# Qodo Embedding Pipeline

A complete local-only embedding pipeline using `Qodo/Qodo-Embed-1-1.5B` for code chunk retrieval with OpenSearch integration.

## Features

- **Local-only**: No external APIs, everything runs on your machine
- **High-quality embeddings**: Uses Qodo-Embed-1-1.5B (1536 dimensions) optimized for code
- **Query prefixing**: Automatically adds "retrieve relevant code for: " to queries for better alignment
- **Unit normalization**: L2-normalized embeddings for consistent similarity scoring
- **UTF-8 safe truncation**: Handles long code chunks without breaking Unicode
- **Bulk indexing**: Efficient batch processing for large datasets
- **A/B testing**: Compare Qodo against existing embedding models

## Installation

### Prerequisites

- Python 3.8+
- OpenSearch running locally (or accessible via network)
- 4GB+ RAM (for model loading)

### Install Dependencies

```bash
# Core dependencies
pip install sentence-transformers "transformers>=4.39.2" numpy requests pydantic-settings

# Optional: Flash Attention for faster inference (if supported GPU)
pip install flash-attn

# Optional: Progress bars and utilities
pip install tqdm python-dotenv
```

## Quick Start

### 1. Create OpenSearch Index

First, create the index in OpenSearch using the Dev Tools:

```json
PUT /code_chunks_v5_qodo
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "index.knn": true,
    "index.knn.algo_param.ef_search": 100
  },
  "mappings": {
    "properties": {
      "id": {"type": "keyword"},
      "repo_id": {"type": "keyword"},
      "rel_path": {"type": "keyword"},
      "path": {"type": "keyword"},
      "abs_path": {"type": "keyword"},
      "ext": {"type": "keyword"},
      "language": {"type": "keyword"},
      "package": {"type": "keyword"},
      "chunk_number": {"type": "integer"},
      "start_line": {"type": "integer"},
      "end_line": {"type": "integer"},
      "text": {"type": "text"},
      "summary_en": {"type": "text"},
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "nmslib",
          "parameters": {
            "ef_construction": 128,
            "m": 24
          }
        }
      },
      "model_name": {"type": "keyword"},
      "model_version": {"type": "keyword"},
      "embedding_type": {"type": "keyword"},
      "chunk_hash": {"type": "keyword"},
      "created_at": {"type": "date"}
    }
  }
}
```

### 2. Index Chunks

Index your chunk JSON files:

```bash
# Basic indexing
python -m rag_qodo_embed.indexer --chunks-json chunks.json

# With custom settings
python -m rag_qodo_embed.indexer \
  --chunks-json chunks.json \
  --index-name code_chunks_v5_qodo \
  --batch-size 32 \
  --force

# Dry run to see what would be indexed
python -m rag_qodo_embed.indexer --chunks-json chunks.json --dry-run
```

### 3. Search Chunks

Query the indexed chunks:

```bash
# Basic search
python -m rag_qodo_embed.search_cli --query "How does authentication work?"

# With custom settings
python -m rag_qodo_embed.search_cli \
  --query "database connection pooling" \
  --k 10 \
  --index-name code_chunks_v5_qodo \
  --output-format json

# Verbose output
python -m rag_qodo_embed.search_cli --query "error handling" --verbose
```

### 4. A/B Testing

Compare Qodo against existing embeddings:

```bash
# Create queries file
echo "How does authentication work?" > queries.txt
echo "database connection pooling" >> queries.txt
echo "error handling patterns" >> queries.txt

# Run evaluation
python -m rag_qodo_embed.evaluate \
  --queries queries.txt \
  --index-a code_chunks_v4 \
  --index-b code_chunks_v5_qodo \
  --k 10 \
  --output results.json
```

## Configuration

All settings can be configured via environment variables or `.env` file:

```bash
# OpenSearch settings
export OPENSEARCH_URL="http://localhost:9200"
export OPENSEARCH_INDEX="code_chunks_v5_qodo"
export OPENSEARCH_USERNAME="admin"
export OPENSEARCH_PASSWORD="admin"

# Model settings
export MODEL_ID="Qodo/Qodo-Embed-1-1.5B"
export MODEL_CACHE_DIR="/path/to/model/cache"

# Embedding settings
export BATCH_SIZE=16
export TRUNCATE_CHARS=8000
export QUERY_PREFIX="retrieve relevant code for: "
export USE_NORMALIZE=true

# Search settings
export DEFAULT_K=25
export MAX_K=100
```

## API Usage

### Python API

```python
from rag_qodo_embed import QodoEmbedder, QodoOpenSearchClient, get_settings

# Initialize components
settings = get_settings()
embedder = QodoEmbedder(settings)
client = QodoOpenSearchClient(settings)

# Create index
client.create_index(force=True)

# Embed documents
texts = ["func main() { ... }", "class UserService { ... }"]
embeddings = embedder.embed_docs(texts)

# Embed query
query_vec = embedder.embed_query("How does the main function work?")

# Search
results = client.knn_search(query_vec.tolist(), k=10)
```

### Integration with Existing Code

To use Qodo embeddings in your existing retrieval pipeline:

```python
# Set environment variable
import os
os.environ['RETRIEVER_INDEX'] = 'code_chunks_v5_qodo'

# Your existing code will now use the Qodo index
from search_v4.service import search_v4
results = search_v4(query, repo_ids, plan)
```

## File Formats

### Input JSON Format

The indexer supports two JSON formats:

**Array format:**
```json
[
  {
    "id": "file.go#1",
    "repo_id": "my-repo",
    "rel_path": "src/file.go",
    "text": "func main() { ... }",
    "language": "go",
    "start_line": 1,
    "end_line": 10
  }
]
```

**Hits format:**
```json
{
  "hits": [
    {
      "id": "file.go#1",
      "repo_id": "my-repo",
      "rel_path": "src/file.go",
      "text": "func main() { ... }",
      "language": "go",
      "start_line": 1,
      "end_line": 10
    }
  ]
}
```

### Output Format

Search results include:

```json
{
  "id": "file.go#1",
  "repo_id": "my-repo",
  "rel_path": "src/file.go",
  "start_line": 1,
  "end_line": 10,
  "language": "go",
  "summary_en": "Main function that initializes the application",
  "_score": 0.8542
}
```

## Performance

### Model Performance

- **Model size**: ~1.5GB (first download)
- **Embedding dimension**: 1536
- **Inference speed**: ~100-200 docs/second (CPU), ~500-1000 docs/second (GPU)
- **Memory usage**: ~2-4GB RAM during inference

### Indexing Performance

- **Batch size**: 16-32 documents per batch (configurable)
- **Bulk indexing**: 100 documents per OpenSearch batch
- **Typical speed**: 50-100 chunks/second end-to-end

### Search Performance

- **Query latency**: <100ms (after model loaded)
- **KNN search**: <50ms for k=25
- **Cache hit**: <10ms

## Troubleshooting

### Common Issues

**Model download fails:**
```bash
# Set custom cache directory
export MODEL_CACHE_DIR="/path/to/stable/cache"
python -m rag_qodo_embed.indexer --chunks-json chunks.json
```

**OpenSearch connection fails:**
```bash
# Check OpenSearch is running
curl http://localhost:9200/_cluster/health

# Check authentication
export OPENSEARCH_USERNAME="admin"
export OPENSEARCH_PASSWORD="admin"
```

**Out of memory:**
```bash
# Reduce batch size
export BATCH_SIZE=8
python -m rag_qodo_embed.indexer --chunks-json chunks.json
```

**Truncation warnings:**
```bash
# Increase truncation limit
export TRUNCATE_CHARS=12000
python -m rag_qodo_embed.indexer --chunks-json chunks.json
```

### Debug Mode

Enable verbose logging:

```bash
python -m rag_qodo_embed.search_cli --query "test" --verbose
```

## Development

### Running Tests

```bash
# Unit tests (if implemented)
python -m pytest rag_qodo_embed/tests/

# Integration test
python -m rag_qodo_embed.indexer --chunks-json test_chunks.json --dry-run
```

### Adding New Features

1. Add configuration to `config.py`
2. Implement feature in appropriate module
3. Update CLI tools if needed
4. Update README with usage examples

## License

This project is part of the Code Chunker system. See the main project for license information.
