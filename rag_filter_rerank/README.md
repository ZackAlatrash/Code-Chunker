# RAG Filter-Rerank System (Local-only)

A fully local pipeline for filtering and reranking code chunks using LLM-based relevance scoring and cross-encoder reranking.

## Features

- **🏠 100% Local**: No external APIs or cloud dependencies
- **🎯 Variance Guard**: Automatically preserves retriever quality when filter can't discriminate
- **⚖️ Hybrid Scoring**: Combines filter scores with original retrieval quality
- **🔄 Fallback Logic**: Graceful degradation at every stage
- **💾 Intelligent Caching**: SQLite-based caching for LLM and reranker results
- **⚡ Parallel Processing**: Concurrent LLM filtering for speed

## Architecture

```
Query → Retriever (100 chunks)
      ↓
      Filter Engine (LLM: qwen2.5-coder:7b)
      - Variance Guard (bypass if scores too uniform)
      - Hybrid Scoring (30% filter + 70% retriever)
      - Threshold Relaxation (ensure minimum survivors)
      ↓
      Local Reranker (BAAI/bge-reranker-v2-m3)
      - Cross-encoder scoring
      - Batch processing
      - GPU acceleration (if available)
      ↓
      Top-K Selection (8-10 chunks)
      ↓
      Evidence Builder → Answerer
```

## Components

### 1. Filter Engine (`filter_engine.py`)

**Role**: Score chunks 0-4 for relevance using local LLM

**Key Features**:
- **Variance Guard**: Detects when LLM gives uniform scores and preserves original order
- **Hybrid Scoring**: `final_score = retriever_score + (alpha * filter_score/4)`
- **Threshold Relaxation**: Lowers threshold if too few survivors
- **Anti-Speculation**: Strict prompts prevent hallucination

**Settings**:
```python
OLLAMA_MODEL = "qwen2.5-coder:7b-instruct"  # 7B recommended for quality
FILTER_THRESHOLD = 3                         # Min score to keep (0-4)
FILTER_VARIANCE_BYPASS = 0.15                # Bypass if stddev < 0.15
HYBRID_ALPHA = 0.3                           # 30% filter, 70% retriever
```

### 2. Local Reranker (`reranker_local.py`)

**Role**: Re-rank filtered chunks using cross-encoder model

**Key Features**:
- Uses `BAAI/bge-reranker-v2-m3` (state-of-the-art cross-encoder)
- GPU acceleration (auto-detects CUDA)
- Batch processing for efficiency
- Caching to avoid redundant scoring

**Settings**:
```python
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_DEVICE = "auto"                       # auto|cuda|cpu
RERANK_BATCH_SIZE = 64
MAX_DOCS_RERANKED = 128
```

### 3. Pipeline Orchestrator (`pipeline.py`)

**Role**: Coordinate all stages with proper error handling

**Key Features**:
- Fallback at every stage
- Deterministic output ordering
- Timing metrics
- Cache statistics

## Installation

```bash
# Install dependencies
pip install transformers torch sentence-transformers

# Pull Ollama model (7B recommended)
ollama pull qwen2.5-coder:7b-instruct

# The reranker model will download automatically on first use (~1.5GB)
```

## Configuration

Create `.env` file (see `.env.example`):

```bash
# Ollama Filter
OLLAMA_MODEL=qwen2.5-coder:7b-instruct
FILTER_THRESHOLD=3
FILTER_VARIANCE_BYPASS=0.15
HYBRID_ALPHA=0.3

# Local Reranker
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_DEVICE=auto
MAX_DOCS_RERANKED=128

# Cache
CACHE_DIR=.rag_cache
CACHE_TTL_S=86400
```

## Usage

### Python API

```python
from rag_filter_rerank.pipeline import FilterRerankPipeline
from rag_filter_rerank.config import get_settings
from rag_filter_rerank.retriever import RetrieverAdapter

# Initialize
settings = get_settings()
retriever = RetrieverAdapter(your_search_function)
pipeline = FilterRerankPipeline(retriever, settings)

# Run pipeline
result = pipeline.run("How does authentication work?")

print(f"Answer: {result.answer}")
print(f"Timing: {result.timing_ms}")
print(f"Filtered: {result.filtered_m}/{result.recall_n}")
```

### CLI

```bash
python -m rag_filter_rerank.cli \
  --query "How does authentication work?" \
  --chunks-json chunks.jsonl \
  --trace
```

## Variance Guard Explained

**Problem**: Small LLMs (like 1.5B) often give uniform scores (e.g., all 3/4), making filtering useless.

**Solution**: Compute score variance before filtering:

```python
if stddev(scores) < FILTER_VARIANCE_BYPASS:
    # Filter can't discriminate - preserve retriever order
    return sorted_by_retriever_score(chunks)
else:
    # Filter is working - use hybrid scoring
    return sorted_by_hybrid_score(filtered_chunks)
```

**Result**: Never worse than baseline search!

## Hybrid Scoring Explained

Even when filter works, original retrieval quality matters:

```python
filter_norm = filter_score / 4.0     # Normalize 0-4 to 0-1
retriever_norm = retriever_score     # Already 0-1
hybrid_score = retriever_norm + (alpha * filter_norm)
```

With `alpha=0.3`:
- 70% weight on original retrieval quality
- 30% weight on filter judgment
- Prevents filter from completely overriding good retrieval

## Performance

**Filter Stage** (7B model):
- ~2-3s per chunk (first run)
- ~0.1s per chunk (cached)
- 8 parallel workers = ~3-5s for 16 chunks

**Reranker Stage** (bge-reranker-v2-m3):
- GPU: ~50ms for 64 items
- CPU: ~500ms for 64 items
- Batched processing for efficiency

**Total Pipeline**:
- Cold (no cache): ~10-15s
- Warm (cached): ~1-2s

## Model Choices

### Filter LLM

**Recommended**: `qwen2.5-coder:7b-instruct`
- Best quality/speed balance
- Good at discriminating relevance
- Fits in 8GB VRAM

**Alternative**: `qwen2.5-coder:14b-instruct`
- Better quality
- Requires 16GB+ VRAM
- ~2x slower

**Not Recommended**: `qwen2.5-coder:1.5b-instruct`
- Too small, gives uniform scores
- Will trigger variance guard (which is fine, but why use it?)

### Reranker Model

**Default**: `BAAI/bge-reranker-v2-m3`
- State-of-the-art cross-encoder
- Multilingual support
- Good speed

**Alternative**: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Faster but lower quality
- Good for rapid prototyping

## Testing

```bash
# Run all tests
pytest rag_filter_rerank/tests/

# Test variance guard
pytest rag_filter_rerank/tests/test_variance_guard.py -v

# Test local reranker
pytest rag_filter_rerank/tests/test_reranker_local.py -v

# Test end-to-end pipeline
pytest rag_filter_rerank/tests/test_pipeline_smoke.py -v
```

## Monitoring

The system logs detailed metrics:

```
Score stats: mean=2.85, stddev=0.89
Score distribution: {0: 1, 1: 2, 2: 3, 3: 8, 4: 2}
Filter: 10/16 passed (threshold>=3)
Final: 10 chunks with hybrid scoring (alpha=0.3)
```

Watch for:
- `stddev < 0.15`: Variance guard triggered
- `Score distribution: {3: 16}`: All uniform scores
- `After relaxation`: Threshold was lowered

## Troubleshooting

### Issue: All chunks get same score

**Cause**: LLM model too weak or prompt too permissive

**Solution**: 
1. Upgrade to 7B model: `OLLAMA_MODEL=qwen2.5-coder:7b-instruct`
2. Variance guard will preserve retriever order automatically

### Issue: Reranker model download fails

**Cause**: No internet or HuggingFace offline

**Solution**:
```bash
# Pre-download model
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3'); \
AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')"
```

### Issue: CUDA out of memory

**Cause**: GPU too small for reranker model

**Solution**: Force CPU mode:
```bash
export RERANK_DEVICE=cpu
```

### Issue: Filter too slow

**Cause**: 7B model slow on CPU

**Options**:
1. Use GPU: Much faster
2. Reduce parallelism: `FILTER_PARALLELISM=4`
3. Skip filter: `DISABLE_RERANK=true` (variance guard makes this safe)

## Migration from Cloud Version

If you're upgrading from the Cohere-based version:

1. Remove old env vars:
   ```bash
   # DELETE these from .env
   COHERE_API_KEY=...
   RERANK_VENDOR=cohere
   ```

2. Add new env vars:
   ```bash
   # ADD these to .env
   OLLAMA_MODEL=qwen2.5-coder:7b-instruct
   RERANK_MODEL=BAAI/bge-reranker-v2-m3
   RERANK_DEVICE=auto
   FILTER_VARIANCE_BYPASS=0.15
   HYBRID_ALPHA=0.3
   ```

3. Pull Ollama model:
   ```bash
   ollama pull qwen2.5-coder:7b-instruct
   ```

4. Clear cache (optional):
   ```bash
   rm -rf .rag_cache
   ```

## Credits

- **Filter LLM**: Qwen2.5-Coder by Alibaba
- **Reranker**: BGE models by BAAI (Beijing Academy of Artificial Intelligence)
- **Framework**: HuggingFace Transformers

## License

MIT
