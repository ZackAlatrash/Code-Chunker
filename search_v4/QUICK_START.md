# search_v4 Quick Start Guide

**Status:** ✅ Ready to Use (Embedding Model Integrated)

---

## 🚀 One-Command Test

```bash
# Set environment
export OPENSEARCH_URL=http://localhost:9200
export INDEX_NAME=code_chunks_v4

# Run search
python -m search_v4.cli \
  --query "where is GetForecastForLocation?" \
  --repos foreca \
  --out results.json
```

---

## ✅ What's Ready

- [x] **Embedding model** - Using sentence-transformers (all-MiniLM-L6-v2)
- [x] **kNN search** - 384-dim cosine similarity
- [x] **BM25 search** - Multi-field keyword matching
- [x] **RRF fusion** - Combines semantic + lexical
- [x] **Symbol boosting** - Code-aware reranking
- [x] **Diversification** - Max 3 chunks per file
- [x] **LLM-ready output** - Top 8 with text, rest with summaries

---

## 📋 Prerequisites

### 1. Dependencies Installed

```bash
pip install requests sentence-transformers
```

### 2. OpenSearch Running

```bash
# Check if OpenSearch is running
curl http://localhost:9200

# Should return version info
```

### 3. Index Created & Populated

```bash
# Create index (if not exists)
# See embed_chunks_v4.py for indexing

# Verify index has data
curl http://localhost:9200/code_chunks_v4/_count
```

---

## 🎯 Usage Examples

### Example 1: Basic Search

```bash
python -m search_v4.cli \
  --query "weather forecast implementation" \
  --repos foreca
```

### Example 2: With Existing Query Plan

```bash
# Generate plan first
python scripts/query_planner.py \
  "where is GetForecastForLocation?" \
  --out plan.json

# Use the plan
python -m search_v4.cli \
  --query "where is GetForecastForLocation?" \
  --repos foreca \
  --planner-out plan.json \
  --out results.json
```

### Example 3: Multi-Repo Search

```bash
python -m search_v4.cli \
  --query "error handling in aviation" \
  --repos aviation weather foreca \
  --out results.json
```

---

## 📊 Expected Output

```json
{
  "query": "where is GetForecastForLocation?",
  "router_repo_ids": ["foreca"],
  "plan": { ... },
  "results": [
    {
      "id": "foreca/service.go#3",
      "repo_id": "foreca",
      "rel_path": "internal/foreca/service.go",
      "primary_symbol": "GetForecastForLocation",
      "all_symbols": ["GetForecastForLocation"],
      "all_roles": ["complete"],
      "start_line": 55,
      "end_line": 87,
      "summary_en": "Fetches weather forecast data for a location",
      "text": "func (s *Service) GetForecastForLocation(...) { ... }"
    },
    ...
  ]
}
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Required
export OPENSEARCH_URL=http://localhost:9200
export INDEX_NAME=code_chunks_v4

# Optional (defaults shown)
export KNN_SIZE=50               # kNN candidates
export BM25_SIZE=300             # BM25 candidates
export TOP_K=16                  # Final results
export WITH_TEXT_TOP=8           # Results with full text
export MAX_PER_FILE=3            # Diversification limit

# Embedding (uses defaults, no config needed)
export EMBED_BACKEND=scripts     # scripts | http | cloud
export EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

---

## 🧪 Verify It Works

### Test 1: Embedding Model

```bash
python -m search_v4.embeddings
# Expected: ✅ Loaded model, dim=384, norm≈1.0
```

### Test 2: OpenSearch Connection

```bash
python -c "
from search_v4.opensearch_client import search
result = search('code_chunks_v4', {'size': 1, 'query': {'match_all': {}}})
print(f'✅ Connected! Found {result[\"hits\"][\"total\"][\"value\"]} docs')
"
```

### Test 3: End-to-End Search

```bash
python -m search_v4.cli \
  --query "test query" \
  --repos foreca \
  --out test_results.json

# Check output
cat test_results.json | python -m json.tool | head -20
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'sentence_transformers'"

```bash
pip install sentence-transformers
```

### "Connection refused to localhost:9200"

OpenSearch is not running. Start it:

```bash
docker-compose -f ops/docker-compose.yml up -d
```

### "Index not found: code_chunks_v4"

Index doesn't exist or is empty:

```bash
# Index your chunks
python embed_chunks_v4.py \
  --chunks-dir ./ChunksV3 \
  --index code_chunks_v4
```

### "No results returned"

- Check repo_ids match indexed documents
- Verify chunks exist for that repo
- Try broader query

---

## 📈 Performance Tips

### 1. Tune Result Counts

```bash
# More candidates = better recall, slower
export KNN_SIZE=100
export BM25_SIZE=500

# Fewer final results = faster
export TOP_K=10
```

### 2. Adjust Diversification

```bash
# More per file = less diverse, more complete coverage
export MAX_PER_FILE=5
```

### 3. Use Query Planning

Pre-generate plans for common queries to avoid LLM call overhead.

---

## 📚 Next Steps

1. **Test with your repos** - Index your code and run searches
2. **Tune boosts** - Adjust weights in `search_v4/rerank.py`
3. **Integrate with LLM** - Feed results to code Q&A system
4. **Add monitoring** - Track search quality metrics
5. **Scale up** - Add caching, load balancing

---

## 🆘 Getting Help

- **Documentation**: `search_v4/README.md`
- **Implementation**: `SEARCH_V4_IMPLEMENTATION.md`
- **Embedding**: `EMBEDDING_INTEGRATION_COMPLETE.md`
- **Schema**: `CHUNK_SCHEMA_V3.md`

---

**Ready to search!** 🚀
