# search_v4: Hybrid Search System

Hybrid search for `code_chunks_v4` combining:
- **kNN (semantic)** vector search
- **BM25 (lexical)** keyword search
- **RRF fusion** for combining rankings
- **Symbol-aware reranking** for code relevance
- **Result diversification** to avoid redundancy

---

## 📋 Architecture

```
User Query
    ↓
query_planner.py (existing) → Query Plan
    ↓
search_v4.service.search_v4()
    ↓
    ├─→ kNN retrieval (semantic, 50 candidates)
    ├─→ BM25 retrieval (lexical, 300 candidates)
    ↓
RRF Fusion → Base Scores
    ↓
Symbol-Aware Boosts
    ↓
Diversification (max 3 per file)
    ↓
Top 16 Results (8 with full text)
```

---

## 🚀 Quick Start

### 1. Set Environment Variables

```bash
export OPENSEARCH_URL=http://localhost:9200
export OPENSEARCH_USER=admin
export OPENSEARCH_PASS=admin
export INDEX_NAME=code_chunks_v4
```

### 2. Wire in Embedding Model

Edit `search_v4/embeddings.py` and implement `get_embedding()`:

```python
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str) -> List[float]:
    return _model.encode([text], normalize_embeddings=True)[0].tolist()
```

### 3. Run Search

```bash
# With existing plan
python -m search_v4.cli \
  --query "where is GetForecastForLocation implemented?" \
  --repos foreca \
  --planner-out plan.json

# Let CLI call planner
python -m search_v4.cli \
  --query "how does caching work?" \
  --repos foreca weather \
  --out results.json
```

---

## 📦 Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker |
| `config.py` | Configuration from env vars |
| `opensearch_client.py` | HTTP client wrapper |
| `retrieval.py` | kNN + BM25 queries |
| `rerank.py` | RRF fusion + boosts + diversification |
| `embeddings.py` | **TODO: Wire in your embedding model** |
| `service.py` | Main orchestrator |
| `cli.py` | Command-line interface |

---

## ⚙️ Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENSEARCH_URL` | `http://localhost:9200` | OpenSearch endpoint |
| `OPENSEARCH_USER` | `""` | Username (optional) |
| `OPENSEARCH_PASS` | `""` | Password (optional) |
| `INDEX_NAME` | `code_chunks_v4` | Index to search |
| `KNN_SIZE` | `50` | kNN candidates to retrieve |
| `BM25_SIZE` | `300` | BM25 candidates to retrieve |
| `KNN_NUM_CANDIDATES` | `1000` | HNSW candidates for kNN |
| `RRF_K` | `60` | RRF constant |
| `TOP_K` | `16` | Final results to return |
| `WITH_TEXT_TOP` | `8` | How many get full text |
| `MAX_PER_FILE` | `3` | Max chunks per file |

---

## 🔍 Search Flow

### 1. Query Planning
Uses existing `query_planner.py` to extract:
- `clarified_query`: Cleaned query
- `identifiers`: Symbol names (for boosting)
- `file_hints`: Path hints
- `language`: Language filter
- `bm25_should`: BM25 query terms
- `hyde_passage`: Hypothetical answer (for embedding)

### 2. Parallel Retrieval
- **kNN**: 50 candidates via cosine similarity on `embedding` field
- **BM25**: 300 candidates via keyword matching on `text`, `summary_en`, etc.

### 3. RRF Fusion
Combines rankings using Reciprocal Rank Fusion:
```
score(doc) = Σ 1/(k + rank_i)
```

### 4. Symbol-Aware Boosts
| Condition | Boost |
|-----------|-------|
| `primary_symbol` in identifiers | +0.25 |
| Any `all_symbols` in identifiers | +0.15 |
| Role is `"complete"` | +0.10 |
| Role is `"declaration"` | +0.05 |
| File hint matches `rel_path` | +0.10 |
| Role is `"mixed"` + better exists | -0.05 |

### 5. Diversification
Limit to 3 chunks per file to avoid redundancy.

### 6. Text Fetching
Only top 8 results get full `text` field (to reduce payload size).

---

## 📤 Output Format

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
      "all_symbols": ["GetForecastForLocation", "..."],
      "all_roles": ["complete"],
      "start_line": 55,
      "end_line": 87,
      "summary_en": "Fetches weather forecast...",
      "text": "func (s *Service) GetForecastForLocation(...) { ... }"
    },
    {
      "id": "foreca/service.go#4",
      "repo_id": "foreca",
      "rel_path": "internal/foreca/service.go",
      "primary_symbol": "GetForecastForLocation",
      "all_symbols": ["GetForecastForLocation"],
      "all_roles": ["body"],
      "start_line": 88,
      "end_line": 160,
      "summary_en": "Continuation of forecast logic..."
      // No "text" field (beyond top 8)
    }
  ]
}
```

---

## 🧪 Testing

### Test with Mock Plan

```bash
# Create a mock plan
cat > /tmp/test_plan.json << 'EOF'
{
  "clarified_query": "weather forecast location",
  "identifiers": ["GetForecastForLocation"],
  "file_hints": ["service.go"],
  "language": "go",
  "bm25_should": [
    {"field": "primary_symbol", "term": "GetForecastForLocation", "boost": 3},
    {"field": "summary_en", "term": "forecast", "boost": 2}
  ]
}
EOF

# Run search
python -m search_v4.cli \
  --query "where is GetForecastForLocation?" \
  --repos foreca \
  --planner-out /tmp/test_plan.json \
  --out results.json
```

### Expected Results

Top results should include:
1. ✅ Chunks from `internal/foreca/service.go`
2. ✅ `primary_symbol: "GetForecastForLocation"`
3. ✅ Top 8 have `text` field
4. ✅ Remaining have `summary_en` only
5. ✅ Max 3 chunks per file

---

## 🔧 Troubleshooting

### "NotImplementedError: Wire in your 384-dim embedding model"
→ Edit `search_v4/embeddings.py` and implement `get_embedding()`

### "OpenSearch connection failed"
→ Check `OPENSEARCH_URL`, credentials, and that OpenSearch is running

### "No results returned"
→ Check that:
- Index `code_chunks_v4` exists and has data
- `repo_ids` match indexed documents
- Embedding model dimension is 384

### "Results are not relevant"
→ Try adjusting boosts in `search_v4/rerank.py` or `search_v4/retrieval.py`

---

## 🎯 Next Steps

1. **Wire embedding model** in `embeddings.py`
2. **Test with real queries** using your indexed data
3. **Tune boosts** based on result quality
4. **Integrate with LLM** for Q&A generation

---

## 📚 Related

- `embed_chunks_v4.py` - Indexing script
- `scripts/query_planner.py` - Query planning
- `CHUNK_SCHEMA_V3.md` - Chunk metadata schema

