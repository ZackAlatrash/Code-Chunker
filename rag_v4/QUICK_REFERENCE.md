# RAG v4 Answerer - Quick Reference

## 🚀 Quick Start

```bash
# 1. Test without LLM (verify Evidence format)
python rag_v4/test_evidence_format.py

# 2. Answer with test chunks (requires Ollama)
python -m rag_v4.answerer \
  "What does the main function do?" \
  --chunks-json rag_v4/test_chunks.json
```

## 📋 Command Syntax

```bash
python -m rag_v4.answerer QUESTION [OPTIONS]
```

### Required
- `QUESTION`: Your code question in quotes

### Options
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--k` | int | 8 | Number of chunks to include |
| `--chunks-json` | path | - | JSON file with chunks |
| `--model` | str | `qwen2.5-coder:7b-instruct` | Ollama model |
| `--ollama-url` | url | `http://localhost:11434/api/chat` | API endpoint |
| `--timeout` | int | 180 | Request timeout (seconds) |

## 🔧 Environment Variables

```bash
export OLLAMA_URL="http://localhost:11434/api/chat"
export OLLAMA_MODEL="qwen2.5-coder:7b-instruct"
```

## 📝 Evidence Format

```
[{i}] {repo_id} | {rel_path} | L{start}–{end} | {symbol} ({kind}/{roles})
summary: {one-line summary}
code:
```{language}
{code up to 10k chars}
```
```

## 🧪 Testing Workflow

```bash
# 1. Extract sample chunks
python rag_v4/test_answerer.py

# 2. Verify Evidence format (no LLM)
python rag_v4/test_evidence_format.py

# 3. Test with LLM
python -m rag_v4.answerer \
  "How does caching work?" \
  --chunks-json rag_v4/test_chunks.json \
  --k 5
```

## 💡 Common Use Cases

### Simple Question
```bash
python -m rag_v4.answerer \
  "What is the main entry point?" \
  --chunks-json chunks.json
```

### Deep Dive (More Context)
```bash
python -m rag_v4.answerer \
  "Explain the entire authentication flow" \
  --chunks-json chunks.json \
  --k 12
```

### Use Larger Model
```bash
python -m rag_v4.answerer \
  "Compare error handling across services" \
  --chunks-json chunks.json \
  --model "qwen2.5-coder:32b-instruct" \
  --timeout 300
```

### Live Search (If Available)
```bash
python -m rag_v4.answerer \
  "How does the cache invalidate?"
  # No --chunks-json, uses search_v4.search
```

## 🐛 Troubleshooting

### "No chunks found"
- Provide `--chunks-json` with valid JSON file
- OR ensure `search_v4/search.py` exists

### "Connection refused"
```bash
# Start Ollama
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

### "Model not found"
```bash
ollama pull qwen2.5-coder:7b-instruct
ollama list
```

### Timeout Errors
```bash
# Increase timeout for complex questions
python -m rag_v4.answerer "question" --timeout 300
```

## 📊 Expected Output

```
Based on the Evidence:

The GetForecastForLocation method [1] is the main entry point for weather 
forecasts. It implements:

1. Cache lookup [1, L52-58] with 5-minute TTL
2. External API call on cache miss [1, L60-75]
3. Error handling with fallback [2, L120-135]

Implementation (service.go [1], L45):
```go
func (s *Service) GetForecastForLocation(ctx context.Context, loc Location) (*Forecast, error) {
    // Check cache first...
}
```

Gaps: Need foreca.Client implementation for full API integration details.
```

**Note:** Look for `[1]`, `[2]` citations in the answer!

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | User documentation |
| `USAGE_EXAMPLES.md` | Detailed examples |
| `IMPLEMENTATION_SUMMARY.md` | Architecture & design |
| `QUICK_REFERENCE.md` | This file |

## 🔗 Integration

### Create Test Chunks from JSONL

```python
import json

chunks = []
with open('ChunksV3/weather_service.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 10: break
        chunks.append(json.loads(line))

with open('test_chunks.json', 'w') as f:
    json.dump(chunks, f, indent=2)
```

### Extract Specific Symbols

```python
# Get all chunks related to "Forecast"
chunks = []
with open('ChunksV3/weather_service.jsonl', 'r') as f:
    for line in f:
        chunk = json.loads(line)
        if 'forecast' in chunk.get('primary_symbol', '').lower():
            chunks.append(chunk)

with open('forecast_chunks.json', 'w') as f:
    json.dump(chunks, f)
```

## 🎯 Best Practices

1. **Start small**: Use `--k 5` first, increase if needed
2. **Test format**: Run `test_evidence_format.py` before LLM calls
3. **Save results**: Reuse `--chunks-json` for related questions
4. **Check citations**: Good answers cite `[1]`, `[2]`, etc.
5. **Model choice**: 7b for speed, 32b for accuracy

## ⚡ Performance Tips

| Context Size | k | Chars/Chunk | LLM Time (7b) |
|--------------|---|-------------|---------------|
| Small        | 5 | 2k          | 3-5s          |
| Medium       | 8 | 5k          | 5-8s          |
| Large        | 12 | 10k         | 10-15s        |

---

**For full details:** See `README.md` and `IMPLEMENTATION_SUMMARY.md`

