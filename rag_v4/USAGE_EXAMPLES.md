# RAG v4 Answerer - Usage Examples

## Quick Start

### 1. Test Evidence Formatting (No LLM Call)

```bash
# Extract sample chunks and view the evidence format
python rag_v4/test_answerer.py
python rag_v4/test_evidence_format.py
```

### 2. Answer with Pre-saved Chunks (Requires Ollama)

```bash
# Make sure Ollama is running:
# ollama serve

# Pull the model if you haven't already:
# ollama pull qwen2.5-coder:7b-instruct

# Run the answerer
python -m rag_v4.answerer \
  "What does the GetForecastForLocation method do?" \
  --chunks-json rag_v4/test_chunks.json \
  --k 5
```

### 3. Answer with Live Search (If search_v4 is available)

```bash
python -m rag_v4.answerer \
  "How does the weather service handle caching?" \
  --k 8
```

## Advanced Usage

### Using a Different Model

```bash
# Use a larger model
python -m rag_v4.answerer \
  "Explain the retry logic in the API client" \
  --chunks-json rag_v4/test_chunks.json \
  --model "qwen2.5-coder:32b-instruct"
```

### Adjusting Context Size

```bash
# Get more chunks for broader context
python -m rag_v4.answerer \
  "How does authentication work across the codebase?" \
  --chunks-json rag_v4/test_chunks.json \
  --k 15
```

### Custom Ollama Endpoint

```bash
# Use a remote Ollama instance
python -m rag_v4.answerer \
  "What are the main entry points?" \
  --chunks-json rag_v4/test_chunks.json \
  --ollama-url "http://remote-server:11434/api/chat"
```

### Longer Timeout for Complex Questions

```bash
# Increase timeout for deep analysis
python -m rag_v4.answerer \
  "Analyze the error handling patterns across all services" \
  --chunks-json rag_v4/test_chunks.json \
  --k 12 \
  --timeout 300
```

## Creating Test Chunks

### From JSONL Files

```python
#!/usr/bin/env python3
import json

# Extract chunks matching a query
def extract_chunks_by_symbol(jsonl_path, symbol_pattern, max_chunks=10):
    chunks = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            chunk = json.loads(line)
            if symbol_pattern.lower() in chunk.get('primary_symbol', '').lower():
                chunks.append(chunk)
                if len(chunks) >= max_chunks:
                    break
    return chunks

# Save for testing
chunks = extract_chunks_by_symbol(
    'ChunksV3/weather_foreca_proxy_service_v3_enriched.jsonl',
    'forecast',
    max_chunks=8
)

with open('rag_v4/forecast_chunks.json', 'w') as f:
    json.dump(chunks, f, indent=2)
```

### From Search Results

If you have `search_v4` set up:

```python
from search_v4.service import search_v4
import json

# Perform a search
results = search_v4(
    query="weather forecast caching",
    router_repo_ids=["weather_foreca_proxy_service"],
    plan={"clarified_query": "weather forecast caching"}
)

# Extract just the chunks
chunks = results.get("results", [])

# Save for testing
with open('rag_v4/search_results.json', 'w') as f:
    json.dump(chunks, f, indent=2)
```

## Environment Setup

### Option 1: Use Environment Variables

```bash
export OLLAMA_URL="http://localhost:11434/api/chat"
export OLLAMA_MODEL="qwen2.5-coder:7b-instruct"

python -m rag_v4.answerer "Your question here" --chunks-json chunks.json
```

### Option 2: Use CLI Arguments

```bash
python -m rag_v4.answerer \
  "Your question here" \
  --chunks-json chunks.json \
  --model "qwen2.5-coder:7b-instruct" \
  --ollama-url "http://localhost:11434/api/chat"
```

## Expected Output Format

The answerer will produce output with citations like:

```
Based on the Evidence:

The GetForecastForLocation method [1] is the main entry point for fetching 
weather forecasts. It implements a caching strategy [1, L67-89] that checks 
Redis before making external API calls.

The method signature (service.go [1], L45):
```go
func (s *Service) GetForecastForLocation(ctx context.Context, loc Location) (*Forecast, error)
```

Key behaviors:
1. Cache lookup with TTL [1, L52-58]
2. External API call if cache miss [1, L60-75]
3. Error handling with fallback [2, L120-135]

If you need more details about the external API integration, we would need 
the foreca.Client implementation.
```

## Troubleshooting

### "No chunks found"

Make sure you either:
- Provide `--chunks-json` with a valid JSON file, OR
- Have `search_v4/search.py` with `search_chunks_v4()` function

### "Connection refused" (Ollama)

```bash
# Start Ollama
ollama serve

# In another terminal, verify it's running
curl http://localhost:11434/api/tags
```

### "Model not found"

```bash
# Pull the model
ollama pull qwen2.5-coder:7b-instruct

# List available models
ollama list
```

### Timeout Errors

For complex questions or large evidence blocks:
```bash
python -m rag_v4.answerer "question" --timeout 300
```

## Integration with Existing Tools

### With Query Planner

```bash
# Use query planner output to guide retrieval
python scripts/query_planner.py "How does caching work?" > plan.json

# Then use the semantic query for retrieval
QUERY=$(jq -r '.clarified_query' plan.json)
python -m rag_v4.answerer "$QUERY" --k 10
```

### With Bulk Analysis

```bash
# Answer multiple questions
for q in "What is the main entry point?" \
         "How is caching implemented?" \
         "What error handling exists?"; do
  echo "=== Question: $q ==="
  python -m rag_v4.answerer "$q" --chunks-json chunks.json
  echo
done
```

## Performance Tips

1. **Reuse chunks**: Save search results to JSON to avoid re-querying
2. **Batch questions**: Use the same chunk set for related questions
3. **Model selection**: Use 7b for speed, 32b for accuracy
4. **Context size**: Start with k=8, increase if answers lack detail

## Next Steps

- [ ] Add OpenAI provider support
- [ ] Implement streaming responses
- [ ] Add conversation history for follow-ups
- [ ] Create web UI wrapper
- [ ] Add metrics/logging

