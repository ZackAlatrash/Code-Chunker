# RAG v4 Answerer

A minimal RAG answerer that formats retrieved v4 code chunks into a single "Evidence" block and sends them to a local LLM for question answering.

## Features

- **Generous code context**: Up to ~10k chars per chunk (no aggressive trimming)
- **Strict message structure**: System → User → Assistant (Evidence) → User (plan prompt) → Assistant (answer)
- **Evidence format**: Stable indexed headers `[1]`, `[2]`, ... with file/line anchors
- **Citation-aware**: Assistant is instructed to cite evidence using `[n]` notation
- **Flexible input**: Works with live search or pre-saved JSON chunks

## Usage

### With Live Search

If `search_v4/search.py` exists and provides `search_chunks_v4()`:

```bash
python -m rag_v4.answerer "Why do throttled requests return stale forecasts?"
```

### With Pre-saved Chunks

```bash
python -m rag_v4.answerer "Explain cache freshness" --chunks-json tmp/hits.json
```

The JSON file can be either:
- A plain array: `[{chunk1}, {chunk2}, ...]`
- An object with hits: `{"hits": [{chunk1}, {chunk2}, ...]}`

## Command-Line Arguments

- `question` (required): The developer question to answer
- `--k INT`: Number of top chunks to include (default: 8)
- `--chunks-json PATH`: Path to JSON file with chunks (optional)
- `--model STR`: Ollama model name (default: `qwen2.5-coder:7b-instruct`)
- `--ollama-url URL`: Ollama API endpoint (default: `http://localhost:11434/api/chat`)
- `--timeout INT`: Request timeout in seconds (default: 180)

## Environment Variables

- `OLLAMA_URL`: Override default Ollama endpoint
- `OLLAMA_MODEL`: Override default model

## Evidence Format

Each chunk is rendered with a stable header:

```
[{i}] {repo_id} | {rel_path} | L{start}–{end} | {primary_symbol} ({primary_kind}/{roles})
summary: {summary_en}
code:
```{language}
{text}
```
```

Example:

```
[1] weather-service | internal/foreca/service.go | L45–87 | GetForecastForLocation (method/complete)
summary: Fetches weather forecast for a given location, with caching and fallback logic.
code:
```go
func (s *Service) GetForecastForLocation(ctx context.Context, loc Location) (*Forecast, error) {
    // ... implementation ...
}
```
```

## Requirements

- Python 3.7+
- `requests` library
- Running Ollama instance (or compatible API)
- Optional: `search_v4/search.py` with `search_chunks_v4()` function

## Integration Notes

- **Token budget**: Intentionally generous (12 chunks × 10k chars). Increase if your model supports it.
- **Citations**: The assistant is prompted to use `[n]` notation. Keep header format stable.
- **Extensibility**: To add OpenAI support, implement `call_openai()` and add a `--provider` flag.

## Example Session

```bash
$ python -m rag_v4.answerer "How does the cache invalidation work?" --k 6

Based on the Evidence:

The cache invalidation strategy [1] uses a TTL-based approach combined with 
explicit invalidation hooks [2][3]. When a forecast is fetched (L67-89 in 
service.go [1]), the cache key includes both location and timestamp.

Invalidation occurs in two scenarios:
1. TTL expiry: 5 minutes for normal forecasts [1, L72]
2. Manual flush: via the /admin/flush endpoint [3, L120-135]

The implementation in cache.go [2] uses Redis EXPIRE commands...
```

## Acceptance Criteria

✅ `python -m rag_v4.answerer "question"` runs and prints an answer with citations  
✅ Works with either live retrieval or `--chunks-json`  
✅ Evidence headers match the specified format  
✅ Assistant cites chunks using `[1]`, `[2]`, etc.  
✅ Up to 10k chars per chunk are included (generous context)

