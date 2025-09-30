# ========= RAG PIPELINE Makefile =========
# Usage examples:
#   make q QUERY="how is chunking handled?"
#   make retrieve QUERY="where is the HTTP server started?" OUT=run/retrieval.json
#   make answer IN=run/retrieval.json MODEL="qwen2.5-coder:7b-instruct-q4_0"
#   make qa QUERY="list public endpoints" FINAL_K=12 SPOTLIGHT=2

# --- Config (override on command line) ---
PYTHON            ?= python
PYTHONPATH        ?= src

# OpenSearch + retrieval
HOST              ?= http://localhost:9200
CHUNKS_INDEX      ?= code_chunks_v2
ROUTER_INDEX      ?= repo_router_v1
REPO_GUIDE_INDEX  ?= repo_guide_v1
FINAL_K           ?= 10
BM25_SIZE         ?= 200
KNN_SIZE          ?= 200
KNN_CANDIDATES    ?= 400
VECTOR_FIELD      ?= vector

# LLM / Answering
MODEL             ?= qwen2.5-coder:7b-instruct
OLLAMA_URL        ?= http://localhost:11434/api/chat
TEMP              ?= 0.0
MAX_LINES         ?= 120
MAX_PROMPT_CHARS  ?= 120000
SPOTLIGHT         ?= 2          # if your answer.py supports spotlight

# I/O
QUERY             ?= how is chunking handled?
OUT               ?= run/retrieval.json
IN                ?= $(OUT)

# Paths to your scripts
SEARCH_SCRIPT     ?= scripts/search_into_json.py
ANSWER_SCRIPT     ?= scripts/answer.py

# --- Helpers ---
RUN_DIR := $(dir $(OUT))

.PHONY: help q retrieve answer qa clean dirs

help:
	@echo "Targets:"
	@echo "  make q QUERY='...'                # retrieve + answer in one go"
	@echo "  make retrieve QUERY='...'         # only retrieval → $(OUT)"
	@echo "  make answer IN=$(OUT)             # only answer (from existing bundle)"
	@echo "  make qa QUERY='...' FINAL_K=12    # tweak knobs"
	@echo "Variables (override as needed): HOST, ROUTER_INDEX, REPO_GUIDE_INDEX, MODEL, OLLAMA_URL, FINAL_K, SPOTLIGHT, etc."

dirs:
	@mkdir -p $(RUN_DIR)

# --- 1) Retrieval: router → hybrid search → bundle (with repo_guides) ---
retrieve: dirs
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SEARCH_SCRIPT) "$(QUERY)" \
	  --host "$(HOST)" \
	  --chunks-index "$(CHUNKS_INDEX)" \
	  --router-index "$(ROUTER_INDEX)" \
	  --repo-guide-index "$(REPO_GUIDE_INDEX)" \
	  --bm25-size $(BM25_SIZE) \
	  --knn-size $(KNN_SIZE) \
	  --knn-candidates $(KNN_CANDIDATES) \
	  --final-k $(FINAL_K) \
	  --vector-field "$(VECTOR_FIELD)" \
	  --out "$(OUT)" \
	  --quiet
	@echo "Wrote bundle → $(OUT)"

# --- 2) Answer: pass bundle to local LLM via Ollama ---
answer:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(ANSWER_SCRIPT) \
	  --in "$(IN)" \
	  --model "$(MODEL)" \
	  --ollama-url "$(OLLAMA_URL)" \
	  --temperature $(TEMP) \
	  --max-lines-per-chunk $(MAX_LINES) \
	  --max-prompt-chars $(MAX_PROMPT_CHARS) \
	  --spotlight $(SPOTLIGHT)

# --- 3) One-shot: retrieve + answer ---
q: retrieve answer

# --- Convenience alias (same as q, but explicit) ---
qa: q

clean:
	@rm -f run/*.json run/*.jsonl 2>/dev/null || true
