# ========= PROJECT INGESTION PIPELINE (v3) =========
# Usage:
#   make -f ingest_v3.mk ingest DIR=/path/to/repo REPO_ID=myproject
#   make -f ingest_v3.mk chunk  DIR=/path/to/repo REPO_ID=myproject
#   make -f ingest_v3.mk index  REPO_ID=myproject
#   make -f ingest_v3.mk router
#   make -f ingest_v3.mk guides REPO_ID=myproject
#
# Notes:
# - This Makefile uses the new symbol-aware chunker (build_chunks_v2.py)
# - Chunks go to index: code_chunks_v3
# - Router docs go to: repo_router_v2

# --- Config ---
PYTHON          ?= python
PYTHONPATH      ?= src
HOST            ?= http://localhost:9200

# Scripts
CHUNK_SCRIPT     ?= scripts/build_chunks_v2.py
INDEX_SCRIPT_V3  ?= scripts/bulk_index_chunks_v3.py
ROUTER_SCRIPT    ?= scripts/build_router_v2.py
GUIDE_SCRIPT     ?= scripts/build_repo_guides.py
# I/O
DIR             ?= .
REPO_ID         ?= unknown
OUT             ?= run/$(REPO_ID)_chunks_v3.jsonl

# Indices (v3/v2)
CHUNKS_INDEX    ?= code_chunks_v3
ROUTER_INDEX    ?= repo_router_v2
GUIDE_INDEX     ?= repo_guide_v1

# Extra args (tweak as needed)
TOKEN_LIMIT     ?= 200
ENCODING        ?= gpt-4

# If your bulk indexer supports --index, set this; otherwise leave empty and it will fall back.
INDEXER_ARGS    ?= --index $(CHUNKS_INDEX)

.PHONY: help ingest chunk index router guides clean dirs create-chunks-index

help:
	@echo "Targets for new repo ingestion (v3):"
	@echo "  make -f ingest_v3.mk ingest DIR=repo REPO_ID=myrepo     # chunk → index → router → guides"
	@echo "  make -f ingest_v3.mk chunk  DIR=repo REPO_ID=myrepo     # only chunk (writes JSONL)"
	@echo "  make -f ingest_v3.mk index  REPO_ID=myrepo              # only index JSONL into $(CHUNKS_INDEX)"
	@echo "  make -f ingest_v3.mk router                             # update router v2 (1 doc per repo)"
	@echo "  make -f ingest_v3.mk guides [REPO_ID=x]                 # update repo guides (all or one)"
	@echo "Vars: HOST, CHUNKS_INDEX, ROUTER_INDEX, GUIDE_INDEX, OUT, TOKEN_LIMIT, ENCODING"

dirs:
	@mkdir -p run

# 0. (Optional) Create the v3 chunks index with mapping before first use
#    If you don't have the index yet, run: make -f ingest_v3.mk create-chunks-index
create-chunks-index:
	@echo "Creating $(CHUNKS_INDEX) (requires OpenSearch _create mapping in your Dev Tools or a curl script)."
	@echo "If you need a ready JSON mapping, ping me — or paste the mapping we discussed for code_chunks_v3."

# 1. Chunk the project (NEW chunker)
chunk: dirs
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(CHUNK_SCRIPT) "$(DIR)" \
	  --out "$(OUT)" \
	  --encoding $(ENCODING) \
	  --token-limit $(TOKEN_LIMIT)
	@echo "✅ Chunks written → $(OUT)"

# 2. Index chunks into OpenSearch (v3)
#    If your bulk indexer doesn't support --index, set INDEXER_ARGS= and ensure it targets $(CHUNKS_INDEX) internally.
index:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(INDEX_SCRIPT_V3) \
	  --index $(CHUNKS_INDEX) \
	  "$(OUT)" "$(HOST)"
	@echo "✅ Chunks bulk-indexed into $(CHUNKS_INDEX) at $(HOST)"
# 3. Update router docs (v2, one doc per repo; auto-cached by pack hash)
router:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(ROUTER_SCRIPT) --host "$(HOST)"
	@echo "✅ Router v2 docs updated in $(ROUTER_INDEX)"

# 4. Update repo guides (LLM-generated repo overviews)
guides:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(GUIDE_SCRIPT) \
	  --host "$(HOST)" \
	  $(if $(REPO_ID),--repo-id "$(REPO_ID)",)
	@echo "✅ Repo guides updated in $(GUIDE_INDEX)"

# 5. Full ingestion pipeline (new chunking → index → router v2 → guides)
ingest: chunk index router guides

clean:
	rm -f run/*_chunks_v3.jsonl