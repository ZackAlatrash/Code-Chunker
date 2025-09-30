# ========= PROJECT INGESTION PIPELINE Makefile =========
# Usage:
#   make ingest DIR=/path/to/repo REPO_ID=myproject
#   make chunk DIR=/path/to/repo
#   make index REPO_ID=myproject
#   make router
#   make guides

# --- Config ---
PYTHON          ?= python
PYTHONPATH      ?= src
HOST            ?= http://localhost:9200

# File paths
CHUNK_SCRIPT    ?= scripts/newchunker.py
INDEX_SCRIPT    ?= scripts/bulk_index_chunks.py
ROUTER_SCRIPT   ?= scripts/build_router_docs.py
GUIDE_SCRIPT    ?= scripts/build_repo_guides.py

# I/O
DIR             ?= .
REPO_ID         ?= unknown
OUT             ?= run/$(REPO_ID)_chunks.jsonl

# Indices
CHUNKS_INDEX    ?= code_chunks_v2
ROUTER_INDEX    ?= repo_router_v1
GUIDE_INDEX     ?= repo_guide_v1

.PHONY: help ingest chunk index router guides clean dirs

help:
	@echo "Targets for new repo ingestion:"
	@echo "  make ingest DIR=repo REPO_ID=myrepo     # chunk → embed+index → router → guides"
	@echo "  make chunk DIR=repo REPO_ID=myrepo      # only chunk"
	@echo "  make index REPO_ID=myrepo               # only embed+index chunks"
	@echo "  make router                             # update repo router docs"
	@echo "  make guides [REPO_ID=x]                 # update repo guides (all or one)"
	@echo "Variables: HOST, CHUNKS_INDEX, ROUTER_INDEX, GUIDE_INDEX, OUT"

dirs:
	@mkdir -p run

# 1. Chunk the project
chunk: dirs
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(CHUNK_SCRIPT) "$(DIR)" \
	  --out "$(OUT)" \
	  --encoding gpt-4 \
	  --token-limit 1200
	@echo "Chunks written → $(OUT)"

# 2. Index chunks into OpenSearch
index:
	# bulk indexer expects positional args: JSONL then HOST
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(INDEX_SCRIPT) "$(OUT)" "$(HOST)"
	@echo "Chunks bulk-indexed into $(CHUNKS_INDEX) at $(HOST)"

# 3. Update repo router docs (one doc per repo)
router:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(ROUTER_SCRIPT) "$(HOST)"
	@echo "Router docs updated in $(ROUTER_INDEX)"

# 4. Update repo guides (LLM-generated repo overviews)
guides:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(GUIDE_SCRIPT) \
	  --host "$(HOST)" \
	  $(if $(REPO_ID),--repo-id "$(REPO_ID)",)
	@echo "Repo guides updated in $(GUIDE_INDEX)"

# 5. Full ingestion pipeline
ingest: chunk index router guides

clean:
	rm -f run/*.jsonl
