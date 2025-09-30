# upsert_router_doc.py
# pip install opensearch-py
import sys, collections
from opensearchpy import OpenSearch

CHUNKS = "code_chunks_v2"
ROUTER = "repo_router_v1"

STOP = {"", ".", "tests", "test", "__pycache__", "node_modules", "build", "dist"}

def main(repo_id, host="http://localhost:9200"):
    client = OpenSearch(hosts=[host])

    # Pull minimal fields for this repo only
    q = {
        "size": 1000,
        "_source": ["repo_id","language","rel_path","symbols"],
        "query": {"term": {"repo_id": repo_id}}
    }

    # Scroll all docs for this repo
    res = client.search(index=CHUNKS, body=q, scroll="2m")
    sid = res.get("_scroll_id")
    langs = collections.Counter()
    mods  = collections.Counter()
    syms  = collections.Counter()

    def ingest_hits(hits):
        for h in hits:
            src = h["_source"]
            if (lng := src.get("language")): langs[lng] += 1
            rel = (src.get("rel_path") or src.get("path") or "")
            top = (rel.split("/", 1)[0] if "/" in rel else rel).strip()
            if top and top not in STOP: mods[top] += 1
            for s in src.get("symbols", []): syms[s] += 1

    ingest_hits(res["hits"]["hits"])
    while True:
        hits = res["hits"]["hits"]
        if not hits: break
        res = client.scroll(scroll_id=sid, scroll="2m")
        if not res["hits"]["hits"]: break
        ingest_hits(res["hits"]["hits"])

    languages = [k for k,_ in langs.most_common(3) if k and k != "unknown"]
    key_modules = [k for k,_ in mods.most_common(8)]
    key_symbols = [k for k,_ in syms.most_common(20)]

    short_title = (f"{repo_id} ({', '.join(languages)})" if languages else repo_id)[:80]
    summary = (
        f"Auto-synthesized router summary for {repo_id}. "
        f"Languages: {', '.join(languages) if languages else 'unknown'}. "
        f"Top modules: {', '.join(key_modules[:5]) if key_modules else 'unknown'}."
    )[:700]

    doc = {
        "repo_id": repo_id,
        "short_title": short_title,
        "summary": summary,
        "languages": languages or ["unknown"],
        "domains": "unknown",
        "key_modules": ", ".join(key_modules),
        "key_symbols": ", ".join(key_symbols),
        "tech_stack": "unknown",
        "entrypoints": "unknown"
    }

    client.index(index=ROUTER, id=repo_id, body=doc)
    print(f"Upserted router doc for {repo_id}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upsert_router_doc.py <repo_id> [http://localhost:9200]")
        sys.exit(1)
    host = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9200"
    main(sys.argv[1], host)