# build_router_docs.py
# pip install opensearch-py tqdm
import os, json, sys, collections
from tqdm import tqdm
from opensearchpy import OpenSearch, helpers

CHUNKS_INDEX = "code_chunks_v2"
ROUTER_INDEX = "repo_router_v1"

def agg_from_chunks(client, size=1000):
    # pull minimal fields to synthesize a router doc per repo_id
    src = {"_source": ["repo_id","language","rel_path","module","symbols","tech_stack","path"]}
    # scroll all
    page = client.search(index=CHUNKS_INDEX, body={"size": size, "query": {"match_all": {}}}, scroll="2m", **src)
    sid = page["_scroll_id"]
    while True:
        hits = page["hits"]["hits"]
        if not hits: break
        for h in hits:
            yield h["_source"]
        page = client.scroll(scroll_id=sid, scroll="2m")

def synthesize_router(docs):
    by_repo = collections.defaultdict(lambda: {
        "languages": collections.Counter(),
        "modules": collections.Counter(),
        "symbols": collections.Counter(),
        "paths": collections.Counter()
    })
    for d in docs:
        r = by_repo[d["repo_id"]]
        if "language" in d: r["languages"][d["language"]] += 1
        if "module" in d:   r["modules"][d["module"]] += 1
        for s in d.get("symbols", []): r["symbols"][s] += 1
        if "rel_path" in d: r["paths"][d["rel_path"].split("/")[0]] += 1

    router_docs = []
    for repo_id, agg in by_repo.items():
        langs = [k for k,_ in agg["languages"].most_common(3) if k and k!="unknown"]
        key_modules = [k for k,_ in agg["paths"].most_common(8) if k not in ("", ".", "tests", "test", "__pycache__")]
        key_symbols = [k for k,_ in agg["symbols"].most_common(20)]
        short_title = f"{repo_id} ({', '.join(langs)})" if langs else repo_id
        summary = (
            f"Auto-synthesized router summary for {repo_id}. "
            f"Languages: {', '.join(langs) if langs else 'unknown'}. "
            f"Top modules: {', '.join(key_modules[:5]) if key_modules else 'unknown'}."
        )
        router_docs.append({
            "_op_type": "index",
            "_index": ROUTER_INDEX,
            "_id": repo_id,
            "_source": {
                "repo_id": repo_id,
                "short_title": short_title[:80],
                "summary": summary[:700],
                "languages": langs or ["unknown"],
                "domains": "unknown",         # fill later via LLM if you want
                "key_modules": ", ".join(key_modules),
                "key_symbols": ", ".join(key_symbols),
                "tech_stack": "unknown",
                "entrypoints": "unknown"
            }
        })
    return router_docs

def main(host="http://localhost:9200"):
    client = OpenSearch(hosts=[host])
    docs = list(agg_from_chunks(client))
    actions = synthesize_router(docs)
    ok, fail = helpers.bulk(client, actions, chunk_size=200)
    print(f"Router docs indexed: {ok}, failed: {fail}")

if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9200"
    main(host)