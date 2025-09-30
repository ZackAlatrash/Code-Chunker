#!/usr/bin/env python3
"""
Fast repo-level guide (one LLM call per repo).
- Scans code_chunks_v2
- Builds a compact "summary pack": directory tree, manifests, top files, symbol stats, and short head snippets
- Calls local Qwen once to produce a grounded repo overview JSON
- Upserts into repo_guide_v1
"""

import argparse, json, time, os, re, hashlib, requests
from collections import defaultdict, Counter
from opensearchpy import OpenSearch

CHUNKS_INDEX = "code_chunks_v2"
REPO_GUIDE_INDEX = "repo_guide_v1"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b-instruct"

# ---------- helpers ----------

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def call_ollama_json(system: str, user: str, url=OLLAMA_URL, model=MODEL, timeout=300, retries=3, backoff=1.7) -> dict:
    """
    Ask Ollama to return strict JSON. Retries, expects format=json. Returns {} on failure.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0}
    }
    last_err = None
    for i in range(retries):
        try:
            r = requests.post(url, json=body, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            raw = (data.get("message") or {}).get("content", "") or ""
            return json.loads(raw)
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)
    print(f"[WARN] LLM JSON parse failed: {last_err}")
    return {}

def get_all_repos(client: OpenSearch) -> list[str]:
    agg = {"size": 0, "aggs": {"repos": {"terms": {"field": "repo_id", "size": 10000}}}}
    res = client.search(index=CHUNKS_INDEX, body=agg)
    return [b["key"] for b in res["aggregations"]["repos"]["buckets"]]

def scan_repo_chunks(client: OpenSearch, repo_id: str):
    q = {
        "size": 500,
        "_source": ["repo_id","rel_path","path","language","ext","symbols","text","chunk_number","start_line","end_line","n_tokens"],
        "query": {"term": {"repo_id": repo_id}}
    }
    res = client.search(index=CHUNKS_INDEX, body=q, scroll="2m")
    sid = res.get("_scroll_id")
    while True:
        hits = res["hits"]["hits"]
        if not hits: break
        for h in hits:
            yield h["_source"]
        res = client.scroll(scroll_id=sid, scroll="2m")

def guess_entrypoints(paths: list[str]) -> list[str]:
    pats = [
        r"(^|/)(main\.py|app\.py|server\.py|manage\.py)$",
        r"(^|/)cli/.*\.(py|js|ts)$",
        r"(^|/)bin/[^/]+$",
        r"(^|/)src/main/(java|go|scala)/",
        r"(^|/)index\.(js|ts)$",
    ]
    outs = []
    for p in paths:
        for pat in pats:
            if re.search(pat, p, flags=re.I):
                outs.append(p); break
    return outs[:5]

def build_tree(paths: list[str], depth=2, max_items=50) -> list[str]:
    # directory tree collapsed to depth=2
    tree = defaultdict(set)
    for p in paths:
        parts = p.split("/")
        if len(parts) == 1:
            tree["."].add(parts[0])
        else:
            key = "/".join(parts[:depth])
            tree[key].add(parts[0] if depth==1 else "/".join(parts[:depth+1]))
    lines = []
    cnt = 0
    for k in sorted(tree.keys()):
        kids = sorted(tree[k])
        lines.append(f"{k}/")
        for child in kids[:10]:
            lines.append(f"  - {child}")
            cnt += 1
            if cnt >= max_items:
                lines.append("  ..."); return lines
    return lines

def pick_top_files(files: dict, by="score", k=10):
    # files: rel_path -> stats dict (size, chunks, sym_count, score)
    ranked = sorted(files.items(), key=lambda kv: kv[1].get(by, 0), reverse=True)
    return [rp for rp, _ in ranked[:k]]

def clamp(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[:max_chars] + "\n# …(trimmed)…"

# ---------- main pack builder ----------

def build_summary_pack(repo_id: str, rows: list[dict], max_head_lines=120, max_head_chars=8000) -> dict:
    files = defaultdict(lambda: {"chunks":0,"size":0,"sym_count":0,"lang":None})
    all_paths, all_langs, all_syms = [], [], []
    heads = {}  # rel_path -> head snippet (few earliest chunks concatenated)
    earliest_span = {}  # rel_path -> (start,end)

    for r in rows:
        rel = r.get("rel_path") or r.get("path")
        if not rel: continue
        files[rel]["chunks"] += 1
        t = r.get("text") or ""
        files[rel]["size"] += len(t)
        syms = r.get("symbols") or []
        files[rel]["sym_count"] += len(syms)
        if files[rel]["lang"] is None:
            files[rel]["lang"] = r.get("language") or r.get("ext")
        all_paths.append(rel)
        all_langs.append(files[rel]["lang"])
        all_syms.extend(syms)

    # simple score: size^0.5 + sym_count
    for rel, st in files.items():
        st["score"] = (st["size"] ** 0.5) + st["sym_count"]

    # assemble short heads for top files
    top_for_heads = pick_top_files(files, by="score", k=8)
    # take earliest chunks (by start_line then chunk_number)
    by_file = defaultdict(list)
    for r in rows:
        rel = r.get("rel_path") or r.get("path")
        if rel in top_for_heads:
            by_file[rel].append(r)
    for rel, items in by_file.items():
        items.sort(key=lambda x: ((x.get("start_line") if x.get("start_line") is not None else 10**9), x.get("chunk_number", 10**9)))
        text_acc = []
        start, end = None, None
        line_total = 0
        for it in items:
            code = it.get("text") or ""
            lines = code.splitlines()
            if start is None and it.get("start_line") is not None:
                start = it.get("start_line")
            if it.get("end_line") is not None:
                end = it.get("end_line") if end is None else max(end, it.get("end_line"))
            if line_total + len(lines) > max_head_lines:
                need = max_head_lines - line_total
                code = "\n".join(lines[:max(0,need)])
            text_acc.append(code)
            line_total += len(code.splitlines())
            if line_total >= max_head_lines:
                break
        head = clamp("\n".join(text_acc), max_head_chars)
        heads[rel] = head
        earliest_span[rel] = (start if start is not None else 1, end if end is not None else max_head_lines)

    langs = [l for l in all_langs if l]
    lang_top = [x for x,_ in Counter(langs).most_common(3)]
    modules = [p.split("/")[0] for p in all_paths if "/" in p]
    modules_top = [m for m,_ in Counter(modules).most_common(8)]
    syms_top = [s for s,_ in Counter(all_syms).most_common(30)]

    tree_lines = build_tree(sorted(set(all_paths)), depth=2, max_items=60)

    entrypoints = guess_entrypoints(list(set(all_paths)))

    pack = {
        "repo_id": repo_id,
        "languages": lang_top,
        "modules": modules_top,
        "entrypoints_guess": entrypoints,
        "top_symbols": syms_top,
        "dir_tree": tree_lines,
        "top_files": pick_top_files(files, by="score", k=10),
        "file_heads": [
            {
                "rel_path": rp,
                "span": earliest_span.get(rp, (1, 1)),
                "head": heads.get(rp, "")
            } for rp in pick_top_files(files, by="score", k=6)
        ]
    }
    return pack

def upsert_repo_guide(client: OpenSearch, repo_id: str, pack: dict, args):
    system = (
        "You are a read-only repository summarizer. Use ONLY the provided snippets & stats. "
        "Return a single JSON object with keys: overview (string), key_flows (array of strings), "
        "entrypoints (array of strings), languages (array of strings), modules (array of strings). "
        "Rules: overview is 200–300 words; every statement that refers to code must include an inline "
        "citation like [rel_path:start-end] using the provided spans; key_flows contains 3–8 bullets, "
        "each ending with a citation; entrypoints up to 5; languages/modules from provided stats only. "
        "If evidence is insufficient, say so explicitly in the overview."
    )
    # Construct a compact user prompt
    file_heads = (pack.get("file_heads") or [])[:4]  # at most 4
    lines = []
    lines.append(f"Repo: {repo_id}")
    lines.append("Languages: " + ", ".join(pack.get("languages", [])))
    lines.append("Modules: " + ", ".join(pack.get("modules", [])))
    lines.append("Entrypoints (guesses): " + ", ".join(pack.get("entrypoints_guess", [])))
    lines.append("Top files: " + ", ".join((pack.get("top_files") or [])[:8]))
    # directory tree: clamp to ~20 lines
    tree = (pack.get("dir_tree") or [])[:20]
    if tree:
        lines.append("\nDirectory tree (truncated):\n" + "\n".join(tree))
    lines.append("\nFile heads (use these spans for citations):")
    for fh in file_heads:
        rp = fh.get("rel_path"); s, e = fh.get("span", ("?","?"))
        head = fh.get("head", "")
        head_short = "\n".join(head.splitlines()[:50])  # 50 lines max
        ext = rp.split(".")[-1] if (rp and "." in rp) else ""
        lines.append(f"\n{rp}  (lines {s}-{e})\n```{ext}\n{head_short}\n```")
    user = "\n".join(lines)

    def _valid(js: dict) -> bool:
        return isinstance(js, dict) and bool((js.get("overview") or "").strip())

    js = call_ollama_json(system, user, url=args.ollama_url, model=args.model)

    if not _valid(js):
        print("[WARN] primary guide empty — retrying with compact backup prompt")
        backup_user = (
            f"Repo: {repo_id}\n"
            "Languages: " + ", ".join(pack.get("languages", [])) + "\n"
            "Modules: " + ", ".join(pack.get("modules", [])) + "\n"
            "Entrypoints (guesses): " + ", ".join(pack.get("entrypoints_guess", [])) + "\n"
            "Top files: " + ", ".join((pack.get("top_files") or [])[:6]) + "\n\n"
            "Snippets:\n" + "\n".join(
                (
                    f"{fh['rel_path']} (lines {fh['span'][0]}-{fh['span'][1]})\n```\n"
                    + "\n".join((fh.get('head','').splitlines()[:30])) + "\n```"
                )
                for fh in (pack.get("file_heads") or [])[:3]
            )
        )
        js = call_ollama_json(system, backup_user, url=args.ollama_url, model=args.model)

    if not _valid(js):
        print("[ERROR] Repo guide still empty; skipping upsert.")
        return
    now = int(time.time()*1000)
    body = {
        "repo_id": repo_id,
        "overview": (js.get("overview") or "")[:6000],
        "key_flows": "\n".join(f"- {x}" for x in (js.get("key_flows") or [])[:8]),
        "entrypoints": ", ".join((js.get("entrypoints") or [])[:5]),
        "languages": list({x for x in (js.get("languages") or pack['languages']) if x})[:6],
        "modules": ", ".join((js.get("modules") or pack['modules'])[:10]),
        "updated_at": now,
        # simple cache key so we can skip re-gen next time if pack identical
        "_pack_sha1": sha1(json.dumps(pack, ensure_ascii=False))
    }
    client.index(index=REPO_GUIDE_INDEX, id=repo_id, body=body)
    print(f"UPSERT repo guide: {repo_id}")

def main():
    ap = argparse.ArgumentParser(description="Fast repo-level guides from chunks (one LLM call per repo).")
    ap.add_argument("--host", default="http://localhost:9200")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--max-head-lines", type=int, default=120)
    ap.add_argument("--max-head-chars", type=int, default=8000)
    ap.add_argument("--force", action="store_true", help="Ignore _pack_sha1 cache")
    ap.add_argument("--ollama-url", default=OLLAMA_URL)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    client = OpenSearch(hosts=[args.host])

    repos = [args.repo_id] if args.repo_id else get_all_repos(client)

    for rid in repos:
        print(f"\n=== Repo: {rid} ===")
        rows = list(scan_repo_chunks(client, rid))
        if not rows:
            print("No chunks — skipping"); continue

        pack = build_summary_pack(rid, rows, args.max_head_lines, args.max_head_chars)

        # caching: compare to existing _pack_sha1
        if not args.force:
            try:
                prev = client.get(index=REPO_GUIDE_INDEX, id=rid)
                if prev and prev.get("_source", {}).get("_pack_sha1") == sha1(json.dumps(pack, ensure_ascii=False)):
                    print("No changes detected — skip.")
                    continue
            except Exception:
                pass

        upsert_repo_guide(client, rid, pack, args)

if __name__ == "__main__":
    main()
