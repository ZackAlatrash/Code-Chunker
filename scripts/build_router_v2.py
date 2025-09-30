#!/usr/bin/env python3
"""
build_router_v2.py
------------------
LLM-enriched repo router builder (one document per repo) targeting the
`repo_router_v2` index.

Key upgrades:
- Compact "summary pack" from code_chunks_v2 (languages/modules/top files/symbols + short heads)
- Single Ollama LLM call per repo with format="json" and robust parsing
- Strong term cleaning (drops generic test/boilerplate words)
- Field caps and validation (won't upsert empty summaries)
- Heuristic fallback when LLM fails (never leaves the router empty)
- Change detection via _pack_sha1 to avoid unnecessary re-generation

Usage:
  python scripts/build_router_v2.py                    # all repos
  python scripts/build_router_v2.py --repo-id myrepo   # single repo
  python scripts/build_router_v2.py --force            # rebuild even if unchanged

You can tune model/url:
  --model qwen2.5-coder:7b-instruct
  --ollama-url http://localhost:11434/api/chat
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List

import requests
from opensearchpy import OpenSearch

# --- Indices ---
CHUNKS_INDEX = "code_chunks_v2"
ROUTER_INDEX = "repo_router_v2"

# --- LLM ---
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5-coder:7b-instruct"

# --- Caps / knobs ---
MAX_TOP_FILES = 12           # important_files list
MAX_HEADS = 4                # how many heads to show LLM
HEAD_LINES = 50              # lines per head sent to LLM
MAX_SYMBOLS = 60             # top symbols to show LLM
MAX_KEYWORDS_OUT = 120       # cap terms joined as comma string
MAX_SUMMARY_CHARS = 6000

# --- Cleaning (stopwords & junky terms often seen in tests) ---
STOP = set("""
the a an and or for from with by of to in on at as is are be was were been being do does did done doing have has had having
class def self return true false none null test tests setup teardown assert mock fixture case suite spider parse request
response http https name method future import python pytest unittest twis ted lxml docs extras example sample readme
file files code src app utils helper base main init __init__ config
""".split())

WORD = re.compile(r"[A-Za-z0-9_+.-]{2,}")

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def trunc(s: str, n: int) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    return s if len(s) <= n else s[:n] + "…"

def clean_terms_csv(v: Any, max_items: int = MAX_KEYWORDS_OUT) -> str:
    """
    Accepts either a string (comma/semicolon-separated) or a list of strings.
    Returns a cleaned, de-duplicated comma-separated string of discriminative terms.
    """
    # Normalize to a single string with commas
    if isinstance(v, list):
        s = ", ".join(str(x).strip() for x in v if str(x).strip())
    elif isinstance(v, str):
        s = v
    else:
        s = ""

    if not s:
        return ""

    parts = re.split(r"[;,]\s*", s)
    out, seen = [], set()
    for part in parts:
        t = part.strip()
        if not t:
            continue
        toks = [w for w in WORD.findall(t.lower())]
        if not toks:
            continue
        # keep if at least one token is meaningful
        if all(tok in STOP or len(tok) < 3 for tok in toks):
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_items:
            break
    return ", ".join(out)

def normalize_list(v: Any, max_items: int | None = None) -> List[str]:
    if isinstance(v, list):
        items = [str(x).strip() for x in v if str(x).strip()]
    elif isinstance(v, str):
        items = [s.strip() for s in re.split(r"[;,]\s*", v) if s.strip()]
    else:
        items = []
    if max_items:
        items = items[:max_items]
    # dedupe preserving order
    seen, out = set(), []
    for it in items:
        low = it.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(it)
    return out

# --------- OpenSearch helpers ---------

def get_all_repos(client: OpenSearch) -> List[str]:
    res = client.search(
        index=CHUNKS_INDEX,
        body={"size": 0, "aggs": {"repos": {"terms": {"field": "repo_id", "size": 10000}}}},
    )
    return [b["key"] for b in res["aggregations"]["repos"]["buckets"]]

def scan_repo_chunks(client: OpenSearch, repo_id: str):
    q = {
        "size": 500,
        "_source": [
            "repo_id",
            "rel_path",
            "path",
            "language",
            "ext",
            "symbols",
            "text",
            "chunk_number",
            "start_line",
            "end_line",
            "n_tokens",
        ],
        "query": {"term": {"repo_id": repo_id}},
    }
    res = client.search(index=CHUNKS_INDEX, body=q, scroll="2m")
    sid = res.get("_scroll_id")
    while True:
        hits = res["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            yield h["_source"]
        res = client.scroll(scroll_id=sid, scroll="2m")

# --------- Pack builder (repo evidence) ---------

def build_pack(rows: List[Dict[str, Any]], include_tests: bool = False) -> Dict[str, Any]:
    """
    Build a compact 'summary pack' used to prompt the LLM. We compute:
    - languages, modules, top symbols
    - important_files (ranked by size^0.5 + sym_count), optionally skipping tests
    - up to MAX_HEADS short heads for the strongest files
    """
    files = defaultdict(lambda: {"size": 0, "sym_count": 0, "lang": None})
    syms_all: List[str] = []
    paths = set()
    by_file = defaultdict(list)

    for r in rows:
        rel = r.get("rel_path") or r.get("path")
        if not rel:
            continue
        by_file[rel].append(r)
        t = r.get("text") or ""
        files[rel]["size"] += len(t)
        syms = r.get("symbols") or []
        files[rel]["sym_count"] += len(syms)
        syms_all.extend(syms)
        if files[rel]["lang"] is None:
            files[rel]["lang"] = r.get("language") or r.get("ext")
        paths.add(rel)

    # score files
    for rel, st in files.items():
        st["score"] = (st["size"] ** 0.5) + st["sym_count"]

    def is_test_path(p: str) -> bool:
        low = p.lower()
        return any(seg in low for seg in ("/tests/", "/test/", "test_", "_test.", "/__tests__/"))

    ranked = sorted(files.items(), key=lambda kv: kv[1]["score"], reverse=True)
    important_files = []
    for rel, _st in ranked:
        if not include_tests and is_test_path(rel):
            continue
        important_files.append(rel)
        if len(important_files) >= MAX_TOP_FILES:
            break

    # heads for a few top files (earliest chunks)
    heads = []
    for rel in important_files[:MAX_HEADS]:
        items = by_file[rel]
        items.sort(
            key=lambda x: (
                x.get("start_line") if x.get("start_line") is not None else 10**9,
                x.get("chunk_number", 10**9),
            )
        )
        total, parts, s_min, e_max = 0, [], None, None
        for it in items:
            code = it.get("text") or ""
            sl, el = it.get("start_line"), it.get("end_line")
            if s_min is None and sl is not None:
                s_min = sl
            if el is not None:
                e_max = el if e_max is None else max(e_max, el)
            lines = code.splitlines()
            need = HEAD_LINES - total
            if need <= 0:
                break
            parts.append("\n".join(lines[:max(0, need)]))
            total += min(len(lines), need)
        heads.append(
            {
                "rel_path": rel,
                "span": [s_min or 1, e_max or HEAD_LINES],
                "head": "\n".join(parts),
            }
        )

    langs = [files[f]["lang"] for f in files if files[f]["lang"]]
    modules = [p.split("/")[0] for p in paths if "/" in p]

    pack = {
        "languages": [x for x, _ in Counter(langs).most_common(5)],
        "modules": [m for m, _ in Counter(modules).most_common(10)],
        "top_symbols": [x for x, _ in Counter(syms_all).most_common(MAX_SYMBOLS)],
        "important_files": important_files,
        "file_heads": heads,
    }
    return pack

# --------- LLM ----------

def call_llm_json(system: str, user: str, url: str, model: str, timeout=90, retries=2) -> Dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "format": "json",          # force JSON
        "options": {"temperature": 0},
    }
    last = None
    for _ in range(retries):
        try:
            r = requests.post(url, json=body, timeout=timeout)
            r.raise_for_status()
            raw = (r.json().get("message") or {}).get("content", "{}")
            return json.loads(raw)
        except Exception as e:
            last = e
    print(f"[WARN] LLM JSON failed: {last}")
    return {}

def llm_router_doc(repo_id: str, pack: Dict[str, Any], url: str, model: str) -> Dict[str, Any]:
    # Tight, discriminative instruction
    system = (
        "You write a compact ROUTER PROFILE for a code repository.\n"
        "Return STRICT JSON with keys: {short_title, summary, languages, tech_stack, modules, important_files, "
        "key_symbols, keywords, synonyms, sample_queries}.\n"
        "Rules:\n"
        "- summary: 120–200 words. Be specific and discriminative; avoid test/boilerplate chatter.\n"
        "- languages/modules: derive only from provided inputs.\n"
        "- important_files: paths from input; pick the most central ones.\n"
        "- key_symbols: function/class names or API names (comma-separated).\n"
        "- keywords: 20–60 comma-separated phrases real developers would type; avoid generic words "
        "  like 'test', 'assert', 'setup', 'teardown'. Prefer distinctive phrases (e.g., 'token limit', 'tree-sitter').\n"
        "- synonyms: alternative phrasings for the same concepts.\n"
        "- sample_queries: 5–10 queries (semicolon-separated) that would retrieve this repo.\n"
        "Do NOT invent facts not supported by the provided heads."
    )
    # Compact pack → user prompt
    lines = []
    lines.append(f"Repo: {repo_id}")
    lines.append("Languages: " + ", ".join(pack.get("languages", [])))
    lines.append("Modules: " + ", ".join(pack.get("modules", [])))
    lines.append("Important files: " + ", ".join(pack.get("important_files", [])))
    lines.append("Top symbols: " + ", ".join(pack.get("top_symbols", [])[:40]))
    lines.append("\nFile heads (use for evidence; keep claims grounded):")
    for fh in pack.get("file_heads", [])[:MAX_HEADS]:
        rp = fh["rel_path"]
        s, e = fh["span"]
        head_short = "\n".join((fh.get("head") or "").splitlines()[:HEAD_LINES])
        ext = rp.split(".")[-1] if "." in rp else ""
        lines.append(f"\n{rp} (lines {s}-{e})\n```{ext}\n{head_short}\n```")
    user = "\n".join(lines)
    js = call_llm_json(system, user, url, model)
    return js or {}

# --------- Heuristic fallback ----------

def heuristic_router_doc(repo_id: str, pack: Dict[str, Any]) -> Dict[str, Any]:
    kws = list(
        dict.fromkeys(
            pack.get("top_symbols", []) + pack.get("modules", []) + [f.split("/")[-1] for f in pack.get("important_files", [])]
        )
    )
    return {
        "short_title": repo_id.replace("-", " ").replace("_", " ").title(),
        "summary": "Auto-generated router profile (heuristic).",
        "languages": pack.get("languages", []),
        "tech_stack": [],
        "modules": pack.get("modules", []),
        "important_files": pack.get("important_files", []),
        "key_symbols": ", ".join(pack.get("top_symbols", [])[:MAX_SYMBOLS]),
        "keywords": ", ".join(kws[:MAX_KEYWORDS_OUT]),
        "synonyms": "",
        "sample_queries": "how to run; where is main; how does chunking work; config file location; entrypoints",
    }

# --------- Upsert w/ validation & cleaning ----------

def upsert_router_doc(client: OpenSearch, repo_id: str, payload: Dict[str, Any], pack_sha: str):
    # normalize fields
    short_title = trunc((payload.get("short_title") or repo_id).strip(), 200)
    summary = trunc((payload.get("summary") or "").strip(), MAX_SUMMARY_CHARS)
    if not summary:
        print(f"[WARN] Empty summary for {repo_id}; skipping upsert.")
        return

    languages = normalize_list(payload.get("languages"), max_items=8)
    tech_stack = normalize_list(payload.get("tech_stack"), max_items=20)
    modules = normalize_list(payload.get("modules"), max_items=20)
    important_files = normalize_list(payload.get("important_files"), max_items=MAX_TOP_FILES)

    key_symbols = clean_terms_csv(payload.get("key_symbols", ""))
    keywords = clean_terms_csv(payload.get("keywords", ""))
    synonyms = clean_terms_csv(payload.get("synonyms", ""))
    # sample queries can remain semicolon or comma separated
    sample_queries = "; ".join(normalize_list(payload.get("sample_queries"), max_items=12))

    body = {
        "repo_id": repo_id,
        "short_title": short_title,
        "summary": summary,
        "languages": languages,
        "tech_stack": tech_stack,
        "modules": modules,
        "important_files": important_files,
        "key_symbols": key_symbols,
        "keywords": keywords,
        "synonyms": synonyms,
        "sample_queries": sample_queries,
        "updated_at": int(time.time() * 1000),
        "_pack_sha1": pack_sha,
    }
    client.index(index=ROUTER_INDEX, id=repo_id, body=body)
    print(f"UPSERT router_v2: {repo_id} (len(summary)={len(summary)})")

# --------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Build LLM-enriched router docs per repo (v2).")
    ap.add_argument("--host", default="http://localhost:9200")
    ap.add_argument("--repo-id", default=None, help="If omitted, process all repos in code_chunks_v2")
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--force", action="store_true", help="Rebuild even if _pack_sha1 unchanged")
    ap.add_argument("--include-tests", action="store_true", help="Allow test files in important_files & heads")
    args = ap.parse_args()

    client = OpenSearch(hosts=[args.host])

    # collect repos
    repos = [args.repo_id] if args.repo_id else get_all_repos(client)
    if not repos:
        print("No repos found in code_chunks_v2.")
        return

    for rid in repos:
        print(f"\n=== Repo: {rid} ===")
        rows = list(scan_repo_chunks(client, rid))
        if not rows:
            print("No chunks; skip.")
            continue

        pack = build_pack(rows, include_tests=args.include_tests)
        pack_sha = sha1(json.dumps(pack, ensure_ascii=False))

        if not args.force:
            try:
                prev = client.get(index=ROUTER_INDEX, id=rid, ignore=[404])
                if prev and prev.get("found") and prev["_source"].get("_pack_sha1") == pack_sha:
                    print("No changes detected — skip.")
                    continue
            except Exception:
                pass

        # LLM attempt
        js = llm_router_doc(rid, pack, args.ollama_url, args.model)
        if not js or not (js.get("summary") or "").strip():
            print("[WARN] LLM empty/failed — using heuristic router.")
            js = heuristic_router_doc(rid, pack)

        try:
            upsert_router_doc(client, rid, js, pack_sha)
        except Exception as e:
            print(f"[ERROR] upsert failed for {rid}: {e}")

if __name__ == "__main__":
    main()