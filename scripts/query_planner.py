#!/usr/bin/env python3
"""
Query Planner for RAG (standalone)
- Input: raw user query (+ optional repo guides JSON)
- Output: structured plan JSON with clarified query, symbols, file hints, boosts, HyDE passage
- Uses local LLM via Ollama (format=json) with robust fallback to heuristics
"""

import argparse, json, os, re, sys, requests
from typing import Any, Dict, List

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5-coder:7b-instruct"

# ---------- Heuristics (used as defaults and fallback) ----------

IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
FILE_RE  = re.compile(r"[\w./-]+\.(py|js|ts|tsx|go|rb|php|java|scala|rs|cpp|c)\b")
LANG_HINTS = {
    "python": ["python", "py", "fastapi", "pydantic", "pytest", "pip", "requirements.txt"],
    "typescript": ["typescript", "ts", "tsx", "vite", "react", "nextjs", "nuxt", "deno"],
    "javascript": ["javascript", "js", "node", "express", "yarn", "npm", "package.json"],
    "go": ["golang", "go", "gin", "go.mod", "go.sum"],
    "java": ["java", "maven", "gradle", "spring"],
}

def guess_language(text: str) -> str:
    t = text.lower()
    best, bestc = "", 0
    for lang, keys in LANG_HINTS.items():
        c = sum(1 for k in keys if k in t)
        if c > bestc:
            best, bestc = lang, c
    return best if bestc > 0 else ""

def heuristic_plan(query: str, repo_guides: List[Dict[str, Any]]) -> Dict[str, Any]:
    ids = list({m.group(0) for m in IDENT_RE.finditer(query) if len(m.group(0)) >= 3})
    files = list({m.group(0) for m in FILE_RE.finditer(query)})
    neg = ["*test*"] if "test" not in query.lower() else []
    lang = guess_language(query)
    should = []
    for sym in ids[:6]:
        should.append({"field": "primary_symbol", "term": sym, "boost": 8})
        should.append({"field": "symbols", "term": sym, "boost": 6})
    for fh in files[:4]:
        should.append({"field": "rel_path", "term": fh, "boost": 3})
    if lang:
        should.append({"field": "language", "term": lang, "boost": 2})
    router_terms = [w for w in re.findall(r"[a-zA-Z]+", query.lower()) if len(w) >= 4][:6]
    hyde = f"The repository likely defines components related to: {', '.join(ids[:4] or router_terms)}. " \
           f"This question asks: {query.strip()}. Provide the definition and usage sites, including file paths and line ranges."
    return {
        "clarified_query": query.strip(),
        "language": lang,
        "identifiers": ids[:10],
        "keywords": [w for w in router_terms if w not in ids][:10],
        "file_hints": files[:8],
        "negative_hints": neg,
        "bm25_should": should[:12],
        "bm25_must": [{"field": "language", "term": lang, "exact": True}] if lang else [],
        "router_terms": (ids[:3] or router_terms[:6]),
        "hyde_passage": hyde
    }

# ---------- LLM caller ----------

def call_ollama_plan(query: str, repo_guides: List[Dict[str, Any]], url: str, model: str, timeout: int = 60) -> Dict[str, Any]:
    # Build a tiny orientation context from repo guides (optional)
    guide_snips = []
    for g in (repo_guides or [])[:2]:
        ov = (g.get("overview") or "")[:800]
        guide_snips.append(f"- {g.get('repo_id')}: {ov}")
    guide_ctx = "\n".join(guide_snips)

    system = (
        "You are a Query Planner for code search. Return STRICT JSON with keys: "
        "{clarified_query, language, identifiers, keywords, file_hints, negative_hints, "
        "bm25_should, bm25_must, router_terms, hyde_passage}. Rules: "
        "- identifiers: likely function/class/variable names (exact strings) "
        "- file_hints: likely filenames/paths "
        "- language: one of python/typescript/javascript/go/java or '' if unclear "
        "- bm25_should/must: items with fields among symbols, primary_symbol, rel_path, text, language "
        "- negative_hints: globs to avoid (e.g., '*test*') "
        "- router_terms: 3–6 unigrams for repo routing "
        "- hyde_passage: 3–6 sentence hypothetical answer for semantic search embedding "
        "Output JSON ONLY."
    )
    user = f"""User query: {query}

Optional repo overview excerpts:
{guide_ctx}
"""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0}
    }
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    raw = (r.json().get("message") or {}).get("content", "{}")
    return json.loads(raw)

def merge_plans(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(fallback)
    for k, v in (primary or {}).items():
        if isinstance(v, str) and v.strip():
            out[k] = v
        elif isinstance(v, list) and v:
            out[k] = v
        elif isinstance(v, dict) and v:
            out[k] = v
    return out

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Standalone LLM Query Planner for RAG.")
    ap.add_argument("query", help="Raw user query")
    ap.add_argument("--repo-guides", help="Path to JSON file (array) with repo guides (optional)", default=None)
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--no-llm", action="store_true", help="Force heuristic planner only")
    ap.add_argument("--out", default="", help="Write plan JSON to file; default prints to stdout")
    args = ap.parse_args()

    guides = []
    if args.repo_guides and os.path.exists(args.repo_guides):
        try:
            with open(args.repo_guides, "r", encoding="utf-8") as f:
                data = json.load(f)
                # accept either {"repo_guides":[...]} or [...]
                guides = data.get("repo_guides", data) if isinstance(data, dict) else data
        except Exception:
            guides = []

    base = heuristic_plan(args.query, guides)

    plan = base
    if not args.no_llm:
        try:
            llm = call_ollama_plan(args.query, guides, args.ollama_url, args.model, args.timeout)
            plan = merge_plans(llm, base)
        except Exception as e:
            # keep heuristic silently; print a hint to stderr
            print(f"[planner] LLM failed, using heuristics: {e}", file=sys.stderr)

    # light post-processing defaults
    defval = lambda x, d: x if x else d
    plan["clarified_query"] = defval(plan.get("clarified_query","").strip(), args.query.strip())
    plan["identifiers"] = defval(plan.get("identifiers"), [])
    plan["keywords"] = defval(plan.get("keywords"), [])
    plan["file_hints"] = defval(plan.get("file_hints"), [])
    plan["negative_hints"] = defval(plan.get("negative_hints"), [])
    plan["bm25_should"] = defval(plan.get("bm25_should"), [])
    plan["bm25_must"] = defval(plan.get("bm25_must"), [])
    plan["router_terms"] = defval(plan.get("router_terms"), [])
    plan["hyde_passage"] = defval(plan.get("hyde_passage","").strip(), "")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print(f"Wrote plan → {args.out}")
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()