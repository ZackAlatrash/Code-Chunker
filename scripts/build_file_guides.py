#!/usr/bin/env python3
import argparse, hashlib, json, time, requests
from collections import defaultdict
from opensearchpy import OpenSearch

CHUNKS_INDEX = "code_chunks_v2"
FILES_INDEX  = "file_guides_v1"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b-instruct"

# --- helpers ---

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def call_ollama_json(system: str, user: str, timeout=300) -> dict:
    """Call Ollama and return parsed JSON object from the model's content.

    - Prefer /api/chat with stream=false.
    - On 404, fall back to /api/generate with stream=false.
    - If server still returns NDJSON stream, parse line-by-line and concatenate.
    - Then try to JSON-decode the assistant content (optionally fenced).
    """

    def parse_chat_payload_text(text: str) -> str:
        # Try single JSON first
        try:
            data = json.loads(text)
            return (data.get("message") or {}).get("content", "")
        except Exception:
            pass
        # NDJSON lines: accumulate message.content
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            msg = (chunk.get("message") or {}).get("content", "")
            if msg:
                out.append(msg)
        return "".join(out)

    def parse_generate_payload_text(text: str) -> str:
        # Try single JSON first
        try:
            data = json.loads(text)
            return data.get("response", "")
        except Exception:
            pass
        # NDJSON lines: accumulate response tokens until done
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            piece = chunk.get("response", "")
            if piece:
                out.append(piece)
        return "".join(out)

    # 1) Try /api/chat with stream=false
    chat_body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    try:
        r = requests.post(OLLAMA_URL, json=chat_body, timeout=timeout)
        r.raise_for_status()
        raw = r.text
        content = parse_chat_payload_text(raw).strip()
    except requests.exceptions.HTTPError as e:
        if getattr(e.response, "status_code", None) != 404:
            raise
        # 2) Fall back to /api/generate (derive URL if needed)
        if OLLAMA_URL.rstrip("/").endswith("/api/chat"):
            gen_url = OLLAMA_URL.rstrip("/")[:-5] + "generate"  # swap 'chat' for 'generate'
        else:
            gen_url = OLLAMA_URL
        gen_body = {
            "model": MODEL,
            "prompt": f"<SYS>\n{system}\n</SYS>\n\n{user}",
            "stream": False,
        }
        r = requests.post(gen_url, json=gen_body, timeout=timeout)
        r.raise_for_status()
        raw = r.text
        content = parse_generate_payload_text(raw).strip()

    # Extract JSON if fenced or surrounded
    start = content.find("{"); end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        content = content[start:end+1]
    try:
        return json.loads(content)
    except Exception:
        return {}

def fetch_chunks_for_repo(client: OpenSearch, repo_id: str, max_files: int | None = None):
    """
    Stream all chunks for repo_id. Yields source dicts.
    """
    q = {
        "size": 500,
        "_source": ["repo_id","rel_path","language","symbols","text","chunk_number","start_line","end_line"],
        "query": {"term": {"repo_id": repo_id}}
    }
    res = client.search(index=CHUNKS_INDEX, body=q, scroll="2m")
    sid = res.get("_scroll_id")
    total = 0
    while True:
        hits = res["hits"]["hits"]
        if not hits: break
        for h in hits:
            yield h["_source"]
        res = client.scroll(scroll_id=sid, scroll="2m")

def build_excerpts(grouped_by_file: dict[str, list[dict]], max_lines=400, max_chars=20000):
    """
    For each file (rel_path), concatenate earliest chunks (by start_line or chunk_number)
    until reaching limits; return excerpt, span_start, span_end.
    """
    out = {}
    for rel_path, items in grouped_by_file.items():
        items = [i for i in items if i.get("text")]
        if not items: 
            out[rel_path] = ("", None, None); continue
        # sort by start_line (fallback chunk_number)
        def key(i):
            sl = i.get("start_line")
            return (sl if sl is not None else 10**9, i.get("chunk_number", 10**9))
        items.sort(key=key)
        lines_acc = []
        span_start = None; span_end = None
        for it in items:
            t = it["text"]
            sl = it.get("start_line"); el = it.get("end_line")
            if span_start is None and sl is not None: span_start = sl
            if el is not None: span_end = el if span_end is None else max(span_end, el)
            # append
            lines_acc.append(t)
            joined = "\n".join(lines_acc)
            # stop if limits
            if len(joined) > max_chars or sum(len(x.splitlines()) for x in lines_acc) > max_lines:
                break
        excerpt = "\n".join(lines_acc)
        out[rel_path] = (excerpt, span_start, span_end)
    return out

def upsert_file_guide(client: OpenSearch, repo_id: str, rel_path: str, language: str,
                      excerpt: str, span_start, span_end, symbols: list[str]):
    now = int(time.time()*1000)
    sha = sha1(excerpt or "")
    doc_id = f"{repo_id}:{rel_path}"
    # Check existing
    try:
        existing = client.get(index=FILES_INDEX, id=doc_id)
        if existing and existing.get("_source", {}).get("sha1_excerpt") == sha:
            print(f"SKIP {rel_path} (unchanged)")
            return
    except Exception:
        pass

    sys_prompt = (
        "You are a read-only summarizer. Only use the provided file excerpt; do not invent facts.\n"
        "Output STRICT JSON with keys: {\"summary\": str, \"top_funcs\": [str], \"symbols\": [str]}.\n"
        "Rules:\n"
        "- summary: 150–250 words; describe what the file does and how it is used.\n"
        f"- Include 2–5 inline citations like [{rel_path}:start-end] for key claims.\n"
        "- top_funcs: list of up to 8 strings in the form 'name (start-end)'.\n"
        "- symbols: up to 20 identifiers present.\n"
        "- If insufficient context, set summary to 'insufficient context'."
    )
    user_prompt = (
        f"Repo: {repo_id}\n"
        f"Path: {rel_path}\n"
        f"Language: {language}\n"
        f"Symbols: {', '.join((symbols or [])[:20])}\n"
        f"Excerpt covers lines {span_start}-{span_end} (may be truncated):\n"
        f"```{language}\n{excerpt}\n```"
    )
    resp = call_ollama_json(sys_prompt, user_prompt)
    summary = (resp.get("summary") or "").strip()
    top_funcs = resp.get("top_funcs") or []
    sym_list = resp.get("symbols") or symbols or []

    body = {
        "repo_id": repo_id,
        "rel_path": rel_path,
        "language": language or "unknown",
        "sha1_excerpt": sha,
        "summary": summary[:4000],
        "top_funcs": ", ".join([str(x) for x in top_funcs][:8]),
        "symbols": ", ".join([str(x) for x in sym_list][:20]),
        "span_start": span_start if span_start is not None else 1,
        "span_end": span_end if span_end is not None else 1,
        "updated_at": now
    }
    client.index(index=FILES_INDEX, id=doc_id, body=body)
    print(f"UPSERT file guide: {rel_path}")

def main():
    ap = argparse.ArgumentParser(description="Generate per-file guides with Qwen (map step).")
    ap.add_argument("--host", default="http://localhost:9200")
    ap.add_argument("--repo-id", default=None, help="Only process this repo_id (default: all repos found in chunks)")
    ap.add_argument("--max-lines", type=int, default=400)
    ap.add_argument("--max-chars", type=int, default=20000)
    args = ap.parse_args()

    client = OpenSearch(hosts=[args.host])

    # Find repo_ids
    repo_ids = []
    if args.repo_id:
        repo_ids = [args.repo_id]
    else:
        agg = {
            "size": 0,
            "aggs": {"repos": {"terms": {"field": "repo_id", "size": 10000}}}
        }
        res = client.search(index=CHUNKS_INDEX, body=agg)
        repo_ids = [b["key"] for b in res["aggregations"]["repos"]["buckets"]]

    for rid in repo_ids:
        print(f"\n=== Repo: {rid} ===")
        files = defaultdict(list)
        for s in fetch_chunks_for_repo(client, rid):
            files[s.get("rel_path") or s.get("path")].append(s)
        excerpts = build_excerpts(files, max_lines=args.max_lines, max_chars=args.max_chars)
        for rel_path, items in files.items():
            # pick language/symbols from first chunk
            lang = (items[0].get("language") or items[0].get("ext") or "unknown")
            syms = []
            for it in items:
                syms.extend(it.get("symbols") or [])
            syms = list(dict.fromkeys(syms))  # dedup preserve order
            excerpt, sstart, send = excerpts[rel_path]
            if not excerpt.strip():
                print(f"SKIP empty excerpt: {rel_path}"); continue
            upsert_file_guide(client, rid, rel_path, lang, excerpt, sstart, send, syms)

if __name__ == "__main__":
    main()
