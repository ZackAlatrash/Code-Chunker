#!/usr/bin/env python3
# answers.py — compact bundle -> better-ordered prompt -> Qwen via Ollama
# Adds "Spotlight" (must-see) snippets requested from the model + heuristic fallback.

import argparse
import json
import os
import re
import sys
import textwrap
import requests
from typing import List, Dict, Any

DEFAULT_MODEL = "qwen2.5-coder:7b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MAX_PROMPT_CHARS = 120_000

# === Developer-focused, long-form system prompt ===
READ_ONLY_SYSTEM = """You are a READ-ONLY code assistant for developers.

NON-NEGOTIABLE RULES:
- Answer ONLY the exact user question. Do NOT reinterpret the topic (e.g., "authentication" ≠ "rate limiting").
- Use ONLY the provided code snippets under 'Sources' as evidence.
- If the snippets do NOT clearly contain content about the asked topic, reply exactly:
  Not found in provided sources.
- Every factual claim must end with a citation like [repo_id|rel_path:start-end].
- Include 1–3 short code excerpts (≤ 15 lines each) only if they directly support the claim.
- Keep answers concise and structured (bullets/headings)."""

# === Few-shot: shows structure + multiple snippets ===
CITATION_FEWSHOT = """Example style:

Summary
- The HTTP server is started by the CLI entrypoint and wired to route handlers. [weather-api|src/cli/main.py:12-40]

Relevant files & roles
- src/cli/main.py — starts the server and mounts routes. [weather-api|src/cli/main.py:1-80]
- src/routes/status.py — health endpoint. [weather-api|src/routes/status.py:1-60]

How it works
- The CLI parses flags, builds a config, then calls create_app(). [weather-api|src/cli/main.py:12-40]
- create_app() registers /status and /metrics handlers. [weather-api|src/app.py:35-78]

Key functions/classes
- create_app(config): Constructs and returns the app. [weather-api|src/app.py:35-78]
- status(): Healthcheck handler. [weather-api|src/routes/status.py:1-30]

Code excerpts
```python
def create_app(config: Config) -> FastAPI:
    app = FastAPI()
    app.include_router(status.router)
    return app """

def _ext_from_path(path: str) -> str:
    if not path or "." not in path: return ""
    return path.rsplit(".", 1)[-1].lower()

def _clip_lines(text: str, max_lines: int) -> str:
    if not text: return ""
    lines = text.splitlines()
    if len(lines) <= max_lines: return text
    
    # For developers: be more intelligent about truncation
    # Show more context at the beginning and preserve key structural elements
    preserved_lines = []
    important_patterns = [
        r'^\s*class\s+\w+',     # class definitions
        r'^\s*def\s+\w+',       # function definitions  
        r'^\s*func\s+\w+',      # go function definitions
        r'^\s*type\s+\w+',      # type definitions
        r'^\s*interface\s+\w+', # interface definitions
        r'^\s*struct\s+\w+',    # struct definitions
        r'^\s*package\s+\w+',   # package declarations
        r'^\s*import\s+',       # imports
        r'^\s*//',              # comments (preserve some context)
        r'^\s*#',               # comments (preserve some context)
    ]
    
    # Take more from the beginning (first 80% of max_lines)
    primary_lines = int(max_lines * 0.8)
    remaining_quota = max_lines - primary_lines
    
    # Add primary lines
    preserved_lines.extend(lines[:primary_lines])
    
    # Try to add important lines from the remainder
    for i in range(primary_lines, len(lines)):
        if remaining_quota <= 0:
            break
        line = lines[i]
        if any(re.match(pattern, line) for pattern in important_patterns):
            preserved_lines.append(line)
            remaining_quota -= 1
    
    result = "\n".join(preserved_lines)
    total_omitted = len(lines) - len(preserved_lines)
    if total_omitted > 0:
        result += f"\n\n# …({total_omitted} lines omitted for brevity)…"
    return result

def format_repo_context(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "(none)"
    out = []
    for r in items:
        # Support either router context (short_title/summary/tech_stack/...) or repo_guides (overview/key_flows/...)
        langs = ", ".join(r.get("languages", []) or [])
        if any(k in r for k in ("overview", "key_flows", "modules")):
            out.append(
                f"- {r.get('repo_id')}\n"
                f"  Languages: {langs or 'unknown'}  Entrypoints: {r.get('entrypoints','')}\n"
                f"  Modules: {r.get('modules','')}\n"
                f"  Overview: {r.get('overview','')}\n"
                f"  Key Flows: {r.get('key_flows','')}"
            )
        else:
            out.append(
                f"- {r.get('repo_id')} | {r.get('short_title','')}\n"
                f"  Languages: {langs or 'unknown'}\n"
                f"  Modules: {r.get('key_modules','')}\n"
                f"  Symbols: {r.get('key_symbols','')}\n"
                f"  Tech: {r.get('tech_stack','')}  Entrypoints: {r.get('entrypoints','')}  Domains: {r.get('domains','')}\n"
                f"  Summary: {r.get('summary','')}"
            )
    return "\n".join(out)

def reorder_sources_for_quality(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Heuristic: implementation files before tests."""
    def is_test(s):
        p = (s.get("rel_path") or "").lower()
        return "tests" in p or "/test" in p or p.startswith("test_") or p.endswith("_test.py")
    impl = [s for s in sources if not is_test(s)]
    tests = [s for s in sources if is_test(s)]
    # keep original ordering inside buckets (already by retrieval score)
    return impl + tests

def pick_spotlights(sources: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Pick top-k sources by score with basic diversity (prefer distinct rel_path).
    Assumes sources roughly sorted by descending score from retrieval.
    """
    seen_paths = set()
    selected = []
    for s in sources:
        path = s.get("rel_path") or s.get("path")
        if path in seen_paths:
            continue
        selected.append(s)
        seen_paths.add(path)
        if len(selected) >= k:
            break
    return selected

def build_prompt(bundle: Dict[str, Any],
                 max_lines_per_chunk: int,
                 max_prompt_chars: int,
                 spotlight_n: int,
                 topic_terms: List[str] = None,
                 max_sources: int = 12,
                 include_repo_context: bool = False) -> str:
    question = bundle.get("query", "").strip()
    sources_all = reorder_sources_for_quality(bundle.get("sources", []))
    sources = sources_all[:max_sources]
    
    # Get repo guides for context
    repo_guides = bundle.get("repo_guides", [])

    how_to_answer = (
        "How to answer:\n"
        "- Answer ONLY if the provided code snippets contain relevant information.\n"
        "- Use ONLY the provided code sources as evidence; do NOT use external knowledge.\n"
        "- Repository context below is for understanding only - do NOT cite it as evidence.\n"
        "- Quote small code excerpts (≤ 15 lines) only when necessary and directly relevant.\n"
        "- End EVERY bullet/paragraph with at least one citation like [repo_id|rel_path:start-end].\n"
        f"- Include a section titled 'Spotlight' with up to {spotlight_n} code blocks (≤ 15 lines each), each with a citation.\n"
        "- If not in sources, say \"Not found in provided sources.\""
    )

    # Sources list header
    src_list_lines = []
    for i, s in enumerate(sources, 1):
        sl = s.get("start_line"); el = s.get("end_line")
        sl = sl if sl is not None else "?"
        el = el if el is not None else "?"
        src_list_lines.append(f"[#{i}] {s.get('repo_id')} | {s.get('rel_path')} | lines {sl}-{el}")
    src_list = "\n".join(src_list_lines)

    # Code blocks
    blocks = []
    for i, s in enumerate(sources, 1):
        sl = s.get("start_line"); el = s.get("end_line")
        sl = sl if sl is not None else "?"
        el = el if el is not None else "?"
        header = f"===== Source #{i}: {s.get('repo_id')} | {s.get('rel_path')} | lines {sl}-{el}"
        code = _clip_lines(s.get("code", ""), max_lines=max_lines_per_chunk)
        lang = _ext_from_path(s.get("rel_path") or "")
        if lang:
            block = f"{header}\n```{lang}\n{code}\n```"
        else:
            block = f"{header}\n```\n{code}\n```"
        blocks.append(block)

    # Topic check in user prompt (optional reinforcement; primary rule is in system)
    topic_section = ""
    if topic_terms:
        topic_section = (
            "\n\nTopic Check:\n"
            f"- Asked topic terms (any): {', '.join(topic_terms)}\n"
            "- If none of the snippets include these terms, reply exactly: Not found in provided sources.\n"
        )

    repo_context = ""
    if repo_guides and include_repo_context:
        ctx_lines = []
        for guide in repo_guides:
            repo_id = guide.get("repo_id", "unknown")
            ctx_lines.append(f"Repository: {repo_id}")
            if guide.get("overview"):    ctx_lines.append(f"  Overview: {guide.get('overview')}")
            if guide.get("key_flows"):   ctx_lines.append(f"  Key flows: {guide.get('key_flows')}")
            if guide.get("entrypoints"): ctx_lines.append(f"  Entry points: {guide.get('entrypoints')}")
            ctx_lines.append("")
        repo_context = (
            "Repository Context (for understanding only - DO NOT cite):\n" +
            "\n".join(ctx_lines) + "\n"
        )

    user = (
        f"Question:\n{question}\n\n"
        f"{how_to_answer}\n\n"
        f"{CITATION_FEWSHOT}\n"
        f"{repo_context}"
        "Sources:\n"
        f"{src_list}\n\n"
        + "\n\n".join(blocks)
        + topic_section
    )

    # Prompt size guard
    if len(user) > max_prompt_chars:
        fixed_head = (
            f"Question:\n{question}\n\n{how_to_answer}\n\n{CITATION_FEWSHOT}\n"
            f"{repo_context}"
            f"Sources:\n{src_list}\n\n"
        )
        kept = []
        total = len(fixed_head)
        for block in blocks:
            if total + len(block) + 2 <= max_prompt_chars:
                kept.append(block)
                total += len(block) + 2
            else:
                break
        user = fixed_head + "\n\n".join(kept)
        if len(kept) < len(blocks):
            user += "\n\n# NOTE: Some sources were omitted to fit prompt size."

    return user

def call_ollama_chat(model: str,
                     system_text: str,
                     user_text: str,
                     url: str = DEFAULT_OLLAMA_URL,
                     temperature: float = 0.0,
                     stream: bool = False,
                     timeout: int = 600) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ],
        "options": {"temperature": temperature},
        "stream": stream
    }
    with requests.post(url, json=body, timeout=timeout, stream=stream) as r:
        r.raise_for_status()
        if not stream:
            data = r.json()
            return (data.get("message") or {}).get("content", "")
        out = []
        for line in r.iter_lines():
            if not line: continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            msg = (chunk.get("message") or {}).get("content", "")
            if msg:
                sys.stdout.write(msg); sys.stdout.flush()
                out.append(msg)
        return "".join(out)

def has_citation(text: str) -> bool:
    return bool(re.search(r"\[[^\|\]]+\|[^:\]]+:\d+-\d+\]", text))

def _para_has_cite(line: str) -> bool:
    return bool(re.search(r"\[[^\|\]]+\|[^:\]]+:\d+-\d+\]\s*$", line.strip()))

def validate_answer(answer: str, min_ratio: float = 0.8) -> bool:
    # Count paragraphs/bullets and require citations on most of them.
    lines = [ln for ln in answer.strip().splitlines() if ln.strip()]
    if not lines: return False
    count = 0
    cited = 0
    for ln in lines:
        # ignore fenced code lines and headings
        if ln.strip().startswith("```") or ln.strip().startswith("#"):
            continue
        count += 1
        if _para_has_cite(ln):
            cited += 1
    if count == 0:
        return False
    return (cited / count) >= min_ratio

def contains_spotlight(text: str) -> bool:
    # Crude but effective: look for a "Spotlight" heading
    return "Spotlight" in text or "spotlight" in text

def repair_citations_if_needed(raw_answer: str,
                               sources_header: str,
                               model: str,
                               url: str,
                               temperature: float = 0.0) -> str:
    if has_citation(raw_answer):
        return raw_answer
    repair_system = "You only add citations to the given text. Do not change meaning."
    repair_user = (
        "Add proper citation tags like [repo_id|rel_path:start-end] to the following answer.\n"
        "Use ONLY the provided Sources list to choose paths and line ranges.\n"
        "Append citations at the end of each bullet/paragraph.\n\n"
        "Sources:\n" + sources_header + "\n\n"
        "Answer:\n" + raw_answer
    )
    fixed = call_ollama_chat(model, repair_system, repair_user, url, temperature, stream=False)
    return fixed if fixed else raw_answer

def render_spotlight_auto(sources: List[Dict[str, Any]], count: int, max_lines: int) -> str:
    picks = pick_spotlights(sources, count)
    if not picks: return ""
    out = ["\n---\n**Spotlight (auto)**"]
    for s in picks:
        sl = s.get("start_line"); el = s.get("end_line")
        sl = sl if sl is not None else "?"
        el = el if el is not None else "?"
        lang = _ext_from_path(s.get("rel_path") or "")
        code = _clip_lines(s.get("code",""), max_lines)
        out.append(
            f"\n{ s.get('repo_id') } | { s.get('rel_path') } | lines {sl}-{el}\n"
            + (f"```{lang}\n{code}\n```" if lang else f"```\n{code}\n```")
            + f"\n[{s.get('repo_id')}|{s.get('rel_path')}:{sl}-{el}]"
        )
    return "\n".join(out)

def evidence_overlap_ok(question: str, sources: list[dict], min_distinct: int = 2, top_n: int = 12) -> bool:
    stopwords = {"the","and","or","a","an","to","of","in","for","on","with","by","as","is","it","are","be","was","were","can","you","me","what","about","do","know","tell"}
    # tokenize question
    question_tokens = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]+", question.lower()):
        if len(token) >= 3 and token not in stopwords:
            question_tokens.add(token)
    if not question_tokens:
        return True

    distinct_found = set()
    for source in sources[:top_n]:
        code     = (source.get("code") or "")  # <-- use code
        symbols  = " ".join(source.get("symbols") or [])
        rel_path = (source.get("rel_path") or "")
        primsym  = (source.get("primary_symbol") or "")
        combined = f"{code} {symbols} {rel_path} {primsym}".lower()
        for token in question_tokens:
            if token in combined:
                distinct_found.add(token)
        if len(distinct_found) >= min_distinct:
            return True
    return len(distinct_found) >= min_distinct

def looks_broad(q: str) -> bool:
    broad = {"systems","architecture","overview","design","explain","describe","what do you know","tell me about"}
    ql = q.lower()
    return any(term in ql for term in broad)

def main():
    ap = argparse.ArgumentParser(description="Answer from retrieval bundle with local Qwen (read-only; Spotlight; enforced citations).")
    ap.add_argument("--in", dest="infile", default="retrieval.json", help="Retrieval bundle (version 1.2 or compatible)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama chat endpoint")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-lines-per-chunk", type=int, default=60)
    ap.add_argument("--max-sources", type=int, default=10, help="Max source snippets to include in the prompt")
    ap.add_argument("--include-repo-context", action="store_true",
                    help="Include repo_guides text as orientation (NEVER cited). Default: off")
    ap.add_argument("--max-prompt-chars", type=int, default=DEFAULT_MAX_PROMPT_CHARS)
    ap.add_argument("--spotlight", type=int, default=3, help="Request up to N spotlight code blocks from the model (developer-focused)")
    ap.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    ap.add_argument("--print-prompt", action="store_true", help="Print the built prompt before calling the model")
    args = ap.parse_args()

    if not os.path.exists(args.infile):
        print(f"Bundle not found: {args.infile}"); sys.exit(1)

    with open(args.infile, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    # Check for topic evidence gate failures
    topic_terms = (bundle.get("diagnostics", {}) or {}).get("topic_terms", [])
    reason = bundle.get("reason", "")
    hits = bundle.get("sources", [])  # In LLM bundle format, hits are called "sources"

    if reason == "no_topic_evidence" or not hits:
        # Return a safe refusal immediately
        print('=== Answer ===\n\nNot found in provided sources.')
        sys.exit(0)

    # Pre-LLM guards: Evidence-overlap and broad-topic checks
    question = bundle.get("query", "").strip()
    if looks_broad(question) and not evidence_overlap_ok(question, hits):
        print("=== Answer ===\n\nNot found in provided sources.")
        sys.exit(0)

    # Build prompt
    user_prompt = build_prompt(
        bundle=bundle,
        max_lines_per_chunk=args.max_lines_per_chunk,
        max_prompt_chars=args.max_prompt_chars,
        spotlight_n=args.spotlight,
        topic_terms=topic_terms,
        max_sources=args.max_sources,
        include_repo_context=args.include_repo_context
    )

    if args.print_prompt:
        print("\n===== PROMPT (USER) =====\n")
        print(user_prompt)
        print("\n=========================\n")

    # First pass
    answer = call_ollama_chat(
        model=args.model,
        system_text=READ_ONLY_SYSTEM,
        user_text=user_prompt,
        url=args.ollama_url,
        temperature=args.temperature,
        stream=args.stream
    )

    if args.stream:
        return

    if not has_citation(answer) or not validate_answer(answer):
        print("\n=== Answer ===\n")
        print("Not found in provided sources.")
        return

    print("\n=== Answer ===\n")
    print(textwrap.dedent(answer).strip())

    # Define ordered_sources for Spotlight fallback:
    ordered_sources = reorder_sources_for_quality(bundle.get("sources", []))
    if args.spotlight > 0 and not contains_spotlight(answer):
        fallback = render_spotlight_auto(ordered_sources, args.spotlight, max_lines=15)
        if fallback:
            print(fallback)

if __name__ == "__main__":
    main()
