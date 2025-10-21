#!/usr/bin/env python3
import argparse, json, os, textwrap, requests
from typing import Any, Dict, List, Optional

# --- Config (env-overridable) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct")
# If you have an HTTP proxy or auth, honor standard envs (requests does automatically)

def render_evidence(chunks: List[Dict[str, Any]], max_items: int = 12, max_code_chars: int = 10000) -> str:
    """
    Renders a big EVIDENCE block from v4 chunks. We keep it generous on purpose.
    Each chunk is expected to have fields like:
      repo_id, rel_path, start_line, end_line, primary_symbol, primary_kind, all_roles, summary_en, text
    """
    lines = ["EVIDENCE", "--------"]
    for i, c in enumerate(chunks[:max_items], 1):
        repo_id = c.get("repo_id", "")
        rel_path = c.get("rel_path", c.get("path", ""))
        start_line = c.get("start_line", "")
        end_line = c.get("end_line", "")
        primary_symbol = c.get("primary_symbol", "")
        primary_kind = c.get("primary_kind", "")
        roles = c.get("all_roles") or []
        roles_str = ",".join(roles) if roles else "n/a"
        header = f"[{i}] {repo_id} | {rel_path} | L{start_line}–{end_line} | {primary_symbol} ({primary_kind}/{roles_str})"
        lines.append(header)

        summ = c.get("summary_en", "")
        if summ:
            # keep one-line; collapse newlines
            summ_clean = summ.replace('\n', ' ').strip()
            lines.append(f"summary: {summ_clean}")

        code = (c.get("text") or "").strip()
        if len(code) > max_code_chars:
            code = code[:max_code_chars] + "\n// … trimmed due to size …"
        lines.append("code:")
        # fence with language if available
        lang = (c.get("language") or "").lower()
        fence = "```" + (lang if lang in {"go","python","typescript","javascript","java","c","cpp","rust","ruby","php"} else "")
        lines.append(fence)
        lines.append(code)
        lines.append("```")
        lines.append("")  # spacer
    return "\n".join(lines)

SYSTEM_PROMPT = """You are an expert code assistant helping developers understand complex codebases. You will receive an EVIDENCE block containing relevant code chunks.

Your task is to provide DETAILED, COMPREHENSIVE answers that include:

1. **Clear Structure**: Use sections like "## Overview", "## Implementation", "## Key Details", "## Important Notes"
2. **Inline Code Snippets**: Include relevant code directly in your answer (not just citations)
3. **Line-Specific References**: Cite evidence like [1, lines 45-50] for precision
4. **Complete Explanations**: Don't just say what the code does - explain HOW and WHY
5. **Practical Context**: Mention error handling, edge cases, performance considerations

STRICT RULES:
- Use ONLY information from the EVIDENCE block
- ALWAYS cite evidence items as [1], [2], [3] matching the headers
- Include code snippets in markdown fenced blocks with language tags (```go, ```python, etc.)
- When showing code, use actual code from EVIDENCE (copy relevant sections)
- If EVIDENCE is insufficient, explicitly state what files/symbols are missing
- DO NOT invent APIs, functions, or behavior not shown in EVIDENCE

FORMAT GUIDELINES:
- Start with a brief overview (1-2 sentences)
- Use ## headers for main sections
- Show code snippets that are 5-30 lines (not full files)
- End with "## Related" section if you see connected code in EVIDENCE"""

def build_messages(user_question: str, evidence_block: str) -> List[Dict[str, str]]:
    """
    Returns messages for an LLM chat API:
      - system
      - user
      - assistant (context: Evidence)
      - user (kick-off request for answer)
    The final 'user' is phrased so model proceeds to answer with citations.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question.strip()},
        {
            "role": "assistant",
            "content": evidence_block
        },
        {
            "role": "user",
            "content": textwrap.dedent(f"""\
                Using the EVIDENCE above, provide a DETAILED answer to this question:
                
                "{user_question.strip()}"
                
                Your answer should include:
                
                ## Overview
                Start with a 1-2 sentence summary of the answer.
                
                ## Implementation
                Explain HOW it works with inline code snippets from EVIDENCE.
                Show the key code sections (5-30 lines each) in fenced blocks.
                Reference specific files and line numbers like [1, lines 45-50].
                
                ## Key Details
                Important behavior, parameters, error handling, or edge cases.
                
                ## Important Notes (if applicable)
                Performance considerations, gotchas, or related functionality.
                
                Remember:
                - Include actual CODE from EVIDENCE in your answer (not just citations)
                - Use markdown fenced blocks: ```go, ```python, etc.
                - Cite evidence items: [1], [2], [3]
                - If EVIDENCE is insufficient, state what's missing
                
                Begin your detailed answer:
            """).strip()
        }
    ]
    return messages

def call_ollama(messages: List[Dict[str, str]], model: Optional[str] = None, url: Optional[str] = None, timeout: int = 180) -> str:
    model = model or OLLAMA_MODEL
    url = url or OLLAMA_URL
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048  # Allow longer responses (default is 128)
        }
    }
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # Ollama's chat returns {"message":{"role":"assistant","content":"..."}, ...}
    msg = (data.get("message") or {}).get("content", "")
    # Some servers return OpenAI-like format; try to fall back cleanly:
    if not msg and "choices" in data:
        msg = data["choices"][0]["message"]["content"]
    return msg or ""

def load_chunks_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # allow either {"hits":[...]} or plain array
    if isinstance(data, dict) and "hits" in data:
        return data["hits"]
    return data if isinstance(data, list) else []

def try_search_chunks_v4(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    If search_v4.search exists, use it. Otherwise return [].
    """
    try:
        from search_v4.search import search_chunks_v4  # type: ignore
        return search_chunks_v4(query=query, k=k)
    except Exception:
        return []

def main():
    ap = argparse.ArgumentParser(description="Answer using v4 chunks Evidence.")
    ap.add_argument("question", help="Developer question")
    ap.add_argument("--k", type=int, default=8, help="Top-k chunks to include")
    ap.add_argument("--chunks-json", help="Optional path to JSON of chunks (array or {hits:[...]})")
    ap.add_argument("--model", default=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL))
    ap.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", OLLAMA_URL))
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    # Get chunks either from JSON file or live search
    chunks = []
    if args.chunks_json and os.path.exists(args.chunks_json):
        chunks = load_chunks_from_json(args.chunks_json)
    else:
        chunks = try_search_chunks_v4(args.question, k=args.k)

    if not chunks:
        print("[answerer] No chunks found. Provide --chunks-json or ensure search_v4.search is available.")
        return

    evidence = render_evidence(chunks, max_items=args.k, max_code_chars=10000)
    messages = build_messages(args.question, evidence)
    answer = call_ollama(messages, model=args.model, url=args.ollama_url, timeout=args.timeout)
    print(answer)

if __name__ == "__main__":
    main()

