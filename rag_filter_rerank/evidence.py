"""
Evidence wrapper for RAG v4 answerer.

Can use either imported rag_v4 functions or fallback implementations.
"""
import sys
import os
import logging
import requests
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Try to import from rag_v4
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from rag_v4.answerer import render_evidence as _rag_v4_render_evidence
    from rag_v4.answerer import build_messages as _rag_v4_build_messages
    from rag_v4.answerer import call_ollama as _rag_v4_call_ollama
    HAS_RAG_V4 = True
    logger.info("Using rag_v4 answerer functions")
except ImportError:
    HAS_RAG_V4 = False
    logger.warning("rag_v4 not available, using fallback implementations")


# Fallback implementations if rag_v4 not available
def _fallback_render_evidence(chunks: List[Dict[str, Any]], max_items: int = 12, max_code_chars: int = 10000) -> str:
    """Fallback evidence renderer."""
    lines = ["EVIDENCE", "--------"]
    for i, c in enumerate(chunks[:max_items], 1):
        repo_id = c.get("repo_id", "")
        rel_path = c.get("rel_path", c.get("path", ""))
        start_line = c.get("start_line", "")
        end_line = c.get("end_line", "")
        primary_symbol = c.get("primary_symbol", "")
        primary_kind = c.get("primary_kind", "")
        
        header = f"[{i}] {repo_id} | {rel_path} | L{start_line}–{end_line} | {primary_symbol} ({primary_kind})"
        lines.append(header)
        
        summ = c.get("summary_en", "")
        if summ:
            summ_clean = summ.replace('\n', ' ').strip()
            lines.append(f"summary: {summ_clean}")
        
        code = (c.get("text") or "").strip()
        if len(code) > max_code_chars:
            code = code[:max_code_chars] + "\n// … trimmed due to size …"
        lines.append("code:")
        lang = (c.get("language") or "").lower()
        fence = "```" + (lang if lang in {"go", "python", "typescript", "javascript", "java"} else "")
        lines.append(fence)
        lines.append(code)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _fallback_build_messages(user_question: str, evidence_block: str) -> List[Dict[str, str]]:
    """Fallback message builder."""
    system_prompt = """You are a meticulous code assistant. You will receive an EVIDENCE block from the assistant containing code chunks from the codebase.

Rules:
- Use ONLY the information in the EVIDENCE block when answering
- Cite evidence items inline like [1], [2] matching the headers in EVIDENCE
- If the answer requires facts not in EVIDENCE, say so and specify which files/symbols would be needed
- Prefer small patches and exact line references
- Do not create APIs or symbols that aren't in EVIDENCE
- Keep answers concise and actionable"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question.strip()},
        {"role": "assistant", "content": evidence_block},
        {
            "role": "user",
            "content": f"Using the EVIDENCE the assistant just provided, please answer this question:\n\n\"{user_question.strip()}\"\n\nRemember to:\n1) Cite evidence items like [1], [2], [3] from the headers\n2) Reference exact file paths and line numbers from EVIDENCE\n3) Keep the answer focused and technical\n4) If EVIDENCE doesn't contain enough information, say what's missing\n\nYour answer:"
        }
    ]


def _fallback_call_ollama(messages: List[Dict[str, str]], model: str, url: str, timeout: int = 180) -> str:
    """Fallback Ollama caller."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        msg = data.get("message", {}).get("content", "")
        if not msg and "choices" in data:
            msg = data["choices"][0]["message"]["content"]
        return msg or ""
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return f"Error calling LLM: {e}"


# Public API - use rag_v4 if available, else fallback
def render_evidence(chunks: List[Dict[str, Any]], max_items: int = 12, max_code_chars: int = 10000) -> str:
    """
    Render evidence block from chunks.
    
    Args:
        chunks: List of chunk dictionaries
        max_items: Maximum items to include
        max_code_chars: Maximum chars per code block
    
    Returns:
        Formatted evidence block
    """
    if HAS_RAG_V4:
        return _rag_v4_render_evidence(chunks, max_items, max_code_chars)
    return _fallback_render_evidence(chunks, max_items, max_code_chars)


def build_messages(user_question: str, evidence_block: str) -> List[Dict[str, str]]:
    """
    Build messages for LLM chat API.
    
    Args:
        user_question: User's question
        evidence_block: Formatted evidence
    
    Returns:
        List of message dicts
    """
    if HAS_RAG_V4:
        return _rag_v4_build_messages(user_question, evidence_block)
    return _fallback_build_messages(user_question, evidence_block)


def call_ollama(messages: List[Dict[str, str]], model: str, url: str, timeout: int = 180) -> str:
    """
    Call Ollama LLM.
    
    Args:
        messages: Chat messages
        model: Model name
        url: Ollama URL
        timeout: Request timeout
    
    Returns:
        LLM response text
    """
    if HAS_RAG_V4:
        return _rag_v4_call_ollama(messages, model, url, timeout)
    return _fallback_call_ollama(messages, model, url, timeout)


def build_evidence_from_chunks(chunks: List[Dict[str, Any]], k: int, max_code_chars: int) -> str:
    """
    Build evidence from top-k chunks.
    
    Args:
        chunks: Chunk dictionaries
        k: Number of chunks to include
        max_code_chars: Max chars per code block
    
    Returns:
        Formatted evidence block
    """
    return render_evidence(chunks[:k], max_items=k, max_code_chars=max_code_chars)


def call_answerer(query: str, evidence: str, settings) -> str:
    """
    Call answerer with evidence.
    
    Args:
        query: User query
        evidence: Formatted evidence block
        settings: Configuration settings
    
    Returns:
        Answer text
    """
    messages = build_messages(query, evidence)
    answer = call_ollama(
        messages,
        model=settings.OLLAMA_MODEL,
        url=settings.OLLAMA_URL,
        timeout=180
    )
    return answer

