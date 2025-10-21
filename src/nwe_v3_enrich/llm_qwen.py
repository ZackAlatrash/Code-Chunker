"""
LLM enrichment using local Ollama with Qwen2.5-coder model.

This module provides functions to enrich chunks with AI-generated summaries
and keywords using a local Ollama instance.
"""

import json
import hashlib
import sqlite3
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional

from .utils import (
    looks_english, 
    lint_keywords, 
    contains_forbidden_terms,
    lint_keywords_enhanced,
    validate_summary_en,
    generate_fallback_summary,
    normalize_summary_en,
    enforce_english_only
)


def digest_for_cache(chunk: Dict[str, Any], prompt_version: str) -> str:
    """
    Create a cache key for the chunk based on its content and prompt version.
    
    Args:
        chunk: Chunk dictionary
        prompt_version: Version of the prompt used
        
    Returns:
        SHA256 hash as cache key
    """
    # Extract relevant fields for cache key
    ast_path = chunk.get("ast_path", "")
    code = chunk.get("text", "")
    imports = sorted(chunk.get("imports_used_minimal", []))
    symbols = sorted(chunk.get("symbols_referenced_strict", []))
    
    # Create deterministic hash
    content = f"{prompt_version}|{ast_path}|{code}|{imports}|{symbols}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def summarize_chunk_qwen(chunk: Dict[str, Any], model: str, prompt_version: str, timeout: float = 10.0, file_synopsis: str = "") -> Dict[str, Any]:
    """
    Generate summary and keywords for a chunk using local Ollama.

    Args:
        chunk: Chunk dictionary with metadata
        model: Ollama model name
        prompt_version: Version of the prompt
        timeout: Request timeout in seconds
        file_synopsis: File synopsis for context

    Returns:
        Dictionary with summary_en and keywords_en fields
    """
    # Enhanced system prompt for English-only, concise output
    system_prompt = """You are enriching code chunks for semantic search. Follow these STRICT RULES:

HARD RULES (must follow):
1. NO SPECULATION. Only describe what is explicitly visible in this chunk's code.
2. CHUNK-LOCAL ONLY. Never infer purpose from typical patterns or other files.
3. NO RAW IDENTIFIERS in keywords (e.g., "getCacheKey", "Service"). Use conceptual nouns (e.g., "cache management").
4. NO VENDOR/ORG NAMES in keywords (e.g., "impalastudios", "zap"). Use functional concepts (e.g., "logging utilities").
5. NO TECH NOT PRESENT. Don't mention databases, caches, tracing, etc. unless clearly used in THIS chunk.
6. CONCISE, NEUTRAL, TECHNICAL tone. No filler words like "code", "chunk", "logic", "method".
7. English only. Lowercase keywords, 2-4 words each, descriptive nouns only.

SPECIAL CASES:
- Package header (just "package X"): "Declares the <pkg> package." | keywords: ["module declaration", "package declaration"]
- Import block: "Imports stdlib and deps for <categories>." | keywords: category nouns like "json parsing", "cli framework"
- Type-only struct: "Defines <Type> with fields for <observable purpose from tags/names>." | keywords: "configuration struct", etc.
- Functions/methods: describe observable behavior only (io, parsing, queries, logging, concurrency, error handling).

OUTPUT: 6-8 lowercase keyword phrases (2-4 words), deduplicated and sorted."""

    # Extract chunk metadata
    ast_path = chunk.get("ast_path", "")
    node_kind = chunk.get("node_kind", "")
    primary_symbol = chunk.get("primary_symbol", "")
    type_name = chunk.get("type_name", "")
    function_name = chunk.get("function_name", "")
    method_name = chunk.get("method_name", "")
    package = chunk.get("package", "")
    imports = chunk.get("imports_used_minimal", [])

    # Build chunk signature
    chunk_signature = ""
    if node_kind == "method":
        receiver = chunk.get("receiver", "")
        if receiver and method_name:
            chunk_signature = f"func ({receiver}) {method_name}()"
    elif node_kind == "function" and function_name:
        chunk_signature = f"func {function_name}()"
    elif node_kind == "type" and type_name:
        type_kind = chunk.get("type_kind", "")
        chunk_signature = f"type {type_name} {type_kind}"

    # Truncate code for prompt (first ~40 lines or ~600 chars) at word boundary
    # Shorter = more focused summaries, less hallucination
    code_text = chunk.get("text", "")
    lines = code_text.split('\n')
    if len(lines) > 40:
        code_text = '\n'.join(lines[:40]) + "\n..."
    elif len(code_text) > 600:
        # Truncate at last word boundary to avoid dangling tokens
        truncated = code_text[:600]
        last_space = truncated.rfind(' ')
        last_newline = truncated.rfind('\n')
        cutoff = max(last_space, last_newline)
        if cutoff > 500:  # Only if we don't lose too much
            code_text = truncated[:cutoff] + "..."
        else:
            code_text = truncated + "..."

    user_prompt = f"""FILE CONTEXT:
{file_synopsis}

NODE TYPE: {node_kind}
SIGNATURE: {chunk_signature}

CODE TO ANALYZE:
{code_text}

TASK:
Write ONE sentence (≤160 chars) describing what THIS chunk does, following the STRICT RULES above.

EXAMPLES OF GOOD SUMMARIES:
- Package header: "Declares the main package."
- Import block: "Imports stdlib and deps for json parsing, cli flags, and logging utilities."
- Type definition: "Defines Config with fields for file paths and retry settings."
- Function with I/O: "Reads json file, unmarshals data, validates structure, and returns parsed objects."
- Method with context: "Loads config, starts worker goroutine, handles signals, and propagates errors."

EXAMPLES OF BAD SUMMARIES (avoid these):
- ❌ "Main package with a single function, though its purpose is not clear" (speculation)
- ❌ "Provides the main entry point for an aviation schedules application" (inferred domain)
- ❌ "Implements caching strategy for user sessions" (tech not present in chunk)
- ❌ "Service method that handles requests" (uses identifier "Service", vague "handles")

KEYWORDS:
Generate 6-8 conceptual noun phrases (2-4 words, lowercase):
- ✅ GOOD: "json parsing", "error handling", "cli framework", "goroutine management", "signal handling"
- ❌ BAD: "GetForecast", "Service", "impalastudios", "zap logger", "code logic"

Produce JSON ONLY:
{{
  "summary_en": "observable behavior in this chunk only",
  "keywords_en": ["concept 1", "concept 2", "concept 3", "concept 4", "concept 5", "concept 6"]
}}"""
    
    # Create the request payload with deterministic settings
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,       # Lower = more deterministic
            "top_p": 0.9,             # Nucleus sampling
            "repeat_penalty": 1.1,    # Reduce repetition
            "seed": 42                # Reproducibility
        }
    }
    
    # Make request to Ollama
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            
        if "response" in result:
            response_text = result["response"]
            try:
                # Parse the JSON response
                llm_result = json.loads(response_text)
                
                # Validate the response format
                if "summary_en" in llm_result and "keywords_en" in llm_result:
                    summary = llm_result["summary_en"]
                    keywords = llm_result["keywords_en"]

                    # Normalize summary to single sentence, ≤160 chars
                    summary = normalize_summary_en(summary)
                    
                    # Check for forbidden terms
                    if contains_forbidden_terms(summary, code_text, file_synopsis):
                        return _retry_llm_request_tighten(payload, timeout, file_synopsis, code_text)

                    # Clean and validate keywords with enhanced linting
                    cleaned_keywords = lint_keywords_enhanced(keywords)
                    
                    # Enforce English-only
                    summary, cleaned_keywords = enforce_english_only(summary, cleaned_keywords)

                    return {
                        "summary_en": summary,
                        "keywords_en": cleaned_keywords
                    }
                
                # If validation fails, retry once
                return _retry_llm_request_english(payload, timeout, file_synopsis, code_text)
                
            except json.JSONDecodeError:
                # If JSON parsing fails, retry once
                return _retry_llm_request_english(payload, timeout, file_synopsis, code_text)
        else:
            raise Exception("No response from Ollama")
            
    except Exception as e:
        print(f"Warning: LLM enrichment failed: {e}")
        # Generate fallback summary using rule-based approach
        node_kind = chunk.get("node_kind", "block")
        signature_info = {
            "receiver": chunk.get("receiver", ""),
            "method_name": chunk.get("method_name", ""),
            "function_name": chunk.get("function_name", ""),
            "type_name": chunk.get("type_name", ""),
            "type_kind": chunk.get("type_kind", "")
        }
        identifiers = [
            chunk.get("primary_symbol", ""),
            chunk.get("type_name", ""),
            chunk.get("package", "")
        ]
        identifiers = [i for i in identifiers if i]  # Remove empty strings
        
        fallback_summary = generate_fallback_summary(node_kind, signature_info, identifiers)
        fallback_keywords = ["code", "chunk", "go", "method", "function", "type", "implementation", "logic"]
        
        return {
            "summary_en": fallback_summary,
            "keywords_en": fallback_keywords
        }


def _retry_llm_request_english(payload: Dict[str, Any], timeout: float, file_synopsis: str, code_text: str) -> Dict[str, Any]:
    """Retry LLM request with stronger English instruction."""
    try:
        # Stronger English prompt for retry
        payload["prompt"] = f"""# Context
- Language: Go
- File synopsis: {file_synopsis}
- Chunk text: {code_text}

# Tasks (ENGLISH ONLY - NO DUTCH):
1) summary: One-sentence description in ENGLISH (≤160 chars) of what this code does.
2) keywords: 6–10 short, lowercase, search-friendly phrases in ENGLISH.

# Output JSON shape (exact keys):
{{
  "summary_en": "<string>",
  "keywords_en": ["k1","k2","k3","k4","k5","k6"]
}}

Return JSON only."""
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            
        if "response" in result:
            llm_result = json.loads(result["response"])
            if "summary_en" in llm_result and "keywords_en" in llm_result:
                summary = llm_result["summary_en"]
                keywords = llm_result["keywords_en"]
                
                # Clean keywords
                cleaned_keywords = lint_keywords(keywords)
                
                # Ensure summary length
                if len(summary) > 160:
                    summary = summary[:157] + "..."
                
                return {
                    "summary_en": summary,
                    "keywords_en": cleaned_keywords
                }
                
    except Exception:
        pass
    
    # Fallback response
    return {
        "summary_en": "LLM enrichment failed after retry",
        "keywords_en": ["error", "retry", "failed", "llm", "enrichment", "chunk", "code", "analysis", "unavailable"]
    }


def _retry_llm_request_tighten(payload: Dict[str, Any], timeout: float, file_synopsis: str, code_text: str) -> Dict[str, Any]:
    """Retry LLM request with tightened anti-hallucination prompt."""
    try:
        # Tightened prompt for retry
        payload["prompt"] = f"""# Context
- Language: Go
- File synopsis: {file_synopsis}
- Chunk text: {code_text}

# Tasks (ENGLISH ONLY):
1) summary: One-sentence description (≤160 chars) of what this code does. ABSOLUTELY DO NOT MENTION any mechanism unless the exact tokens appear in the code.
2) keywords: 6–10 short, lowercase, search-friendly phrases.

# Output JSON shape (exact keys):
{{
  "summary_en": "<string>",
  "keywords_en": ["k1","k2","k3","k4","k5","k6"]
}}

Return JSON only."""
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            
        if "response" in result:
            llm_result = json.loads(result["response"])
            if "summary_en" in llm_result and "keywords_en" in llm_result:
                summary = llm_result["summary_en"]
                keywords = llm_result["keywords_en"]
                
                # Clean keywords
                cleaned_keywords = lint_keywords(keywords)
                
                # Ensure summary length
                if len(summary) > 160:
                    summary = summary[:157] + "..."
                
                return {
                    "summary_en": summary,
                    "keywords_en": cleaned_keywords
                }
                
    except Exception:
        pass
    
    # Fallback response
    return {
        "summary_en": "LLM enrichment failed after retry",
        "keywords_en": ["error", "retry", "failed", "llm", "enrichment", "chunk", "code", "analysis", "unavailable"]
    }


class LLMCache:
    """Simple SQLite cache for LLM results."""
    
    def __init__(self, cache_path: str = ".cache/nwe_v3_enrich.sqlite"):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    chunk_id TEXT,
                    file_sha TEXT,
                    prompt_version TEXT,
                    ast_path TEXT,
                    summary_llm TEXT,
                    keywords_llm TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result."""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute(
                "SELECT summary_llm, keywords_llm FROM llm_cache WHERE cache_key = ?",
                (cache_key,)
            )
            row = cursor.fetchone()
            if row:
                # Convert old format to new format
                return {
                    "summary_en": row[0],
                    "keywords_en": json.loads(row[1])
                }
        return None
    
    def set(self, cache_key: str, chunk_id: str, file_sha: str, 
            prompt_version: str, ast_path: str, result: Dict[str, Any]):
        """Cache the result."""
        with sqlite3.connect(self.cache_path) as conn:
            # Convert new format to old format for storage
            summary = result.get("summary_en", result.get("summary_llm", ""))
            keywords = result.get("keywords_en", result.get("keywords_llm", []))
            
            conn.execute("""
                INSERT OR REPLACE INTO llm_cache 
                (cache_key, chunk_id, file_sha, prompt_version, ast_path, summary_llm, keywords_llm)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key, chunk_id, file_sha, prompt_version, ast_path,
                summary, json.dumps(keywords)
            ))
