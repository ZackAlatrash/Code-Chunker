"""
Utility functions for enrichment processing.

This module provides helper functions for language detection, keyword linting,
and other enrichment utilities.
"""

import re
from typing import List, Optional


# Common Dutch stopwords for language detection
DUTCH_STOPWORDS = {
    "de", "het", "een", "voor", "met", "van", "op", "aan", "in", "is", "zijn",
    "heeft", "wordt", "kan", "zal", "moet", "gaat", "komt", "maakt", "doet",
    "geeft", "neemt", "zet", "stelt", "vindt", "krijgt", "ziet", "hoort",
    "voelt", "denkt", "weet", "begrijpt", "leert", "leest", "schrijft",
    "spreekt", "luistert", "kijkt", "loopt", "rijdt", "vliegt", "zwemt",
    "eet", "drinkt", "slaapt", "werkt", "speelt", "zingt", "danst",
    "lacht", "huilt", "schreeuwt", "fluistert", "ademt", "leeft", "sterft"
}

# Forbidden terms that should not appear unless present in code
# These are common hallucinations where LLMs infer patterns not in the chunk
FORBIDDEN_TERMS = {
    # Retry/resilience patterns
    "exponential backoff", "retry", "retries", "circuit breaker", "rate limiter",
    "timeout", "backoff", "throttling", "throttle",
    # Caching/performance
    "caching strategy", "cache management", "connection pooling", "connection pool",
    "load balancing", "load balancer",
    # Auth/security patterns
    "authentication flow", "authorization logic", "session management",
    "token validation", "access control",
    # Architecture patterns
    "distributed system", "microservice", "service discovery",
    "event driven", "event sourcing", "saga pattern",
    # Async/queue patterns
    "async processing", "asynchronous", "queue management", "message queue",
    "pub sub", "publish subscribe",
    # Observability (unless explicitly seen)
    "distributed tracing", "trace propagation", "span creation",
    # Generic domain speculation
    "business logic", "domain model", "use case", "workflow",
    # Vague tech terms
    "data pipeline", "data processing", "etl process", "batch processing"
}


def looks_english(text: str) -> bool:
    """
    Check if text appears to be in English using simple heuristics.
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to be English
    """
    if not text:
        return True
    
    # Check for high ratio of non-ASCII characters
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if len(text) > 0 and ascii_chars / len(text) < 0.8:
        return False
    
    # Check for Dutch stopwords
    words = re.findall(r'\b\w+\b', text.lower())
    dutch_word_count = sum(1 for word in words if word in DUTCH_STOPWORDS)
    
    if len(words) > 0 and dutch_word_count / len(words) > 0.3:
        return False
    
    return True


def lint_keywords(keywords: List[str]) -> List[str]:
    """
    Clean and lint keywords for search-friendly format.
    
    Args:
        keywords: Raw keywords list
        
    Returns:
        Cleaned keywords list
    """
    if not keywords:
        return []
    
    cleaned = []
    seen = set()
    
    for keyword in keywords:
        if not keyword:
            continue
            
        # Convert to lowercase and clean
        clean_keyword = keyword.lower().strip()
        
        # Remove underscores and replace with spaces
        clean_keyword = clean_keyword.replace('_', ' ')
        
        # Remove long module paths (keep only last 2 parts)
        parts = clean_keyword.split()
        if len(parts) > 4:
            # Keep only last 4 words
            clean_keyword = ' '.join(parts[-4:])
        
        # Remove non-ASCII characters
        clean_keyword = ''.join(c for c in clean_keyword if ord(c) < 128)
        
        # Skip if empty or too short
        if len(clean_keyword) < 2:
            continue
            
        # Skip if contains absolute paths or import paths
        if any(pattern in clean_keyword for pattern in ['/', '\\', 'github.com', 'golang.org', 'google.golang.org']):
            continue
            
        # Skip if already seen
        if clean_keyword in seen:
            continue
            
        cleaned.append(clean_keyword)
        seen.add(clean_keyword)
        
        # Limit to 10 keywords
        if len(cleaned) >= 10:
            break
    
    # Ensure we have at least 6 keywords if possible
    return cleaned[:10] if len(cleaned) >= 6 else cleaned


def contains_forbidden_terms(text: str, code_text: str, file_synopsis: str = "") -> bool:
    """
    Check if text contains forbidden terms that aren't present in the code.
    
    Args:
        text: Text to check (e.g., summary)
        code_text: Original code text
        file_synopsis: File synopsis for additional context
        
    Returns:
        True if forbidden terms are present without justification in code
    """
    if not text:
        return False
    
    text_lower = text.lower()
    code_lower = code_text.lower()
    synopsis_lower = file_synopsis.lower()
    
    for term in FORBIDDEN_TERMS:
        if term in text_lower:
            # Check if term appears in code or synopsis
            if term not in code_lower and term not in synopsis_lower:
                return True
    
    return False


def extract_enclosing_method_signature(chunks: List[dict], current_index: int, max_lookback: int = 2) -> Optional[str]:
    """
    Extract enclosing method signature from neighboring chunks.
    
    Args:
        chunks: List of chunks for the same file
        current_index: Index of current chunk
        max_lookback: Maximum chunks to look back
        
    Returns:
        Method signature string if found, None otherwise
    """
    # Look backward for method signature
    for i in range(max(0, current_index - max_lookback), current_index):
        chunk = chunks[i]
        text = chunk.get("text", "")
        
        # Look for method signature pattern
        method_match = re.search(r'^func\s*\([^)]*\)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(', text, re.MULTILINE)
        if method_match:
            return method_match.group(0).strip()
    
    return None


def infer_method_from_neighbors(chunks: List[dict], current_index: int) -> Optional[tuple[str, str]]:
    """
    Infer method receiver and name from neighboring chunks.
    
    Args:
        chunks: List of chunks for the same file
        current_index: Index of current chunk
        
    Returns:
        Tuple of (receiver, method_name) if found, None otherwise
    """
    # Look backward for method signature
    for i in range(max(0, current_index - 2), current_index):
        chunk = chunks[i]
        text = chunk.get("text", "")
        
        # Look for method signature pattern
        method_match = re.search(r'^func\s*\(([^)]*)\)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(', text, re.MULTILINE)
        if method_match:
            receiver = method_match.group(1).strip()
            method_name = method_match.group(2)
            return receiver, method_name
    
    return None


# Verb stoplist for keyword linting
VERB_STOPLIST = {
    "get", "set", "wrap", "load", "return", "handle", "create", "make", "build",
    "process", "execute", "run", "call", "invoke", "perform", "do", "use",
    "implement", "provide", "generate", "parse", "validate", "check", "verify",
    "update", "modify", "change", "transform", "convert", "format", "encode",
    "decode", "serialize", "deserialize", "marshal", "unmarshal", "send",
    "receive", "fetch", "retrieve", "store", "save", "delete", "remove",
    "add", "insert", "append", "prepend", "merge", "split", "join", "connect",
    "disconnect", "open", "close", "start", "stop", "begin", "end", "init",
    "initialize", "cleanup", "destroy", "release", "acquire", "lock", "unlock"
}

# Filler/vague words to filter from keywords (per user's rule #9)
FILLER_WORDS = {
    "code", "chunk", "logic", "method", "function", "implementation",
    "routine", "procedure", "block", "section", "part", "piece",
    "system", "module", "component", "unit", "element", "item",
    "thing", "stuff", "data", "information", "value", "result"
}


def lint_keywords_enhanced(keywords: list) -> list:
    """
    Enhanced keyword linting with strict rules for NL search.
    
    Args:
        keywords: List of keyword strings
        
    Returns:
        Linted list of 6-8 keywords, lowercase, 2-4 words, nouns only, sorted
    """
    if not keywords:
        return []
    
    cleaned = []
    seen = set()
    domain_nouns = set()  # Collect potential domain nouns for synthesis
    
    for kw in keywords:
        if not kw or not isinstance(kw, str):
            continue
            
        # Convert to lowercase and clean
        kw = kw.lower().strip()
        
        # Remove file extensions and paths
        kw = re.sub(r'\.\w+$', '', kw)  # Remove .go, .pb, etc.
        kw = re.sub(r'[/\\]', ' ', kw)  # Replace path separators with spaces
        kw = re.sub(r'[^\w\s]', ' ', kw)  # Remove special characters
        kw = re.sub(r'\s+', ' ', kw).strip()  # Normalize whitespace
        
        # Skip if empty or too short
        if not kw or len(kw) < 2:
            continue
        
        # Check for identifiers: camelCase, PascalCase, underscores, digits, dots
        if (re.search(r'[a-z][A-Z]', kw) or  # camelCase
            re.search(r'[A-Z][a-z]', kw) or  # PascalCase  
            '_' in kw or                      # underscores
            re.search(r'\d', kw) or          # digits
            '.' in kw):                      # dotted names
            # Extract potential domain nouns from identifiers
            # Convert to lowercase first, then extract words
            kw_lower = kw.lower()
            words = re.findall(r'\b[a-z]+\b', kw_lower)
            for word in words:
                if len(word) >= 3 and word not in VERB_STOPLIST:
                    domain_nouns.add(word)
            continue
            
        # Check word count (2-4 words)
        words = kw.split()
        if len(words) < 2 or len(words) > 4:
            continue
            
        # Check for verbs (skip if any word is a verb)
        if any(word in VERB_STOPLIST for word in words):
            continue
            
        # Check for filler words (skip if any word is filler)
        if any(word in FILLER_WORDS for word in words):
            continue
            
        # Skip if already seen
        if kw in seen:
            continue
            
        cleaned.append(kw)
        seen.add(kw)
    
    # Sort for stability
    cleaned.sort()
    
    # DISABLED: Template synthesis creates repetitive keywords
    # Instead, return whatever high-quality keywords we found, even if < 6
    # The LLM should generate natural language keywords directly
    
    # Final limit
    if len(cleaned) > 8:
        cleaned = cleaned[:8]
    
    return cleaned


def validate_summary_en(summary: str) -> bool:
    """
    Validate that summary_en meets requirements.
    
    Args:
        summary: Summary text to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not summary or not isinstance(summary, str):
        return False
        
    # Check length
    if len(summary) > 160:
        return False
        
    # Check for newlines
    if '\n' in summary:
        return False
        
    # Basic English check (no non-ASCII)
    if any(ord(char) > 127 for char in summary):
        return False
        
    return True


def normalize_summary_en(text: str) -> str:
    """
    Normalize summary to be single sentence, ≤160 chars, English-only.
    
    Args:
        text: Raw summary text
        
    Returns:
        Normalized summary string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Take first sentence only - split on sentence endings
    # Handle common abbreviations conservatively
    sentences = re.split(r'[.!?]+', text)
    if sentences:
        first_sentence = sentences[0].strip()
        
        # Handle trailing quotes/parentheses
        first_sentence = re.sub(r'[")]+$', '', first_sentence).strip()
        
        # If >160 chars, truncate at last whitespace before 160
        if len(first_sentence) > 160:
            truncated = first_sentence[:160]
            last_space = truncated.rfind(' ')
            if last_space > 100:  # Only truncate if we have reasonable space
                first_sentence = truncated[:last_space]
            else:
                first_sentence = truncated
        
        return first_sentence
    
    return ""


def enforce_english_only(summary: str, keywords: list) -> tuple[str, list]:
    """
    Lightweight English-only enforcement for summary and keywords.
    
    Args:
        summary: Summary text
        keywords: List of keywords
        
    Returns:
        Tuple of (cleaned_summary, cleaned_keywords)
    """
    # Check if >15% of chars are non-ASCII
    if summary:
        ascii_chars = sum(1 for c in summary if ord(c) < 128)
        if len(summary) > 0 and ascii_chars / len(summary) < 0.85:
            # Strip non-ASCII chars from summary
            summary = ''.join(c for c in summary if ord(c) < 128)
    
    # Filter non-ASCII keywords
    if keywords:
        keywords = [kw for kw in keywords if all(ord(c) < 128 for c in kw)]
    
    return summary, keywords


def generate_fallback_summary(node_kind: str, signature_info: dict, identifiers: list) -> str:
    """
    Generate a fallback summary when LLM fails.
    
    Args:
        node_kind: The detected node kind
        signature_info: Extracted signature information
        identifiers: List of relevant identifiers
        
    Returns:
        Fallback summary string
    """
    # Extract top 2 nouns from identifiers
    nouns = []
    for ident in identifiers:
        if isinstance(ident, str) and ident.isalpha() and len(ident) > 2:
            # Simple heuristic: capitalized words are likely nouns
            if ident[0].isupper():
                nouns.append(ident.lower())
    
    top_nouns = nouns[:2]
    noun_phrase = " and ".join(top_nouns) if top_nouns else "code"
    
    if node_kind == "method":
        method_name = signature_info.get("method_name", "method")
        receiver = signature_info.get("receiver", "")
        # Extract type name from receiver (remove variable name and pointer)
        type_name = ""
        if receiver:
            if "*" in receiver:
                # Handle "s *Service" -> "Service"
                parts = receiver.split("*")
                if len(parts) > 1:
                    type_name = parts[1].strip()
            else:
                # Handle "s Service" -> "Service"
                parts = receiver.split()
                if len(parts) > 1:
                    type_name = parts[-1].strip()
        
        if type_name:
            return f"Method {type_name}.{method_name} — {noun_phrase}."
        else:
            return f"Method {method_name} — {noun_phrase}."
    elif node_kind == "function":
        func_name = signature_info.get("function_name", "function")
        return f"Function {func_name} — {noun_phrase}."
    elif node_kind == "type":
        type_name = signature_info.get("type_name", "type")
        return f"Type {type_name} — {noun_phrase}."
    elif node_kind == "header":
        return f"Package/import header — {noun_phrase}."
    else:
        return f"Code chunk — {noun_phrase}."
