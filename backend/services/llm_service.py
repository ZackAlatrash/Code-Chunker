"""
LLM Service

Provides clean interface to LLM functionality using existing scripts.
"""

import sys
import os
from typing import List, Dict, Any, Optional
import requests

# Add scripts directory to path  
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

# Import LLM functionality from existing scripts
from answer import (
    READ_ONLY_SYSTEM,
    build_prompt,
    format_repo_context,
    reorder_sources_for_quality,
    pick_spotlights,
    call_ollama_chat,
    evidence_overlap_ok,
    looks_broad,
    has_citation,
    validate_answer,
    contains_spotlight,
    render_spotlight_auto
)

DEFAULT_MODEL = "qwen2.5-coder:7b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

class LLMService:
    """Service for handling LLM operations"""
    
    def __init__(self, model: str = DEFAULT_MODEL, ollama_url: str = DEFAULT_OLLAMA_URL):
        self.model = model
        self.ollama_url = ollama_url
    
    def call_llm(self, prompt: str, temperature: float = 0.0, timeout: int = 300) -> str:
        """Call Ollama LLM with a prompt"""
        return call_ollama_chat(
            model=self.model,
            system_text=READ_ONLY_SYSTEM,
            user_text=prompt,
            url=self.ollama_url,
            temperature=temperature
        )
    
    def build_qa_prompt(self, bundle: Dict[str, Any], max_lines_per_chunk: int = 120,
                       max_prompt_chars: int = 120000, spotlight_n: int = 3,
                       topic_terms: List[str] = None, max_sources: int = 12,
                       include_repo_context: bool = False) -> str:
        """Build Q&A prompt from retrieval bundle"""
        return build_prompt(
            bundle=bundle,
            max_lines_per_chunk=max_lines_per_chunk,
            max_prompt_chars=max_prompt_chars,
            spotlight_n=spotlight_n,
            topic_terms=topic_terms,
            max_sources=max_sources,
            include_repo_context=include_repo_context
        )
    
    def reorder_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reorder sources for better quality (implementation before tests)"""
        return reorder_sources_for_quality(sources)
    
    def select_spotlight_sources(self, sources: List[Dict[str, Any]], 
                                count: int) -> List[Dict[str, Any]]:
        """Pick top sources for spotlight section"""
        return pick_spotlights(sources, count)
    
    def format_repository_context(self, repo_guides: List[Dict[str, Any]]) -> str:
        """Format repository context for prompts"""
        return format_repo_context(repo_guides)
    
    def answer_question(self, bundle: Dict[str, Any], 
                       temperature: float = 0.0,
                       max_lines: int = 120,
                       spotlight_count: int = 3,
                       topic_terms: List[str] = None,
                       max_sources: int = 12,
                       include_repo_context: bool = False) -> str:
        """Generate answer from retrieval bundle"""
        
        # Check for topic evidence gate failures
        topic_terms = topic_terms or (bundle.get("diagnostics", {}) or {}).get("topic_terms", [])
        reason = bundle.get("reason", "")
        hits = bundle.get("sources", [])

        if reason == "no_topic_evidence" or not hits:
            return "Not found in provided sources."
        
        # Pre-LLM guards: Evidence-overlap and broad-topic checks
        question = bundle.get("query", "").strip()
        if looks_broad(question) and not evidence_overlap_ok(question, hits):
            return "Not found in provided sources."
        
        # Build and execute prompt
        prompt = self.build_qa_prompt(
            bundle=bundle,
            max_lines_per_chunk=max_lines,
            spotlight_n=spotlight_count,
            topic_terms=topic_terms,
            max_sources=max_sources,
            include_repo_context=include_repo_context
        )
        answer = self.call_llm(prompt, temperature)
        
        # Use new validator
        if not has_citation(answer) or not validate_answer(answer):
            return "Not found in provided sources."
        
        # Spotlight fallback if the model didn't include one
        if spotlight_count > 0 and not contains_spotlight(answer):
            ordered_sources = reorder_sources_for_quality(hits)
            # Use developer-friendly limits: more lines and more snippets
            dev_spotlight_count = max(spotlight_count, 3)  # At least 3 snippets for context
            fallback = render_spotlight_auto(ordered_sources, dev_spotlight_count, max_lines=75)
            if fallback:
                answer += "\n\n" + fallback
        
        return answer
        
