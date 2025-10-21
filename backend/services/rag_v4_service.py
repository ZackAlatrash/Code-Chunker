"""
RAG v4 Service - Evidence-based answering with citations

Wraps the rag_v4 answerer for use in the FastAPI backend.
"""
import sys
import os
from typing import List, Dict, Any, Optional

# Add rag_v4 to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from rag_v4.answerer import render_evidence, build_messages, call_ollama

class RAGv4Service:
    """Service for RAG v4 Evidence-based answering"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434/api/chat", ollama_model: str = "qwen2.5-coder:7b-instruct"):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
    
    def answer_question(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        max_items: int = 12,
        max_code_chars: int = 10000,
        timeout: int = 240  # Increased for longer, detailed answers
    ) -> Dict[str, Any]:
        """
        Answer a question using Evidence-based prompting.
        
        Args:
            question: Developer question
            chunks: Retrieved code chunks (from search_v4 or legacy search)
            max_items: Maximum number of chunks to include
            max_code_chars: Maximum chars per chunk code block
            timeout: Request timeout in seconds
        
        Returns:
            Dictionary with:
                - answer: str - The LLM's answer
                - evidence: str - The Evidence block shown to LLM
                - chunks_used: int - Number of chunks included
                - model: str - Model used
        """
        if not chunks:
            return {
                "answer": "No code chunks found to answer your question.",
                "evidence": "",
                "chunks_used": 0,
                "model": self.ollama_model
            }
        
        # Render evidence
        evidence = render_evidence(chunks, max_items=max_items, max_code_chars=max_code_chars)
        
        # Build messages
        messages = build_messages(question, evidence)
        
        # Call LLM
        try:
            answer = call_ollama(
                messages,
                model=self.ollama_model,
                url=self.ollama_url,
                timeout=timeout
            )
        except Exception as e:
            answer = f"Error calling LLM: {str(e)}"
        
        return {
            "answer": answer,
            "evidence": evidence,
            "chunks_used": min(len(chunks), max_items),
            "model": self.ollama_model
        }
    
    def extract_citations(self, answer: str) -> List[int]:
        """
        Extract citation indices [1], [2], etc. from the answer.
        
        Returns:
            List of citation indices (1-based)
        """
        import re
        citations = re.findall(r'\[(\d+)\]', answer)
        return sorted(set(int(c) for c in citations))

