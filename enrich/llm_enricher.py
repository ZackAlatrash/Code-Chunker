"""
LLM enrichment core for generating Dutch summaries and keywords.

This module handles model calls, batching, retries, and content processing
for enriching code chunks with Dutch language summaries.
"""

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import requests


class LLMEnricher:
    """Core LLM enrichment engine with batching and retry logic."""
    
    # Generated file patterns
    GENERATED_PATTERNS = [
        r'\.pb\.go$',
        r'\.gen\.go$',
        r'_generated\.',
        r'\.g\.dart$',
        r'Designer\.cs$',
        r'\.generated\.',
        r'\.pb\.py$',
        r'\.pb\.js$',
        r'\.pb\.ts$'
    ]
    
    def __init__(
        self,
        model: str = "qwen2.5-coder:7b-instruct",
        base_url: str = "http://localhost:11434",
        max_workers: int = 4,
        batch_size: int = 16,
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        """Initialize the LLM enricher."""
        self.model = model
        self.base_url = base_url
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Model parameters for deterministic output
        self.model_params = {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 256,
            "stream": False
        }
    
    def _is_generated_file(self, file_path: str, chunk: Dict[str, Any]) -> bool:
        """Check if file/chunk should be treated as generated."""
        # Check explicit flag
        if chunk.get("is_generated", False):
            return True
        
        # Check file path patterns
        for pattern in self.GENERATED_PATTERNS:
            if re.search(pattern, file_path, re.IGNORECASE):
                return True
        
        return False
    
    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate text to max_chars with ellipsis."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars-3] + "..."
    
    def _hash_content(self, content: str) -> str:
        """Generate SHA256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _call_ollama(self, system_prompt: str, user_prompt: str, retry_count: int = 0) -> str:
        """Call Ollama API with retry logic."""
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            **self.model_params
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = 0.5 * (2 ** retry_count)  # Exponential backoff
                time.sleep(wait_time)
                return self._call_ollama(system_prompt, user_prompt, retry_count + 1)
            else:
                raise Exception(f"LLM call failed after {self.max_retries} retries: {e}")
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, handling markdown fences."""
        # Remove markdown code fences if present
        response = re.sub(r'^```(?:json)?\s*\n?', '', response, flags=re.MULTILINE)
        response = re.sub(r'\n?```\s*$', '', response, flags=re.MULTILINE)
        
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response: {response[:200]}...")
    
    def generate_file_synopsis(self, file_path: str, language: str, file_text: str) -> str:
        """Generate file-level synopsis in Dutch."""
        from .prompts import get_file_synopsis_prompt
        
        file_text_excerpt = self._truncate_text(file_text, 4000)
        system_prompt, user_prompt = get_file_synopsis_prompt(
            file_path, language, file_text_excerpt
        )
        
        response = self._call_ollama(system_prompt, user_prompt)
        return response.strip()
    
    def enrich_chunk(
        self,
        chunk: Dict[str, Any],
        file_synopsis_nl: str,
        file_synopsis_hash: str
    ) -> Dict[str, Any]:
        """Enrich a single chunk with Dutch summary and keywords."""
        from .prompts import get_per_chunk_prompt
        
        # Check if this is a generated file
        file_path = chunk.get("path", chunk.get("rel_path", ""))
        if self._is_generated_file(file_path, chunk):
            return self._generate_template_enrichment(chunk, file_synopsis_hash)
        
        # Prepare chunk metadata
        ast_path = chunk.get("ast_path", "")
        node_kind = chunk.get("node_kind", "")
        type_name = chunk.get("type_name", "")
        function_name = chunk.get("function_name", "")
        method_name = chunk.get("method_name", "")
        
        function_or_method = function_name or method_name or ""
        
        chunk_text = chunk.get("text", "")
        chunk_text_excerpt = self._truncate_text(chunk_text, 2000)
        
        # Generate enrichment
        system_prompt, user_prompt = get_per_chunk_prompt(
            file_synopsis_nl, ast_path, node_kind, type_name,
            function_or_method, chunk_text_excerpt
        )
        
        try:
            response = self._call_ollama(system_prompt, user_prompt)
            result = self._parse_json_response(response)
            
            # Validate result structure
            if "summary_nl" not in result or "keywords_nl" not in result:
                raise ValueError("Missing required fields in LLM response")
            
            # Validate summary length
            summary_nl = result["summary_nl"]
            if len(summary_nl) > 160:
                summary_nl = summary_nl[:157] + "..."
            
            # Validate keywords
            keywords_nl = result["keywords_nl"]
            if not isinstance(keywords_nl, list) or len(keywords_nl) < 5 or len(keywords_nl) > 10:
                # Fallback to a reasonable number
                keywords_nl = keywords_nl[:10] if len(keywords_nl) > 10 else keywords_nl
            
            chunk_text_hash = self._hash_content(chunk_text)
            
            return {
                "summary_nl": summary_nl,
                "keywords_nl": keywords_nl,
                "enrich_provenance": {
                    "model": self.model,
                    "created_at": datetime.now().isoformat(),
                    "file_synopsis_hash": file_synopsis_hash,
                    "chunk_text_hash": chunk_text_hash,
                    "input_lang": "nl",
                    "skipped_reason": None
                }
            }
            
        except Exception as e:
            # Retry once with explicit JSON instruction
            try:
                system_prompt, user_prompt = get_per_chunk_prompt(
                    file_synopsis_nl, ast_path, node_kind, type_name,
                    function_or_method, chunk_text_excerpt, is_retry=True
                )
                response = self._call_ollama(system_prompt, user_prompt)
                result = self._parse_json_response(response)
                
                summary_nl = result["summary_nl"]
                if len(summary_nl) > 160:
                    summary_nl = summary_nl[:157] + "..."
                
                keywords_nl = result["keywords_nl"]
                if not isinstance(keywords_nl, list) or len(keywords_nl) < 5 or len(keywords_nl) > 10:
                    keywords_nl = keywords_nl[:10] if len(keywords_nl) > 10 else keywords_nl
                
                chunk_text_hash = self._hash_content(chunk_text)
                
                return {
                    "summary_nl": summary_nl,
                    "keywords_nl": keywords_nl,
                    "enrich_provenance": {
                        "model": self.model,
                        "created_at": datetime.now().isoformat(),
                        "file_synopsis_hash": file_synopsis_hash,
                        "chunk_text_hash": chunk_text_hash,
                        "input_lang": "nl",
                        "skipped_reason": None
                    }
                }
                
            except Exception as retry_error:
                # Return error enrichment
                chunk_text_hash = self._hash_content(chunk_text)
                return {
                    "summary_nl": f"LLM verrijking mislukt: {str(e)[:100]}",
                    "keywords_nl": ["error", "llm", "mislukt", "verrijking", "chunk"],
                    "enrich_provenance": {
                        "model": self.model,
                        "created_at": datetime.now().isoformat(),
                        "file_synopsis_hash": file_synopsis_hash,
                        "chunk_text_hash": chunk_text_hash,
                        "input_lang": "nl",
                        "skipped_reason": "llm_error"
                    }
                }
    
    def _generate_template_enrichment(self, chunk: Dict[str, Any], file_synopsis_hash: str) -> Dict[str, Any]:
        """Generate templated enrichment for generated files."""
        file_path = chunk.get("path", chunk.get("rel_path", ""))
        chunk_text = chunk.get("text", "")
        chunk_text_hash = self._hash_content(chunk_text)
        
        # Determine template based on file type
        if re.search(r'\.pb\.go$', file_path, re.IGNORECASE):
            summary_nl = "Gegenereerde protobuf-berichten en hulpfuncties; geen domeinlogica."
            keywords_nl = ["protobuf", "gegenereerd", "berichten", "hulpmethoden", "grpc"]
        elif re.search(r'\.gen\.go$', file_path, re.IGNORECASE):
            summary_nl = "Gegenereerde Go-code; geen handmatige wijzigingen."
            keywords_nl = ["gegenereerd", "go", "code", "automatisch", "tool"]
        else:
            summary_nl = "Gegenereerde code; geen domeinlogica."
            keywords_nl = ["gegenereerd", "code", "automatisch", "tool", "geen-logica"]
        
        return {
            "summary_nl": summary_nl,
            "keywords_nl": keywords_nl,
            "enrich_provenance": {
                "model": self.model,
                "created_at": datetime.now().isoformat(),
                "file_synopsis_hash": file_synopsis_hash,
                "chunk_text_hash": chunk_text_hash,
                "input_lang": "nl",
                "skipped_reason": "generated"
            }
        }
    
    def enrich_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        file_synopsis_nl: str,
        file_synopsis_hash: str
    ) -> List[Dict[str, Any]]:
        """Enrich a batch of chunks with parallel processing."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self.enrich_chunk, chunk, file_synopsis_nl, file_synopsis_hash): chunk
                for chunk in chunks
            }
            
            # Collect results in order
            chunk_to_result = {}
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    chunk_to_result[id(chunk)] = result
                except Exception as e:
                    # Create error result
                    chunk_text = chunk.get("text", "")
                    chunk_text_hash = self._hash_content(chunk_text)
                    chunk_to_result[id(chunk)] = {
                        "summary_nl": f"Batch verrijking mislukt: {str(e)[:100]}",
                        "keywords_nl": ["error", "batch", "mislukt", "verrijking"],
                        "enrich_provenance": {
                            "model": self.model,
                            "created_at": datetime.now().isoformat(),
                            "file_synopsis_hash": file_synopsis_hash,
                            "chunk_text_hash": chunk_text_hash,
                            "input_lang": "nl",
                            "skipped_reason": "llm_error"
                        }
                    }
            
            # Return results in original order
            for chunk in chunks:
                results.append(chunk_to_result[id(chunk)])
        
        return results
