"""
LLM Filter Engine using Ollama with variance guard and hybrid scoring.

Scores chunks for relevance with enhanced prompts, error handling, and fallback logic.
"""
import json
import hashlib
import logging
import requests
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from .schemas import Chunk, FilterDecision, FilterResult
from .cache import Cache
from .truncation import truncate_for_filter

logger = logging.getLogger(__name__)

# Secret patterns to skip (security)
SECRET_PATTERNS = [
    r'AKIA[0-9A-Z]{16}',  # AWS keys
    r'["\']?password["\']?\s*[:=]\s*["\'][^"\']+["\']',  # Passwords
    r'["\']?api[_-]?key["\']?\s*[:=]\s*["\'][^"\']+["\']',  # API keys
]


class FilterEngine:
    """LLM-based filter using Ollama."""
    
    def __init__(self, cache: Cache, settings):
        """
        Initialize filter engine.
        
        Args:
            cache: Cache instance
            settings: Configuration settings
        """
        self.cache = cache
        self.settings = settings
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with anti-speculation guardrails."""
        return """You are a careful code-judge. Decide if the chunk can help answer the question.

STRICT RULES:
- Judge ONLY from the visible code and provided metadata.
- Prefer chunks with concrete symbols (functions, methods, types) over file headers/import-only chunks.
- Unless the user's question is explicitly about imports, dependencies, build, modules/package layout,
  treat package-only or import-only chunks as NOT helpful.
- Do not speculate beyond the code.
- Score based on direct relevance to the question.
- Return ONLY valid JSON, no other text.

JSON Schema:
{
  "score": <0-4 integer>,
  "keep": <true|false boolean>,
  "why": "<7-20 words explaining score>"
}

Score Guide:
- 0: Completely irrelevant (e.g., package/import-only for a non-import question)
- 1: Tangentially related, unlikely to help
- 2: Possibly relevant, may provide context
- 3: Relevant, likely helpful
- 4: Directly answers or critical to answer

Special cases:
- If the question mentions imports/dependencies/modules/go.mod/package structure, evaluate import-only chunks normally.
- Otherwise, package-only or import-only chunks should get score 0 and keep=false.

Return JSON ONLY."""
    
    def _build_user_prompt(self, query: str, chunk: Chunk) -> str:
        """Build user prompt with chunk context."""
        # Truncate code
        code_trunc = truncate_for_filter(chunk, self.settings.FILTER_MAX_CODE_CHARS)
        
        # Build prompt
        prompt = f"""Question:
{query}

Chunk metadata:
- File: {chunk.rel_path}
- Lines: {chunk.start_line}-{chunk.end_line}
- Language: {chunk.language}"""
        
        if chunk.summary_en:
            prompt += f"\n- Summary: {chunk.summary_en}"
        
        prompt += f"""

Chunk text (may be truncated):
```{chunk.language}
{code_trunc}
```

Evaluate relevance and return JSON ONLY:
{{"score": <0-4>, "keep": <true|false>, "why": "<explanation>"}}"""
        
        return prompt
    
    def _contains_secrets(self, text: str) -> bool:
        """Check if text contains potential secrets."""
        import re
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _call_ollama(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> Dict[str, Any]:
        """
        Call Ollama API.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
        
        Returns:
            Parsed JSON response
        """
        payload = {
            "model": self.settings.OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(
                self.settings.OLLAMA_URL,
                json=payload,
                timeout=self.settings.FILTER_TIMEOUT_S
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract content
            content = data.get("message", {}).get("content", "")
            
            # Parse JSON
            return json.loads(content)
        
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, response: {content[:200]}")
            raise
        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise
    
    def score_chunk(self, query: str, chunk: Chunk) -> FilterDecision:
        """
        Score a single chunk for relevance.
        
        Args:
            query: User query
            chunk: Chunk to score
        
        Returns:
            FilterDecision with score and explanation
        """
        # Check for secrets
        if chunk.text and self._contains_secrets(chunk.text):
            logger.warning(f"Chunk {chunk.id} contains potential secrets, skipping")
            return FilterDecision(score=0, keep=False, why="security-skip")
        
        # Build cache key
        prompt_hash = hashlib.sha256(self.system_prompt.encode()).hexdigest()[:8]
        cache_key = f"filter:{self.settings.PROMPT_VERSION}:{prompt_hash}:{chunk.hashable_key(query, self.settings.PROMPT_VERSION)}"
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for chunk {chunk.id}")
            return FilterDecision(**cached)
        
        # Build prompts
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_prompt(query, chunk)}
        ]
        
        # Call LLM with retry on parse error
        try:
            result = self._call_ollama(messages, temperature=0.1)
            logger.debug(f"Chunk {chunk.id}: LLM returned {result}")
        except (json.JSONDecodeError, KeyError) as e:
            # Retry with temperature=0 for more deterministic output
            logger.warning(f"JSON parse error for chunk {chunk.id}: {e}, retrying with temperature=0")
            try:
                result = self._call_ollama(messages, temperature=0.0)
                logger.debug(f"Chunk {chunk.id}: Retry returned {result}")
            except Exception as e:
                logger.error(f"Filter failed for chunk {chunk.id} after retry: {e}")
                result = {"score": 0, "keep": False, "why": "parse-failed"}
        except Exception as e:
            logger.error(f"Filter failed for chunk {chunk.id}: {e}")
            result = {"score": 0, "keep": False, "why": "llm-error"}
        
        # Validate and create decision
        try:
            decision = FilterDecision(
                score=int(result.get("score", 0)),
                keep=bool(result.get("keep", False)),
                why=str(result.get("why", "unknown"))[:100]  # Cap length
            )
        except Exception as e:
            logger.error(f"Invalid filter response for chunk {chunk.id}: {e}")
            decision = FilterDecision(score=0, keep=False, why="invalid-response")
        
        # Cache result
        self.cache.set(cache_key, decision.model_dump(), ttl_s=self.settings.CACHE_TTL_S)
        
        return decision
    
    def filter_chunks(self, query: str, chunks: List[Chunk]) -> List[FilterResult]:
        """
        Filter chunks with variance guard and hybrid scoring.
        
        Args:
            query: User query
            chunks: Chunks to filter
        
        Returns:
            List of FilterResult for chunks that pass threshold
        """
        if not chunks:
            return []
        
        logger.info(f"Filtering {len(chunks)} chunks (threshold>={self.settings.FILTER_THRESHOLD})")
        
        # Score chunks in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.settings.FILTER_PARALLELISM) as executor:
            future_to_chunk = {
                executor.submit(self.score_chunk, query, chunk): chunk
                for chunk in chunks
            }
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    decision = future.result()
                    results.append(FilterResult(
                        chunk=chunk,
                        score=decision.score,
                        why=decision.why,
                        cached=False  # TODO: track cache hits
                    ))
                except Exception as e:
                    logger.error(f"Filter exception for chunk {chunk.id}: {e}")
                    results.append(FilterResult(
                        chunk=chunk,
                        score=0,
                        why="exception",
                        cached=False
                    ))
        
        # Compute score statistics for variance guard
        scores = [r.score for r in results]
        mean_score = statistics.mean(scores) if scores else 0
        stddev_score = statistics.stdev(scores) if len(scores) > 1 else 0
        
        # Log score distribution for diagnostics
        score_dist = {}
        for r in results:
            score_dist[r.score] = score_dist.get(r.score, 0) + 1
        logger.info(f"Score stats: mean={mean_score:.2f}, stddev={stddev_score:.2f}")
        logger.info(f"Score distribution: {dict(sorted(score_dist.items()))}")
        
        # Variance Guard: If scores are too uniform, bypass filtering
        if stddev_score < self.settings.FILTER_VARIANCE_BYPASS:
            logger.warning(f"Score variance too low ({stddev_score:.3f} < {self.settings.FILTER_VARIANCE_BYPASS})")
            logger.warning("Filter is not discriminating - preserving retriever order")
            
            # Use hybrid scoring with alpha=0 (pure retriever order)
            for r in results:
                retriever_norm = r.chunk.retriever_score or 0.0
                r.hybrid_score = retriever_norm
            
            # Sort by retriever score to preserve original quality
            results.sort(key=lambda r: (-r.hybrid_score, r.chunk.provenance_id))
            return results
        
        # Apply threshold
        survivors = [r for r in results if r.score >= self.settings.FILTER_THRESHOLD]
        logger.info(f"Filter: {len(survivors)}/{len(results)} passed (threshold>={self.settings.FILTER_THRESHOLD})")
        
        # Relax threshold if too few survivors
        if len(survivors) < self.settings.FILTER_MIN_SURVIVORS:
            logger.warning(f"Only {len(survivors)} survivors, relaxing threshold to >=2")
            survivors = [r for r in results if r.score >= 2]
            logger.info(f"After relaxation: {len(survivors)} survivors")
            
            if len(survivors) < self.settings.FILTER_MIN_SURVIVORS:
                logger.warning(f"Still only {len(survivors)}, relaxing to >=1")
                survivors = [r for r in results if r.score >= 1]
                logger.info(f"After second relaxation: {len(survivors)} survivors")
                
                # Last resort: Take top chunks by retriever score
                if len(survivors) < self.settings.FILTER_MIN_SURVIVORS:
                    logger.warning(f"Still only {len(survivors)}, falling back to top retriever scores")
                    survivors = sorted(results, key=lambda r: -(r.chunk.retriever_score or 0))[:self.settings.PIPELINE_TOPK_EVIDENCE * 2]
        
        # Hybrid Scoring: Combine filter score with retriever score
        # alpha controls weight of filter score (0.3 = 30% filter, 70% retriever)
        alpha = self.settings.HYBRID_ALPHA
        for r in survivors:
            filter_norm = r.score / 4.0  # Normalize 0-4 to 0-1
            retriever_norm = r.chunk.retriever_score or 0.0
            r.hybrid_score = retriever_norm + (alpha * filter_norm)
        
        # Sort by hybrid score (combining filter and retriever quality)
        survivors.sort(key=lambda r: (-r.hybrid_score, r.chunk.provenance_id))
        
        logger.info(f"Final: {len(survivors)} chunks with hybrid scoring (alpha={alpha})")
        
        return survivors

