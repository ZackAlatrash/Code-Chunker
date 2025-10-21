"""
Pydantic schemas for RAG Filter-Rerank system.

All models include provenance tracking for reproducibility.
"""
import hashlib
from typing import Optional, List
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Code chunk with metadata and optional provenance."""
    
    id: str
    repo_id: str
    rel_path: str
    start_line: int
    end_line: int
    language: str
    text: Optional[str] = None
    summary_en: Optional[str] = None
    chunk_hash: Optional[str] = None
    
    # Optional scores from retrieval/filtering/reranking
    retriever_score: Optional[float] = None
    filter_score: Optional[int] = None
    rerank_score: Optional[float] = None
    
    class Config:
        frozen = False  # Allow mutation for score updates
    
    def hashable_key(self, query: str, prompt_version: str) -> str:
        """
        Generate cache key for this chunk with query context.
        
        Args:
            query: User query string
            prompt_version: Prompt template version
        
        Returns:
            Deterministic cache key
        """
        key_parts = [
            prompt_version,
            hashlib.sha256(query.encode()).hexdigest()[:16],
            self.id,
            self.chunk_hash or hashlib.sha256((self.text or "").encode()).hexdigest()[:16]
        ]
        return ":".join(key_parts)
    
    @property
    def provenance_id(self) -> str:
        """Unique provenance ID for deterministic ordering."""
        chunk_hash = self.chunk_hash or hashlib.sha256((self.text or "").encode()).hexdigest()[:8]
        return f"{self.id}:{chunk_hash}"
    
    def __hash__(self):
        return hash(self.provenance_id)


class FilterDecision(BaseModel):
    """Decision from LLM filter."""
    
    score: int = Field(..., ge=0, le=4, description="Relevance score 0-4")
    keep: bool = Field(..., description="Whether to keep this chunk")
    why: str = Field(..., description="Brief explanation (7-20 words)")


class FilterResult(BaseModel):
    """Chunk with filter decision and hybrid score."""
    
    chunk: Chunk
    score: int
    why: str
    cached: bool = False
    hybrid_score: float = 0.0  # Combined filter + retriever score
    
    class Config:
        frozen = False


class RerankItem(BaseModel):
    """Item to send to reranker."""
    
    id: str
    text: str
    
    def __hash__(self):
        return hash(self.id)


class RerankScore(BaseModel):
    """Score from reranker."""
    
    id: str
    score: float
    cached: bool = False


class PipelineResult(BaseModel):
    """Complete pipeline execution result."""
    
    query: str
    recall_n: int
    filtered_m: int
    reranked: List[dict]  # List of {id, score, rel_path, start_line, end_line, provenance_id}
    answer: str
    evidence_k: int
    
    # Timing info (milliseconds)
    timing_ms: dict = Field(default_factory=dict)
    
    # Cache statistics
    cache_stats: dict = Field(default_factory=dict)
    
    # Feature flags used
    flags: dict = Field(default_factory=dict)

