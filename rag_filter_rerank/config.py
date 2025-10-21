"""
Configuration for RAG Filter-Rerank system (Local-only).

Uses Pydantic BaseSettings to load from environment variables with sensible defaults.
All reranking is done locally with HuggingFace transformers models.
"""
import os
from typing import Optional, Literal
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field, field_validator, model_validator


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables."""
    
    # Ollama Filter Settings
    OLLAMA_URL: str = Field(default="http://localhost:11434/api/chat", description="Ollama API URL")
    OLLAMA_MODEL: str = Field(default="qwen2.5-coder:7b-instruct", description="Ollama model for filtering (7B recommended)")
    
    # Filter Parameters
    FILTER_MAX_CODE_CHARS: int = Field(default=1200, description="Max code chars to send to filter")
    FILTER_PARALLELISM: int = Field(default=8, description="Concurrent filter requests")
    FILTER_THRESHOLD: int = Field(default=3, description="Minimum score to keep (0-4)")
    FILTER_MIN_SURVIVORS: int = Field(default=8, description="Min chunks after filtering")
    FILTER_TIMEOUT_S: int = Field(default=30, description="Filter request timeout")
    FILTER_VARIANCE_BYPASS: float = Field(default=0.15, description="StdDev threshold to bypass filtering")
    
    # Local Reranker Settings
    RERANK_VENDOR: Literal["local"] = Field(default="local", description="Reranker vendor (local only)")
    RERANK_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3", description="HuggingFace reranker model")
    RERANK_DEVICE: Literal["auto", "cuda", "cpu"] = Field(default="auto", description="Device for reranker")
    RERANK_BATCH_SIZE: int = Field(default=64, description="Batch size for local reranking")
    MAX_DOCS_RERANKED: int = Field(default=128, description="Max total docs to rerank")
    RERANK_TIMEOUT_S: int = Field(default=30, description="Rerank timeout (for future use)")
    
    # Cache Settings
    CACHE_DIR: str = Field(default=".rag_cache", description="Cache directory path")
    CACHE_TTL_S: int = Field(default=86400, description="Cache TTL in seconds (24h default)")
    BUST_CACHE: bool = Field(default=False, description="Force fresh runs, ignore cache")
    REDIS_URL: Optional[str] = Field(default=None, description="Redis URL (optional)")
    
    # Pipeline Settings
    PIPELINE_TOPN_RECALL: int = Field(default=100, description="Initial retrieval count")
    PIPELINE_TOPK_EVIDENCE: int = Field(default=8, description="Final evidence chunks")
    PROMPT_VERSION: str = Field(default="v2", description="Prompt template version")
    
    # Feature Flags
    RAG_FILTER_RERANK: Literal["on", "off", "shadow"] = Field(default="on", description="Pipeline mode")
    DISABLE_RERANK: bool = Field(default=False, description="Skip reranking stage")
    
    # Hybrid Scoring Parameters
    HYBRID_ALPHA: float = Field(default=0.3, description="Weight for filter score in hybrid (0.0-1.0)")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @field_validator("FILTER_THRESHOLD")
    @classmethod
    def validate_threshold(cls, v):
        if not 0 <= v <= 4:
            raise ValueError("FILTER_THRESHOLD must be between 0 and 4")
        return v
    
    @field_validator("FILTER_MIN_SURVIVORS")
    @classmethod
    def validate_min_survivors(cls, v):
        if v < 1:
            raise ValueError("FILTER_MIN_SURVIVORS must be at least 1")
        return v
    
    @field_validator("FILTER_VARIANCE_BYPASS")
    @classmethod
    def validate_variance_bypass(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("FILTER_VARIANCE_BYPASS must be between 0 and 1")
        return v
    
    @field_validator("HYBRID_ALPHA")
    @classmethod
    def validate_hybrid_alpha(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("HYBRID_ALPHA must be between 0 and 1")
        return v


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        # Load from .env if present
        from dotenv import load_dotenv
        load_dotenv()
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings (useful for testing)."""
    global _settings
    from dotenv import load_dotenv
    load_dotenv(override=True)
    _settings = Settings()
    return _settings
