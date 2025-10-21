"""
Configuration for Qodo Embedding Pipeline

Pydantic settings for local-only code embedding using Qodo/Qodo-Embed-1-1.5B.
All settings can be overridden via environment variables.
"""
import os
from typing import Optional, Literal
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables."""
    
    # OpenSearch Settings
    OPENSEARCH_URL: str = Field(
        default="http://localhost:9200", 
        description="OpenSearch cluster URL"
    )
    OPENSEARCH_INDEX: str = Field(
        default="code_chunks_v5_qodo", 
        description="OpenSearch index name for Qodo embeddings"
    )
    OPENSEARCH_USERNAME: Optional[str] = Field(
        default=None, 
        description="OpenSearch username (if auth required)"
    )
    OPENSEARCH_PASSWORD: Optional[str] = Field(
        default=None, 
        description="OpenSearch password (if auth required)"
    )
    
    # Model Settings
    MODEL_ID: str = Field(
        default="Qodo/Qodo-Embed-1-1.5B", 
        description="HuggingFace model ID for Qodo embeddings"
    )
    MODEL_CACHE_DIR: Optional[str] = Field(
        default=None, 
        description="Custom cache directory for model downloads"
    )
    
    # Embedding Settings
    BATCH_SIZE: int = Field(
        default=16, 
        description="Batch size for embedding documents"
    )
    TRUNCATE_CHARS: int = Field(
        default=int(os.getenv("QODO_TRUNCATE_CHARS", "30000")), 
        description="Maximum characters to embed per document"
    )
    QUERY_PREFIX: str = Field(
        default="retrieve relevant code for: ", 
        description="Prefix added to queries for better retrieval alignment"
    )
    USE_NORMALIZE: bool = Field(
        default=True, 
        description="Whether to L2-normalize embeddings"
    )
    
    # Search Settings
    DEFAULT_K: int = Field(
        default=25, 
        description="Default number of results for KNN search"
    )
    MAX_K: int = Field(
        default=100, 
        description="Maximum allowed K for search"
    )
    
    # Index Settings
    INDEX_SHARDS: int = Field(
        default=1, 
        description="Number of shards for the index"
    )
    INDEX_REPLICAS: int = Field(
        default=0, 
        description="Number of replicas for the index"
    )
    
    # Performance Settings
    REQUEST_TIMEOUT: int = Field(
        default=30, 
        description="Request timeout in seconds"
    )
    MAX_RETRIES: int = Field(
        default=3, 
        description="Maximum retries for failed requests"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @field_validator("BATCH_SIZE")
    @classmethod
    def validate_batch_size(cls, v):
        if v < 1:
            raise ValueError("BATCH_SIZE must be at least 1")
        return v
    
    @field_validator("TRUNCATE_CHARS")
    @classmethod
    def validate_truncate_chars(cls, v):
        if v < 100:
            raise ValueError("TRUNCATE_CHARS must be at least 100")
        return v
    
    @field_validator("DEFAULT_K")
    @classmethod
    def validate_default_k(cls, v):
        if v < 1:
            raise ValueError("DEFAULT_K must be at least 1")
        return v
    
    @field_validator("MAX_K")
    @classmethod
    def validate_max_k(cls, v):
        if v < 1:
            raise ValueError("MAX_K must be at least 1")
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
