"""
Cache interface with diskcache and Redis backends.

Supports TTL, cache busting, and namespace separation.
"""
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class Cache(ABC):
    """Abstract cache interface."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl_s: Optional[int] = None):
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def stats(self) -> dict:
        """Get cache statistics."""
        pass


class DiskcacheBackend(Cache):
    """Disk-based cache using diskcache library."""
    
    def __init__(self, cache_dir: str, ttl_s: int = 86400):
        """
        Initialize diskcache backend.
        
        Args:
            cache_dir: Directory for cache storage
            ttl_s: Default TTL in seconds
        """
        try:
            import diskcache
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = diskcache.Cache(str(self.cache_dir))
            self.default_ttl = ttl_s
            self.hits = 0
            self.misses = 0
            logger.info(f"Initialized diskcache at {cache_dir}")
        except ImportError:
            raise ImportError("diskcache not installed: pip install diskcache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            value = self.cache.get(key)
            if value is not None:
                self.hits += 1
                logger.debug(f"Cache HIT: {key[:50]}...")
            else:
                self.misses += 1
                logger.debug(f"Cache MISS: {key[:50]}...")
            return value
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl_s: Optional[int] = None):
        """Set value in disk cache."""
        try:
            expire = ttl_s if ttl_s is not None else self.default_ttl
            self.cache.set(key, value, expire=expire)
            logger.debug(f"Cache SET: {key[:50]}... (TTL={expire}s)")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        try:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
    
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "backend": "diskcache",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "size_mb": sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file()) / (1024 * 1024)
        }


class RedisBackend(Cache):
    """Redis-based cache."""
    
    def __init__(self, redis_url: str, ttl_s: int = 86400):
        """
        Initialize Redis backend.
        
        Args:
            redis_url: Redis connection URL
            ttl_s: Default TTL in seconds
        """
        try:
            import redis
            self.client = redis.from_url(redis_url, decode_responses=False)
            self.client.ping()  # Test connection
            self.default_ttl = ttl_s
            self.hits = 0
            self.misses = 0
            logger.info(f"Initialized Redis cache at {redis_url}")
        except ImportError:
            raise ImportError("redis not installed: pip install redis")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            value_bytes = self.client.get(key)
            if value_bytes:
                self.hits += 1
                value = json.loads(value_bytes.decode())
                logger.debug(f"Cache HIT: {key[:50]}...")
                return value
            else:
                self.misses += 1
                logger.debug(f"Cache MISS: {key[:50]}...")
                return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl_s: Optional[int] = None):
        """Set value in Redis."""
        try:
            expire = ttl_s if ttl_s is not None else self.default_ttl
            value_bytes = json.dumps(value).encode()
            self.client.setex(key, expire, value_bytes)
            logger.debug(f"Cache SET: {key[:50]}... (TTL={expire}s)")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def clear(self):
        """Clear all cache entries (DANGEROUS in shared Redis!)."""
        logger.warning("Redis clear() is disabled to prevent data loss in shared instances")
    
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        try:
            info = self.client.info("memory")
            used_mb = info.get("used_memory", 0) / (1024 * 1024)
        except:
            used_mb = 0
        return {
            "backend": "redis",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "size_mb": used_mb
        }


def get_cache(settings) -> Cache:
    """
    Factory function to create appropriate cache backend.
    
    Args:
        settings: Configuration settings
    
    Returns:
        Cache instance (Redis if REDIS_URL set, else diskcache)
    """
    # Check if cache should be busted
    if settings.BUST_CACHE:
        logger.info("BUST_CACHE=True, cache will be bypassed (in-memory only)")
        # Return a no-op cache that never hits
        class NoOpCache(Cache):
            def get(self, key: str) -> Optional[Any]:
                return None
            def set(self, key: str, value: Any, ttl_s: Optional[int] = None):
                pass
            def clear(self):
                pass
            def stats(self) -> dict:
                return {"backend": "noop", "hits": 0, "misses": 0, "hit_rate": "0%"}
        return NoOpCache()
    
    # Try Redis if URL provided
    if settings.REDIS_URL:
        try:
            return RedisBackend(settings.REDIS_URL, settings.CACHE_TTL_S)
        except Exception as e:
            logger.warning(f"Redis unavailable, falling back to diskcache: {e}")
    
    # Default to diskcache
    return DiskcacheBackend(settings.CACHE_DIR, settings.CACHE_TTL_S)

