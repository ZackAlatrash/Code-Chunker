"""
Content-hash caching system for enrichment results.

This module provides a simple disk cache using SQLite to store
enrichment results keyed by content hashes to enable idempotent runs.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any


class EnrichmentCache:
    """SQLite-based cache for enrichment results."""
    
    def __init__(self, cache_path: str = ".cache/enrichment.sqlite"):
        """Initialize the cache with the given path."""
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database with the required schema."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS enrichment_cache (
                    chunk_id TEXT,
                    file_sha TEXT,
                    chunk_text_hash TEXT,
                    file_synopsis_hash TEXT,
                    summary_nl TEXT,
                    keywords_nl TEXT,
                    model TEXT,
                    created_at TEXT,
                    PRIMARY KEY (chunk_id, file_sha, chunk_text_hash)
                )
            """)
            conn.commit()
    
    def _hash_content(self, content: str) -> str:
        """Generate SHA256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get(self, chunk_id: str, file_sha: str, chunk_text: str) -> Optional[Dict[str, Any]]:
        """Get cached enrichment result if available."""
        chunk_text_hash = self._hash_content(chunk_text)
        
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("""
                SELECT summary_nl, keywords_nl, model, created_at, file_synopsis_hash
                FROM enrichment_cache
                WHERE chunk_id = ? AND file_sha = ? AND chunk_text_hash = ?
            """, (chunk_id, file_sha, chunk_text_hash))
            
            row = cursor.fetchone()
            if row:
                summary_nl, keywords_nl_json, model, created_at, file_synopsis_hash = row
                keywords_nl = json.loads(keywords_nl_json)
                
                return {
                    "summary_nl": summary_nl,
                    "keywords_nl": keywords_nl,
                    "enrich_provenance": {
                        "model": model,
                        "created_at": created_at,
                        "file_synopsis_hash": file_synopsis_hash,
                        "chunk_text_hash": chunk_text_hash,
                        "input_lang": "nl",
                        "skipped_reason": None
                    }
                }
        
        return None
    
    def set(
        self,
        chunk_id: str,
        file_sha: str,
        chunk_text: str,
        file_synopsis_hash: str,
        summary_nl: str,
        keywords_nl: list[str],
        model: str
    ):
        """Store enrichment result in cache."""
        chunk_text_hash = self._hash_content(chunk_text)
        created_at = datetime.now().isoformat()
        
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO enrichment_cache
                (chunk_id, file_sha, chunk_text_hash, file_synopsis_hash, 
                 summary_nl, keywords_nl, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_id, file_sha, chunk_text_hash, file_synopsis_hash,
                summary_nl, json.dumps(keywords_nl), model, created_at
            ))
            conn.commit()
    
    def clear(self):
        """Clear all cached entries."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("DELETE FROM enrichment_cache")
            conn.commit()
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM enrichment_cache")
            total_entries = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT file_sha) FROM enrichment_cache")
            unique_files = cursor.fetchone()[0]
            
            return {
                "total_entries": total_entries,
                "unique_files": unique_files
            }
