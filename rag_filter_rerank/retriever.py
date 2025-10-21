"""
Pluggable retriever adapters for different data sources.

Supports search_v4 integration and JSON file loading.
"""
import sys
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List
from .schemas import Chunk

logger = logging.getLogger(__name__)


class RetrieverAdapter(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def search(self, query: str, top_n: int) -> List[Chunk]:
        """
        Search for relevant chunks.
        
        Args:
            query: User query
            top_n: Number of results to return
        
        Returns:
            List of Chunk objects
        """
        pass


class JsonFileAdapter(RetrieverAdapter):
    """Load chunks from a JSON/JSONL file."""
    
    def __init__(self, file_path: str):
        """
        Initialize with JSON file path.
        
        Args:
            file_path: Path to JSON or JSONL file
        """
        self.file_path = file_path
        self.chunks = self._load_chunks()
        logger.info(f"Loaded {len(self.chunks)} chunks from {file_path}")
    
    def _load_chunks(self) -> List[Chunk]:
        """Load chunks from file."""
        chunks = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # Try JSON first
            try:
                data = json.load(f)
                
                # Handle different formats
                if isinstance(data, list):
                    raw_chunks = data
                elif isinstance(data, dict):
                    # Check for various wrapper keys
                    raw_chunks = data.get("hits", data.get("chunks", data.get("results", [])))
                else:
                    raw_chunks = []
            except json.JSONDecodeError:
                # Try JSONL
                f.seek(0)
                raw_chunks = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            raw_chunks.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        
        # Convert to Chunk objects
        for raw in raw_chunks:
            try:
                # Handle both direct chunks and wrapped chunks (_source)
                if "_source" in raw:
                    raw = raw["_source"]
                
                chunk = Chunk(
                    id=raw.get("id", ""),
                    repo_id=raw.get("repo_id", ""),
                    rel_path=raw.get("rel_path", raw.get("path", "")),
                    start_line=int(raw.get("start_line", 0)),
                    end_line=int(raw.get("end_line", 0)),
                    language=raw.get("language", ""),
                    text=raw.get("text"),
                    summary_en=raw.get("summary_en"),
                    chunk_hash=raw.get("chunk_hash")
                )
                chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Failed to parse chunk: {e}")
                continue
        
        return chunks
    
    def search(self, query: str, top_n: int) -> List[Chunk]:
        """Return top-n chunks (no actual search, just truncate)."""
        return self.chunks[:top_n]


class SearchV4Adapter(RetrieverAdapter):
    """Adapter for search_v4 hybrid search."""
    
    def __init__(self, repo_ids: List[str] = None):
        """
        Initialize search_v4 adapter.
        
        Args:
            repo_ids: Optional list of repo IDs to filter
        """
        self.repo_ids = repo_ids or []
        
        # Try to import search_v4
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
            from search_v4.service import search_v4
            self.search_v4 = search_v4
            logger.info("search_v4 module loaded successfully")
        except ImportError as e:
            raise ImportError(f"search_v4 not available: {e}. Use JsonFileAdapter instead.")
    
    def search(self, query: str, top_n: int) -> List[Chunk]:
        """
        Search using search_v4 hybrid search.
        
        Args:
            query: User query
            top_n: Number of results
        
        Returns:
            List of Chunk objects
        """
        # Build minimal query plan
        plan = {
            "clarified_query": query,
            "identifiers": [],
            "file_hints": [],
            "language": None,
            "bm25_should": []
        }
        
        # Call search_v4
        try:
            results = self.search_v4(
                query=query,
                router_repo_ids=self.repo_ids,
                plan=plan,
                fetch_all_texts=True  # Need full text for filtering
            )
            
            # Convert results to Chunk objects
            chunks = []
            for result in results.get("results", [])[:top_n]:
                try:
                    chunk = Chunk(
                        id=result.get("id", ""),
                        repo_id=result.get("repo_id", ""),
                        rel_path=result.get("rel_path", ""),
                        start_line=int(result.get("start_line", 0)),
                        end_line=int(result.get("end_line", 0)),
                        language=result.get("language", ""),
                        text=result.get("text"),
                        summary_en=result.get("summary_en"),
                        chunk_hash=result.get("chunk_hash")
                    )
                    chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Failed to convert search result: {e}")
                    continue
            
            logger.info(f"search_v4 returned {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            logger.error(f"search_v4 failed: {e}")
            return []


def get_retriever(chunks_json: str = None, repo_ids: List[str] = None) -> RetrieverAdapter:
    """
    Factory function to get appropriate retriever.
    
    Args:
        chunks_json: Optional path to JSON file
        repo_ids: Optional repo IDs for search_v4
    
    Returns:
        RetrieverAdapter instance
    """
    if chunks_json:
        return JsonFileAdapter(chunks_json)
    else:
        try:
            return SearchV4Adapter(repo_ids)
        except ImportError:
            raise ValueError(
                "No retriever available. Provide --chunks-json or ensure search_v4 is installed."
            )

