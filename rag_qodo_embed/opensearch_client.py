"""
OpenSearch client for Qodo embeddings.

Provides bulk indexing and KNN search capabilities for the Qodo embedding pipeline.
"""
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

from .config import Settings

logger = logging.getLogger(__name__)


class QodoOpenSearchClient:
    """OpenSearch client for Qodo embedding operations."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize OpenSearch client.
        
        Args:
            settings: Configuration settings (uses default if None)
        """
        self.settings = settings or Settings()
        self.base_url = self.settings.OPENSEARCH_URL.rstrip('/')
        self.index_name = self.settings.OPENSEARCH_INDEX
        
        # Setup authentication if provided
        self.auth = None
        if self.settings.OPENSEARCH_USERNAME and self.settings.OPENSEARCH_PASSWORD:
            self.auth = (self.settings.OPENSEARCH_USERNAME, self.settings.OPENSEARCH_PASSWORD)
        
        # Request session for connection pooling
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.verify = False  # Disable SSL verification for local dev
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make HTTP request to OpenSearch.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If request fails
        """
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        # Set default timeout
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.settings.REQUEST_TIMEOUT
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"OpenSearch request failed: {method} {url} - {e}")
            raise
    
    def create_index(self, force: bool = False) -> bool:
        """
        Create the OpenSearch index with proper mapping for Qodo embeddings.
        
        Args:
            force: Whether to delete existing index first
            
        Returns:
            True if index was created successfully
        """
        # Check if index exists
        try:
            self._make_request('HEAD', f'/{self.index_name}')
            if not force:
                logger.info(f"Index {self.index_name} already exists")
                return True
            else:
                logger.info(f"Deleting existing index {self.index_name}")
                self._make_request('DELETE', f'/{self.index_name}')
        except requests.RequestException:
            # Index doesn't exist, which is fine
            pass
        
        # Determine embedding dimension dynamically from embedder
        try:
            from .embedder import QodoEmbedder
            dim = QodoEmbedder(self.settings).get_embedding_dim()
        except Exception:
            dim = 1536  # Fallback

        # Create index with mapping
        mapping = {
            "settings": {
                "number_of_shards": self.settings.INDEX_SHARDS,
                "number_of_replicas": self.settings.INDEX_REPLICAS,
                "index.knn": True,
                "index.knn.algo_param.ef_search": 100
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "repo_id": {"type": "keyword"},
                    "rel_path": {"type": "keyword"},
                    "path": {"type": "keyword"},
                    "abs_path": {"type": "keyword"},
                    "ext": {"type": "keyword"},
                    "language": {"type": "keyword"},
                    "package": {"type": "keyword"},
                    "chunk_number": {"type": "integer"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "text": {"type": "text"},
                    "summary_en": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene"
                        }
                    },
                    "model_name": {"type": "keyword"},
                    "model_version": {"type": "keyword"},
                    "embedding_type": {"type": "keyword"},
                    "raw_text_hash": {"type": "keyword"},
                    "truncated_text_hash": {"type": "keyword"},
                    "created_at": {"type": "date"}
                }
            }
        }
        
        try:
            self._make_request(
                'PUT', 
                f'/{self.index_name}',
                json=mapping
            )
            logger.info(f"✅ Created index {self.index_name}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to create index {self.index_name}: {e}")
            return False

    def sample_one(self, fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch a single sample document (without large embedding), returning _source.
        """
        body = {
            "size": 1,
            "_source": fields or True,
            "query": {"match_all": {}}
        }
        try:
            resp = self._make_request('POST', f'/{self.index_name}/_search', json=body)
            data = resp.json()
            hits = data.get('hits', {}).get('hits', [])
            if not hits:
                return None
            return hits[0].get('_source', {})
        except requests.RequestException:
            return None
    
    def bulk_index(self, docs: List[Dict[str, Any]]) -> bool:
        """
        Bulk index documents into OpenSearch.
        
        Args:
            docs: List of documents to index
            
        Returns:
            True if all documents were indexed successfully
        """
        if not docs:
            logger.warning("No documents to index")
            return True
        
        logger.info(f"Bulk indexing {len(docs)} documents to {self.index_name}")
        
        # Prepare bulk request body
        bulk_body = []
        for doc in docs:
            # Index action
            bulk_body.append({
                "index": {
                    "_index": self.index_name,
                    "_id": doc.get("id", "")
                }
            })
            # Document
            bulk_body.append(doc)
        
        # Convert to NDJSON format
        ndjson_lines = []
        for item in bulk_body:
            ndjson_lines.append(json.dumps(item))
        ndjson_body = '\n'.join(ndjson_lines) + '\n'
        
        try:
            response = self._make_request(
                'POST',
                '/_bulk',
                data=ndjson_body,
                headers={'Content-Type': 'application/x-ndjson'}
            )
            
            result = response.json()
            
            # Check for errors
            if result.get('errors'):
                error_count = 0
                for item in result.get('items', []):
                    if 'index' in item and 'error' in item['index']:
                        error_count += 1
                        logger.error(f"Index error: {item['index']['error']}")
                
                if error_count > 0:
                    logger.error(f"Bulk index failed with {error_count} errors")
                    return False
            
            logger.info(f"✅ Successfully indexed {len(docs)} documents")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Bulk index failed: {e}")
            return False
    
    def knn_search(
        self, 
        query_vec: List[float], 
        k: int = 25,
        _source: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform KNN search using query vector.
        
        Args:
            query_vec: Query embedding vector
            k: Number of results to return
            _source: Fields to return in results
            
        Returns:
            List of search results
        """
        if k > self.settings.MAX_K:
            k = self.settings.MAX_K
            logger.warning(f"K limited to {self.settings.MAX_K}")
        
        # Default source fields
        if _source is None:
            _source = [
                "id", "repo_id", "rel_path", "path", "start_line", "end_line", 
                "summary_en", "language", "package", "chunk_number"
            ]
        
        search_body = {
            "size": k,
            "_source": _source,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vec,
                        "k": k
                    }
                }
            }
        }
        
        try:
            response = self._make_request(
                'POST',
                f'/{self.index_name}/_search',
                json=search_body
            )
            
            result = response.json()
            hits = result.get('hits', {}).get('hits', [])
            
            # Extract source documents
            docs = []
            for hit in hits:
                doc = hit.get('_source', {})
                doc['_score'] = hit.get('_score', 0.0)
                docs.append(doc)
            
            logger.info(f"KNN search returned {len(docs)} results")
            return docs
            
        except requests.RequestException as e:
            logger.error(f"KNN search failed: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index stats
        """
        try:
            response = self._make_request('GET', f'/{self.index_name}/_stats')
            stats = response.json()
            
            indices = stats.get('indices', {})
            if self.index_name in indices:
                return indices[self.index_name]
            return {}
            
        except requests.RequestException as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    def delete_index(self) -> bool:
        """
        Delete the index.
        
        Returns:
            True if index was deleted successfully
        """
        try:
            self._make_request('DELETE', f'/{self.index_name}')
            logger.info(f"✅ Deleted index {self.index_name}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to delete index {self.index_name}: {e}")
            return False
