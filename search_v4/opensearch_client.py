"""
OpenSearch client wrapper using requests.

Simple HTTP client for OpenSearch operations with authentication support.
"""
import requests
from typing import Dict, Any, List, Optional

from .config import OPENSEARCH_URL, OPENSEARCH_USER, OPENSEARCH_PASS, REQUEST_TIMEOUT


def _auth() -> Optional[tuple]:
    """Return HTTP Basic Auth tuple if credentials are configured."""
    if OPENSEARCH_USER:
        return (OPENSEARCH_USER, OPENSEARCH_PASS)
    return None


def search(index: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a search query against OpenSearch.
    
    Args:
        index: Index name to search
        body: Query body (dict)
    
    Returns:
        Search response as dict
    
    Raises:
        requests.HTTPError: If request fails
    """
    url = f"{OPENSEARCH_URL}/{index}/_search"
    try:
        r = requests.post(
            url,
            json=body,
            auth=_auth(),
            timeout=REQUEST_TIMEOUT,
            verify=False  # For self-signed certs in dev
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"OpenSearch search failed: {e}") from e


def mget(index: str, ids: List[str], _source: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Multi-get documents by ID.
    
    Args:
        index: Index name
        ids: List of document IDs
        _source: Optional list of fields to retrieve
    
    Returns:
        Multi-get response as dict
    
    Raises:
        requests.HTTPError: If request fails
    """
    url = f"{OPENSEARCH_URL}/{index}/_mget"
    # OpenSearch _mget expects docs array with _id fields
    docs = [{"_id": doc_id} for doc_id in ids]
    if _source is not None:
        for doc in docs:
            doc["_source"] = _source
    body: Dict[str, Any] = {"docs": docs}
    
    try:
        r = requests.post(
            url,
            json=body,
            auth=_auth(),
            timeout=REQUEST_TIMEOUT,
            verify=False
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"OpenSearch mget failed: {e}") from e

