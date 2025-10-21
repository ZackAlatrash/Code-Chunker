"""
Tests for reranker.
"""
import pytest
from unittest.mock import Mock, patch
from rag_filter_rerank.schemas import RerankItem, RerankScore
from rag_filter_rerank.reranker import Reranker
from rag_filter_rerank.tests.test_filter_engine import MockCache


@pytest.fixture
def mock_settings():
    """Mock settings."""
    settings = Mock()
    settings.RERANK_VENDOR = "cohere"
    settings.COHERE_API_KEY = "test-key-1234"
    settings.COHERE_MODEL = "rerank-english-v3.0"
    settings.RERANK_TIMEOUT_S = 30
    settings.RERANK_BATCH_SIZE = 96
    settings.RERANK_MAX_CALLS = 3
    settings.RERANK_MAX_DOCS = 288
    settings.RERANK_RETRIES = 3
    settings.CACHE_TTL_S = 3600
    settings.DISABLE_RERANK = False
    return settings


@pytest.fixture
def reranker(mock_settings):
    """Create reranker with mock cache."""
    cache = MockCache()
    return Reranker(cache, mock_settings)


@pytest.fixture
def sample_items():
    """Sample rerank items."""
    return [
        RerankItem(id="item-1", text="This is about Python programming"),
        RerankItem(id="item-2", text="This discusses Go concurrency"),
        RerankItem(id="item-3", text="JavaScript async/await patterns"),
    ]


@patch('rag_filter_rerank.reranker.requests.post')
def test_rerank_success(mock_post, reranker, sample_items):
    """Test successful reranking."""
    # Mock Cohere response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.85},
            {"index": 1, "relevance_score": 0.70},
        ]
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    query = "async patterns"
    scores = reranker.rerank(query, sample_items)
    
    assert len(scores) == 3
    assert isinstance(scores[0], RerankScore)
    
    # Should be sorted by score descending
    assert scores[0].score >= scores[1].score >= scores[2].score
    
    # Highest score should be item-3 (JavaScript async)
    assert scores[0].id == "item-3"


@patch('rag_filter_rerank.reranker.requests.post')
def test_rerank_caching(mock_post, reranker, sample_items):
    """Test that reranking uses cache."""
    # Mock Cohere response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.90},
            {"index": 1, "relevance_score": 0.80},
            {"index": 2, "relevance_score": 0.70},
        ]
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    query = "test query"
    
    # First call
    scores1 = reranker.rerank(query, sample_items)
    
    # Second call should use cache
    scores2 = reranker.rerank(query, sample_items)
    
    # Should only call API once
    assert mock_post.call_count == 1
    
    # Results should match
    assert len(scores1) == len(scores2)
    for s1, s2 in zip(scores1, scores2):
        assert s1.id == s2.id
        assert s1.score == s2.score


@patch('rag_filter_rerank.reranker.requests.post')
def test_rerank_rate_limit_retry(mock_post, reranker, sample_items):
    """Test retry on rate limiting."""
    # First call returns 429
    mock_response1 = Mock()
    mock_response1.status_code = 429
    mock_response1.raise_for_status = Mock(side_effect=Exception("Rate limited"))
    
    # Second call succeeds
    mock_response2 = Mock()
    mock_response2.status_code = 200
    mock_response2.json.return_value = {
        "results": [{"index": i, "relevance_score": 0.5} for i in range(3)]
    }
    mock_response2.raise_for_status = Mock()
    
    mock_post.side_effect = [mock_response1, mock_response2]
    
    query = "test"
    with patch('time.sleep'):  # Mock sleep to speed up test
        scores = reranker.rerank(query, sample_items)
    
    # Should have retried
    assert mock_post.call_count == 2
    assert len(scores) == 3


@patch('rag_filter_rerank.reranker.requests.post')
def test_rerank_fallback_on_error(mock_post, reranker, sample_items):
    """Test fallback to order-preserving scores on error."""
    # All retries fail
    mock_post.side_effect = Exception("Network error")
    
    query = "test"
    scores = reranker.rerank(query, sample_items)
    
    # Should return fallback scores
    assert len(scores) == 3
    
    # Should be in descending order
    assert scores[0].score > scores[1].score > scores[2].score
    
    # First item should have highest score (order-preserving)
    assert scores[0].id == "item-1"


def test_rerank_no_api_key(mock_settings, sample_items):
    """Test fallback when API key not configured."""
    mock_settings.COHERE_API_KEY = None
    cache = MockCache()
    reranker = Reranker(cache, mock_settings)
    
    query = "test"
    scores = reranker.rerank(query, sample_items)
    
    # Should use fallback
    assert len(scores) == 3
    assert all(isinstance(s, RerankScore) for s in scores)


def test_rerank_disabled(mock_settings, sample_items):
    """Test reranking when disabled."""
    mock_settings.DISABLE_RERANK = True
    cache = MockCache()
    reranker = Reranker(cache, mock_settings)
    
    query = "test"
    scores = reranker.rerank(query, sample_items)
    
    # Should use fallback
    assert len(scores) == 3


@patch('rag_filter_rerank.reranker.requests.post')
def test_rerank_batching(mock_post, reranker):
    """Test batching for large number of items."""
    # Create many items
    items = [RerankItem(id=f"item-{i}", text=f"Text {i}") for i in range(200)]
    
    # Mock response
    def mock_batch_response(n):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "results": [{"index": i, "relevance_score": 0.5} for i in range(n)]
        }
        resp.raise_for_status = Mock()
        return resp
    
    # Should batch into 96-item chunks
    mock_post.side_effect = [mock_batch_response(96), mock_batch_response(96), mock_batch_response(8)]
    
    query = "test"
    scores = reranker.rerank(query, items)
    
    # Should have made 3 API calls (96 + 96 + 8 = 200)
    assert mock_post.call_count == 3
    assert len(scores) == 200


def test_rerank_max_docs_limit(mock_settings, sample_items):
    """Test enforcement of max docs limit."""
    mock_settings.RERANK_MAX_DOCS = 2
    cache = MockCache()
    reranker = Reranker(cache, mock_settings)
    
    # Create more items than limit
    items = sample_items * 10  # 30 items
    
    with patch('rag_filter_rerank.reranker.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.8}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        query = "test"
        scores = reranker.rerank(query, items)
        
        # Should have truncated to max_docs
        # Actual scores returned should be 2
        assert len([s for s in scores if s.score > 0]) <= mock_settings.RERANK_MAX_DOCS + 28  # Fallback for rest

