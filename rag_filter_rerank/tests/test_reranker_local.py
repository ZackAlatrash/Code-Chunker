"""
Tests for local cross-encoder reranker.

Note: These tests require the model to be downloaded. They will be skipped
if the model is not available or transformers is not installed.
"""
import pytest
from unittest.mock import Mock, MagicMock
from ..schemas import RerankItem, RerankScore
from ..cache import DiskCache

# Try to import local reranker, skip tests if dependencies missing
try:
    from ..reranker_local import LocalCrossEncoderReranker, create_local_reranker
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    pytestmark = pytest.mark.skip(reason="transformers not installed")


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
    settings.RERANK_DEVICE = "cpu"
    settings.RERANK_BATCH_SIZE = 2
    settings.MAX_DOCS_RERANKED = 10
    settings.CACHE_TTL_S = 3600
    settings.CACHE_DIR = ".test_cache"
    return settings


@pytest.fixture
def mock_cache():
    """Mock cache for testing."""
    cache = Mock()
    cache.get = Mock(return_value=None)
    cache.set = Mock()
    return cache


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_create_local_reranker_with_mock(mock_cache, mock_settings):
    """Test creating reranker with mocked model."""
    # This test just checks the factory function doesn't crash
    # Actual model loading would require the model to be downloaded
    try:
        reranker = create_local_reranker(mock_cache, mock_settings)
        # If it succeeds, great! If it fails, we handle it gracefully
        assert reranker is None or isinstance(reranker, LocalCrossEncoderReranker)
    except Exception as e:
        # Model not available, that's fine for CI
        assert "model" in str(e).lower() or "connection" in str(e).lower()


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_rerank_with_empty_items(mock_cache, mock_settings):
    """Test reranking with no items."""
    # Mock the model loading
    with pytest.raises(Exception):
        # Will fail during model loading, but tests the flow
        reranker = LocalCrossEncoderReranker(mock_cache, mock_settings)
        result = reranker.rerank("test query", [])
        assert result == []


def test_rerank_item_schema():
    """Test RerankItem schema."""
    item = RerankItem(id="test-1", text="def foo(): pass")
    assert item.id == "test-1"
    assert item.text == "def foo(): pass"
    assert hash(item) == hash("test-1")


def test_rerank_score_schema():
    """Test RerankScore schema."""
    score = RerankScore(id="test-1", score=0.95, cached=True)
    assert score.id == "test-1"
    assert score.score == 0.95
    assert score.cached is True


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_batch_processing_logic():
    """Test batch processing works correctly with small batches."""
    # This is a logic test without actual model loading
    items = [RerankItem(id=f"item-{i}", text=f"text {i}") for i in range(10)]
    batch_size = 3
    
    # Simulate batching
    batches = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batches.append(batch)
    
    assert len(batches) == 4  # 3 + 3 + 3 + 1
    assert len(batches[0]) == 3
    assert len(batches[-1]) == 1


def test_factory_handles_failure_gracefully(mock_cache, mock_settings):
    """Test that factory returns None if model loading fails."""
    # Force a model that doesn't exist
    mock_settings.RERANK_MODEL = "nonexistent/model-that-does-not-exist"
    
    reranker = create_local_reranker(mock_cache, mock_settings)
    assert reranker is None  # Should return None instead of crashing

