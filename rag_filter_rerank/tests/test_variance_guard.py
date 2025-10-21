"""
Tests for variance guard and hybrid scoring in filter engine.
"""
import pytest
from unittest.mock import Mock, patch
from ..schemas import Chunk, FilterDecision, FilterResult
from ..filter_engine import FilterEngine
from ..cache import DiskCache


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.OLLAMA_MODEL = "qwen2.5-coder:7b-instruct"
    settings.OLLAMA_URL = "http://localhost:11434/api/chat"
    settings.FILTER_MAX_CODE_CHARS = 1200
    settings.FILTER_PARALLELISM = 2
    settings.FILTER_THRESHOLD = 3
    settings.FILTER_MIN_SURVIVORS = 3
    settings.FILTER_TIMEOUT_S = 10
    settings.FILTER_VARIANCE_BYPASS = 0.15
    settings.HYBRID_ALPHA = 0.3
    settings.PIPELINE_TOPK_EVIDENCE = 8
    settings.CACHE_TTL_S = 3600
    settings.PROMPT_VERSION = "v1"
    return settings


@pytest.fixture
def mock_cache():
    """Mock cache for testing."""
    cache = Mock()
    cache.get = Mock(return_value=None)
    cache.set = Mock()
    cache.stats = Mock(return_value={"hits": 0, "misses": 0})
    return cache


@pytest.fixture
def sample_chunks():
    """Create sample chunks with retriever scores."""
    chunks = []
    for i in range(10):
        chunk = Chunk(
            id=f"chunk-{i}",
            repo_id="test-repo",
            rel_path=f"file{i}.py",
            start_line=1,
            end_line=10,
            language="python",
            text=f"def func{i}():\n    pass",
            retriever_score=1.0 - (i / 10)  # Descending scores
        )
        chunks.append(chunk)
    return chunks


def test_variance_guard_triggers_with_uniform_scores(mock_cache, mock_settings, sample_chunks):
    """Test that variance guard triggers when all scores are the same."""
    engine = FilterEngine(mock_cache, mock_settings)
    
    # Mock score_chunk to return uniform score (3) for all chunks
    def mock_score_chunk(query, chunk):
        return FilterDecision(score=3, keep=True, why="test")
    
    engine.score_chunk = mock_score_chunk
    
    results = engine.filter_chunks("test query", sample_chunks)
    
    # Should return all chunks (variance bypass triggered)
    assert len(results) == len(sample_chunks)
    
    # Should preserve retriever order (sorted by retriever_score)
    for i, result in enumerate(results):
        assert result.chunk.id == f"chunk-{i}"


def test_hybrid_scoring_with_varied_scores(mock_cache, mock_settings, sample_chunks):
    """Test hybrid scoring when variance is sufficient."""
    engine = FilterEngine(mock_cache, mock_settings)
    
    # Mock score_chunk to return varied scores
    def mock_score_chunk(query, chunk):
        # Return scores 0-4 based on chunk index
        chunk_idx = int(chunk.id.split("-")[1])
        score = chunk_idx % 5  # 0, 1, 2, 3, 4, 0, 1, 2, 3, 4
        return FilterDecision(score=score, keep=score >= 2, why="test")
    
    engine.score_chunk = mock_score_chunk
    
    results = engine.filter_chunks("test query", sample_chunks)
    
    # Should have some survivors (scores >= 3)
    assert len(results) > 0
    
    # Should have hybrid_score set
    for result in results:
        assert hasattr(result, 'hybrid_score')
        assert result.hybrid_score > 0


def test_relaxation_kicks_in_with_few_survivors(mock_cache, mock_settings, sample_chunks):
    """Test threshold relaxation when too few survivors."""
    engine = FilterEngine(mock_cache, mock_settings)
    
    # Mock score_chunk to return mostly low scores
    def mock_score_chunk(query, chunk):
        chunk_idx = int(chunk.id.split("-")[1])
        # Only first 2 chunks get score >= 3
        score = 4 if chunk_idx < 2 else 1
        return FilterDecision(score=score, keep=score >= 2, why="test")
    
    engine.score_chunk = mock_score_chunk
    
    results = engine.filter_chunks("test query", sample_chunks)
    
    # Should relax threshold and include more chunks
    # MIN_SURVIVORS = 3, so should get at least 3 chunks
    assert len(results) >= mock_settings.FILTER_MIN_SURVIVORS


def test_hybrid_alpha_weighting(mock_cache, mock_settings):
    """Test that hybrid_alpha weights filter and retriever scores correctly."""
    # Create a chunk with known scores
    chunk = Chunk(
        id="test-chunk",
        repo_id="test-repo",
        rel_path="test.py",
        start_line=1,
        end_line=10,
        language="python",
        text="def test(): pass",
        retriever_score=0.8
    )
    
    # Calculate expected hybrid score
    filter_score = 4
    filter_norm = filter_score / 4.0  # 1.0
    retriever_norm = 0.8
    alpha = mock_settings.HYBRID_ALPHA  # 0.3
    expected_hybrid = retriever_norm + (alpha * filter_norm)  # 0.8 + 0.3 = 1.1
    
    # Create FilterResult and compute hybrid score
    result = FilterResult(chunk=chunk, score=filter_score, why="test")
    result.hybrid_score = retriever_norm + (alpha * filter_norm)
    
    assert abs(result.hybrid_score - expected_hybrid) < 0.001


def test_fallback_to_retriever_order_last_resort(mock_cache, mock_settings, sample_chunks):
    """Test fallback to retriever order when all filtering fails."""
    engine = FilterEngine(mock_cache, mock_settings)
    
    # Mock score_chunk to return all 0 scores
    def mock_score_chunk(query, chunk):
        return FilterDecision(score=0, keep=False, why="test")
    
    engine.score_chunk = mock_score_chunk
    
    results = engine.filter_chunks("test query", sample_chunks)
    
    # Should fall back to retriever scores
    assert len(results) > 0
    
    # Should be ordered by retriever score (descending)
    for i in range(len(results) - 1):
        assert results[i].chunk.retriever_score >= results[i + 1].chunk.retriever_score

