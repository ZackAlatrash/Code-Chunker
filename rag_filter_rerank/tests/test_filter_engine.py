"""
Tests for filter engine.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_filter_rerank.schemas import Chunk, FilterDecision, FilterResult
from rag_filter_rerank.filter_engine import FilterEngine
from rag_filter_rerank.cache import Cache


class MockCache(Cache):
    """Mock cache for testing."""
    
    def __init__(self):
        self.store = {}
    
    def get(self, key):
        return self.store.get(key)
    
    def set(self, key, value, ttl_s=None):
        self.store[key] = value
    
    def clear(self):
        self.store = {}
    
    def stats(self):
        return {"backend": "mock", "hits": 0, "misses": 0, "hit_rate": "0%", "size_mb": 0}


@pytest.fixture
def mock_settings():
    """Mock settings."""
    settings = Mock()
    settings.OLLAMA_URL = "http://localhost:11434/api/chat"
    settings.OLLAMA_MODEL = "qwen2.5-coder:7b-instruct"
    settings.FILTER_MAX_CODE_CHARS = 1200
    settings.FILTER_PARALLELISM = 2
    settings.FILTER_THRESHOLD = 2
    settings.FILTER_MIN_SURVIVORS = 3
    settings.FILTER_TIMEOUT_S = 10
    settings.CACHE_TTL_S = 3600
    settings.PROMPT_VERSION = "v1"
    return settings


@pytest.fixture
def filter_engine(mock_settings):
    """Create filter engine with mock cache."""
    cache = MockCache()
    return FilterEngine(cache, mock_settings)


@pytest.fixture
def sample_chunk():
    """Sample chunk for testing."""
    return Chunk(
        id="chunk-1",
        repo_id="test-repo",
        rel_path="src/main.py",
        start_line=10,
        end_line=20,
        language="python",
        text="def hello():\n    return 'world'",
        summary_en="Returns a greeting",
        chunk_hash="abc123"
    )


def test_build_system_prompt(filter_engine):
    """Test system prompt building."""
    prompt = filter_engine.system_prompt
    
    assert "code-judge" in prompt.lower()
    assert "JSON" in prompt
    assert "score" in prompt.lower()


def test_build_user_prompt(filter_engine, sample_chunk):
    """Test user prompt building."""
    query = "How do I get a greeting?"
    prompt = filter_engine._build_user_prompt(query, sample_chunk)
    
    assert query in prompt
    assert "src/main.py" in prompt
    assert "10-20" in prompt
    assert "python" in prompt
    assert "Returns a greeting" in prompt
    assert "def hello()" in prompt


@patch('rag_filter_rerank.filter_engine.requests.post')
def test_score_chunk_success(mock_post, filter_engine, sample_chunk):
    """Test successful chunk scoring."""
    # Mock Ollama response
    mock_response = Mock()
    mock_response.json.return_value = {
        "message": {
            "content": '{"score": 3, "keep": true, "why": "Directly answers the question"}'
        }
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    query = "How do I get a greeting?"
    decision = filter_engine.score_chunk(query, sample_chunk)
    
    assert isinstance(decision, FilterDecision)
    assert decision.score == 3
    assert decision.keep is True
    assert "answers" in decision.why.lower()


@patch('rag_filter_rerank.filter_engine.requests.post')
def test_score_chunk_caching(mock_post, filter_engine, sample_chunk):
    """Test that scoring uses cache."""
    # First call
    mock_response = Mock()
    mock_response.json.return_value = {
        "message": {
            "content": '{"score": 3, "keep": true, "why": "test"}'
        }
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    query = "test query"
    decision1 = filter_engine.score_chunk(query, sample_chunk)
    
    # Second call should use cache
    decision2 = filter_engine.score_chunk(query, sample_chunk)
    
    # Should only call API once
    assert mock_post.call_count == 1
    assert decision1.score == decision2.score


@patch('rag_filter_rerank.filter_engine.requests.post')
def test_score_chunk_parse_error_retry(mock_post, filter_engine, sample_chunk):
    """Test retry on JSON parse error."""
    # First call returns invalid JSON
    mock_response1 = Mock()
    mock_response1.json.return_value = {
        "message": {
            "content": 'invalid json {'
        }
    }
    mock_response1.raise_for_status = Mock()
    
    # Second call (retry) returns valid JSON
    mock_response2 = Mock()
    mock_response2.json.return_value = {
        "message": {
            "content": '{"score": 2, "keep": true, "why": "retry worked"}'
        }
    }
    mock_response2.raise_for_status = Mock()
    
    mock_post.side_effect = [mock_response1, mock_response2]
    
    query = "test query"
    decision = filter_engine.score_chunk(query, sample_chunk)
    
    # Should have retried
    assert mock_post.call_count == 2
    assert decision.score == 2


@patch('rag_filter_rerank.filter_engine.requests.post')
def test_score_chunk_complete_failure(mock_post, filter_engine, sample_chunk):
    """Test fallback on complete failure."""
    mock_post.side_effect = Exception("Network error")
    
    query = "test query"
    decision = filter_engine.score_chunk(query, sample_chunk)
    
    # Should return safe fallback
    assert decision.score == 0
    assert decision.keep is False
    assert "error" in decision.why.lower()


def test_contains_secrets(filter_engine):
    """Test secret detection."""
    # Should detect AWS keys
    text_with_secret = "AKIAIOSFODNN7EXAMPLE"
    assert filter_engine._contains_secrets(text_with_secret)
    
    # Should detect passwords
    text_with_password = 'password="secret123"'
    assert filter_engine._contains_secrets(text_with_password)
    
    # Normal code should pass
    normal_code = "def hello(): return 'world'"
    assert not filter_engine._contains_secrets(normal_code)


@patch('rag_filter_rerank.filter_engine.requests.post')
def test_filter_chunks_threshold(mock_post, filter_engine):
    """Test filtering with threshold."""
    # Create chunks with different scores
    chunks = [
        Chunk(id=f"chunk-{i}", repo_id="test", rel_path=f"file{i}.py",
              start_line=1, end_line=10, language="python", text="code")
        for i in range(5)
    ]
    
    # Mock responses with varying scores
    def mock_response_factory(score):
        resp = Mock()
        keep = "true" if score >= 2 else "false"
        resp.json.return_value = {
            "message": {"content": f'{{"score": {score}, "keep": {keep}, "why": "test"}}'}
        }
        resp.raise_for_status = Mock()
        return resp
    
    mock_post.side_effect = [
        mock_response_factory(4),  # Keep
        mock_response_factory(3),  # Keep
        mock_response_factory(2),  # Keep
        mock_response_factory(1),  # Drop
        mock_response_factory(0),  # Drop
    ]
    
    query = "test"
    results = filter_engine.filter_chunks(query, chunks)
    
    # Should keep 3 chunks (scores >= 2)
    assert len(results) == 3
    assert all(r.score >= 2 for r in results)
    
    # Should be sorted by score descending
    assert results[0].score >= results[1].score >= results[2].score


@patch('rag_filter_rerank.filter_engine.requests.post')
def test_filter_chunks_threshold_relaxation(mock_post, filter_engine):
    """Test threshold relaxation when too few survivors."""
    # Create chunks
    chunks = [
        Chunk(id=f"chunk-{i}", repo_id="test", rel_path=f"file{i}.py",
              start_line=1, end_line=10, language="python", text="code")
        for i in range(5)
    ]
    
    # All chunks score 1 (below threshold of 2)
    def mock_response_factory():
        resp = Mock()
        resp.json.return_value = {
            "message": {"content": '{"score": 1, "keep": false, "why": "marginal"}'}
        }
        resp.raise_for_status = Mock()
        return resp
    
    mock_post.side_effect = [mock_response_factory() for _ in range(5)]
    
    query = "test"
    results = filter_engine.filter_chunks(query, chunks)
    
    # Should relax threshold to >=1 since initial filtering left < min_survivors
    # min_survivors is 3, so should get at least some results
    assert len(results) >= 3 or len(results) == len(chunks)

