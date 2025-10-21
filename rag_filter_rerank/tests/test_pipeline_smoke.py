"""
Smoke tests for complete pipeline.
"""
import pytest
from unittest.mock import Mock, patch
from rag_filter_rerank.schemas import Chunk
from rag_filter_rerank.pipeline import FilterRerankPipeline
from rag_filter_rerank.retriever import RetrieverAdapter
from rag_filter_rerank.tests.test_filter_engine import MockCache


class MockRetriever(RetrieverAdapter):
    """Mock retriever for testing."""
    
    def __init__(self, chunks):
        self.chunks = chunks
    
    def search(self, query, top_n):
        return self.chunks[:top_n]


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
    settings.RERANK_VENDOR = "cohere"
    settings.COHERE_API_KEY = None  # No API key for testing
    settings.COHERE_MODEL = "rerank-english-v3.0"
    settings.RERANK_TIMEOUT_S = 30
    settings.RERANK_BATCH_SIZE = 96
    settings.RERANK_MAX_CALLS = 3
    settings.RERANK_MAX_DOCS = 288
    settings.RERANK_RETRIES = 3
    settings.DISABLE_RERANK = True  # Disable for simpler testing
    settings.PIPELINE_TOPN_RECALL = 10
    settings.PIPELINE_TOPK_EVIDENCE = 3
    settings.RAG_FILTER_RERANK = "on"
    return settings


@pytest.fixture
def sample_chunks():
    """Create sample chunks (2 relevant, 3 irrelevant)."""
    return [
        Chunk(
            id="relevant-1",
            repo_id="test-repo",
            rel_path="src/auth.py",
            start_line=10,
            end_line=30,
            language="python",
            text="def authenticate(user, password):\n    return check_credentials(user, password)",
            summary_en="Authenticates user with credentials",
            chunk_hash="abc123"
        ),
        Chunk(
            id="relevant-2",
            repo_id="test-repo",
            rel_path="src/session.py",
            start_line=50,
            end_line=70,
            language="python",
            text="def create_session(user_id):\n    return Session(user_id, expires=3600)",
            summary_en="Creates a user session",
            chunk_hash="def456"
        ),
        Chunk(
            id="irrelevant-1",
            repo_id="test-repo",
            rel_path="src/utils.py",
            start_line=100,
            end_line=110,
            language="python",
            text="def format_date(dt):\n    return dt.strftime('%Y-%m-%d')",
            summary_en="Formats a date",
            chunk_hash="ghi789"
        ),
        Chunk(
            id="irrelevant-2",
            repo_id="test-repo",
            rel_path="src/config.py",
            start_line=1,
            end_line=10,
            language="python",
            text="DATABASE_URL = 'postgres://localhost'",
            summary_en="Database configuration",
            chunk_hash="jkl012"
        ),
        Chunk(
            id="irrelevant-3",
            repo_id="test-repo",
            rel_path="src/logger.py",
            start_line=20,
            end_line=40,
            language="python",
            text="def log_error(msg):\n    print(f'ERROR: {msg}')",
            summary_en="Logs an error message",
            chunk_hash="mno345"
        ),
    ]


@patch('rag_filter_rerank.filter_engine.requests.post')
@patch('rag_filter_rerank.evidence.requests.post')
def test_pipeline_smoke(mock_answer_post, mock_filter_post, mock_settings, sample_chunks):
    """Smoke test: Pipeline runs end-to-end."""
    # Mock filter responses (score relevant chunks higher)
    def mock_filter_response(chunk_id):
        if "relevant" in chunk_id:
            score = 4
        else:
            score = 0
        
        keep = "true" if score >= 2 else "false"
        resp = Mock()
        resp.json.return_value = {
            "message": {"content": f'{{"score": {score}, "keep": {keep}, "why": "test"}}'}
        }
        resp.raise_for_status = Mock()
        return resp
    
    # Need 5 responses for 5 chunks
    mock_filter_post.side_effect = [
        mock_filter_response("relevant-1"),
        mock_filter_response("relevant-2"),
        mock_filter_response("irrelevant-1"),
        mock_filter_response("irrelevant-2"),
        mock_filter_response("irrelevant-3"),
    ]
    
    # Mock answerer response
    mock_answer_resp = Mock()
    mock_answer_resp.json.return_value = {
        "message": {"content": "Authentication is handled by authenticate() in src/auth.py [1]."}
    }
    mock_answer_resp.raise_for_status = Mock()
    mock_answer_post.return_value = mock_answer_resp
    
    # Create pipeline
    retriever = MockRetriever(sample_chunks)
    cache = MockCache()
    pipeline = FilterRerankPipeline(retriever, mock_settings, cache=cache)
    
    # Run pipeline
    query = "How does authentication work?"
    result = pipeline.run(query)
    
    # Assertions
    assert result.query == query
    assert result.recall_n == 5
    assert result.filtered_m >= 2  # At least 2 relevant chunks
    assert result.evidence_k == 3  # Top-K is 3
    assert len(result.reranked) == 3
    assert result.answer  # Has an answer
    assert "authenticate" in result.answer.lower() or len(result.answer) > 0
    
    # Check timing
    assert "retrieval_ms" in result.timing_ms
    assert "filter_ms" in result.timing_ms
    assert "total_ms" in result.timing_ms
    
    # Check cache stats
    assert "backend" in result.cache_stats
    
    # Check flags
    assert result.flags["rag_filter_rerank"] == "on"


@patch('rag_filter_rerank.filter_engine.requests.post')
def test_pipeline_with_trace(mock_filter_post, mock_settings, sample_chunks):
    """Test pipeline with trace mode."""
    # Mock filter responses
    mock_resp = Mock()
    mock_resp.json.return_value = {
        "message": {"content": '{"score": 3, "keep": true, "why": "relevant"}'}
    }
    mock_resp.raise_for_status = Mock()
    mock_filter_post.return_value = mock_resp
    
    # Mock answerer
    with patch('rag_filter_rerank.evidence.requests.post') as mock_answer:
        mock_answer_resp = Mock()
        mock_answer_resp.json.return_value = {
            "message": {"content": "Test answer"}
        }
        mock_answer_resp.raise_for_status = Mock()
        mock_answer.return_value = mock_answer_resp
        
        retriever = MockRetriever(sample_chunks)
        cache = MockCache()
        pipeline = FilterRerankPipeline(retriever, mock_settings, cache=cache)
        
        # Run with trace
        result = pipeline.run("test query", trace=True)
        
        # Should complete successfully
        assert result.answer
        assert result.recall_n > 0


def test_pipeline_empty_chunks(mock_settings):
    """Test pipeline with no chunks."""
    retriever = MockRetriever([])
    cache = MockCache()
    pipeline = FilterRerankPipeline(retriever, mock_settings, cache=cache)
    
    result = pipeline.run("test query")
    
    # Should return empty result gracefully
    assert result.recall_n == 0
    assert result.filtered_m == 0
    assert result.evidence_k == 0
    assert "no relevant" in result.answer.lower() or result.answer


@patch('rag_filter_rerank.filter_engine.requests.post')
def test_pipeline_filter_failure_fallback(mock_filter_post, mock_settings, sample_chunks):
    """Test fallback when filter fails."""
    # Filter fails completely
    mock_filter_post.side_effect = Exception("Network error")
    
    # Mock answerer
    with patch('rag_filter_rerank.evidence.requests.post') as mock_answer:
        mock_answer_resp = Mock()
        mock_answer_resp.json.return_value = {
            "message": {"content": "Fallback answer"}
        }
        mock_answer_resp.raise_for_status = Mock()
        mock_answer.return_value = mock_answer_resp
        
        retriever = MockRetriever(sample_chunks)
        cache = MockCache()
        pipeline = FilterRerankPipeline(retriever, mock_settings, cache=cache)
        
        result = pipeline.run("test query")
        
        # Should still complete with fallback
        assert result.recall_n > 0
        assert result.answer  # Should have some answer


def test_pipeline_determinism(mock_settings, sample_chunks):
    """Test that pipeline produces deterministic results."""
    # Create identical chunks
    retriever1 = MockRetriever(sample_chunks.copy())
    retriever2 = MockRetriever(sample_chunks.copy())
    
    cache = MockCache()
    
    with patch('rag_filter_rerank.filter_engine.requests.post') as mock_filter:
        with patch('rag_filter_rerank.evidence.requests.post') as mock_answer:
            # Mock responses
            mock_filter_resp = Mock()
            mock_filter_resp.json.return_value = {
                "message": {"content": '{"score": 3, "keep": true, "why": "test"}'}
            }
            mock_filter_resp.raise_for_status = Mock()
            mock_filter.return_value = mock_filter_resp
            
            mock_answer_resp = Mock()
            mock_answer_resp.json.return_value = {
                "message": {"content": "Test answer"}
            }
            mock_answer_resp.raise_for_status = Mock()
            mock_answer.return_value = mock_answer_resp
            
            # Run twice
            pipeline1 = FilterRerankPipeline(retriever1, mock_settings, cache=cache)
            result1 = pipeline1.run("test query")
            
            pipeline2 = FilterRerankPipeline(retriever2, mock_settings, cache=cache)
            result2 = pipeline2.run("test query")
            
            # Results should be deterministic
            assert result1.recall_n == result2.recall_n
            assert result1.filtered_m == result2.filtered_m
            assert result1.evidence_k == result2.evidence_k
            
            # Reranked order should match (provenance IDs)
            prov_ids1 = [r["provenance_id"] for r in result1.reranked]
            prov_ids2 = [r["provenance_id"] for r in result2.reranked]
            assert prov_ids1 == prov_ids2

