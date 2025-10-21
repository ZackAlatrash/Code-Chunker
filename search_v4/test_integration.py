"""
Integration test for search_v4.

Tests the complete search flow with a mock embedding model.
"""
import json
from typing import List

# Mock embedding model for testing
def mock_get_embedding(text: str) -> List[float]:
    """Return a dummy 384-dim vector (all 0.1s)."""
    return [0.1] * 384


def test_search_flow():
    """Test the complete search flow with mocked components."""
    # Temporarily patch the embedding function
    import search_v4.embeddings as emb_module
    original_fn = emb_module.get_embedding
    emb_module.get_embedding = mock_get_embedding
    
    try:
        from search_v4.service import search_v4
        
        # Mock query plan
        plan = {
            "clarified_query": "weather forecast location",
            "identifiers": ["GetForecastForLocation", "Service"],
            "file_hints": ["service.go", "forecast"],
            "language": "go",
            "bm25_should": [
                {"field": "primary_symbol", "term": "GetForecastForLocation", "boost": 3.0},
                {"field": "summary_en", "term": "weather forecast", "boost": 2.0},
                {"field": "text", "term": "location", "boost": 1.0}
            ],
            "hyde_passage": "This function gets weather forecast data for a specific location."
        }
        
        # Execute search
        print("Testing search_v4 with mock data...")
        print(f"Query: 'where is GetForecastForLocation?'")
        print(f"Repos: ['foreca']")
        print(f"Plan identifiers: {plan['identifiers']}")
        print()
        
        results = search_v4(
            query="where is GetForecastForLocation?",
            router_repo_ids=["foreca"],
            plan=plan
        )
        
        # Validate structure
        assert "query" in results
        assert "router_repo_ids" in results
        assert "plan" in results
        assert "results" in results
        assert isinstance(results["results"], list)
        
        print(f"✅ Search executed successfully")
        print(f"   Found {len(results['results'])} results")
        
        if results["results"]:
            first = results["results"][0]
            print(f"\nFirst result:")
            print(f"   ID: {first.get('id')}")
            print(f"   Repo: {first.get('repo_id')}")
            print(f"   Path: {first.get('rel_path')}")
            print(f"   Symbol: {first.get('primary_symbol')}")
            print(f"   Roles: {first.get('all_roles')}")
            print(f"   Lines: {first.get('start_line')}-{first.get('end_line')}")
            print(f"   Has text: {'text' in first}")
        
        print("\n✅ All assertions passed!")
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Restore original function
        emb_module.get_embedding = original_fn


if __name__ == "__main__":
    print("=" * 80)
    print("SEARCH_V4 INTEGRATION TEST")
    print("=" * 80)
    print()
    
    try:
        results = test_search_flow()
        
        print("\n" + "=" * 80)
        print("TEST PASSED ✅")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Wire in real embedding model in search_v4/embeddings.py")
        print("2. Set OpenSearch environment variables")
        print("3. Run: python -m search_v4.cli --query 'your query' --repos foreca")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED ❌")
        print("=" * 80)
        print(f"\nError: {e}")
        exit(1)

