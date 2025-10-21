#!/usr/bin/env python3
"""
Integration Example: Using Qodo Embeddings with Existing Search System

This example shows how to integrate Qodo embeddings into the existing
search_v4 system for A/B testing and production use.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_qodo_embed import QodoEmbedder, QodoOpenSearchClient, get_settings
from search_v4.service import search_v4


def example_index_chunks():
    """Example: Index chunks using Qodo embeddings."""
    print("🔧 Example: Indexing chunks with Qodo embeddings")
    print("=" * 60)
    
    # Initialize Qodo components
    settings = get_settings()
    embedder = QodoEmbedder(settings)
    client = QodoOpenSearchClient(settings)
    
    # Create index
    print("Creating OpenSearch index...")
    if client.create_index(force=True):
        print("✅ Index created successfully")
    else:
        print("❌ Failed to create index")
        return
    
    # Example chunks (in practice, load from JSON file)
    example_chunks = [
        {
            "id": "example1",
            "repo_id": "test-repo",
            "rel_path": "src/main.go",
            "text": "func main() {\n    fmt.Println(\"Hello, World!\")\n}",
            "language": "go",
            "start_line": 1,
            "end_line": 3,
            "summary_en": "Main function that prints hello world"
        },
        {
            "id": "example2", 
            "repo_id": "test-repo",
            "rel_path": "src/auth.go",
            "text": "func authenticateUser(username, password string) bool {\n    // Authentication logic\n    return true\n}",
            "language": "go",
            "start_line": 1,
            "end_line": 4,
            "summary_en": "User authentication function"
        }
    ]
    
    # Prepare documents
    docs = []
    texts = []
    for chunk in example_chunks:
        doc = {
            **chunk,
            "model_name": settings.MODEL_ID,
            "model_version": "1.0.0",
            "embedding_type": "code",
            "chunk_hash": "example_hash",
            "created_at": "2024-01-01T00:00:00Z"
        }
        docs.append(doc)
        texts.append(chunk["text"])
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embedder.embed_docs(texts)
    print(f"✅ Generated {embeddings.shape[0]} embeddings")
    
    # Add embeddings to documents
    for doc, embedding in zip(docs, embeddings):
        doc["embedding"] = embedding.tolist()
    
    # Index documents
    print("Indexing documents...")
    if client.bulk_index(docs):
        print("✅ Documents indexed successfully")
    else:
        print("❌ Failed to index documents")


def example_search_chunks():
    """Example: Search chunks using Qodo embeddings."""
    print("\n🔍 Example: Searching chunks with Qodo embeddings")
    print("=" * 60)
    
    # Initialize components
    settings = get_settings()
    embedder = QodoEmbedder(settings)
    client = QodoOpenSearchClient(settings)
    
    # Example queries
    queries = [
        "How does authentication work?",
        "main function implementation",
        "error handling patterns"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Generate query embedding
        query_vec = embedder.embed_query(query)
        
        # Search
        results = client.knn_search(query_vec.tolist(), k=3)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.get('rel_path', 'unknown')} (score: {result.get('_score', 0.0):.3f})")
                if result.get('summary_en'):
                    print(f"     {result['summary_en']}")
        else:
            print("  No results found")


def example_integration_with_search_v4():
    """Example: Using Qodo index with existing search_v4 system."""
    print("\n🔗 Example: Integration with search_v4 system")
    print("=" * 60)
    
    # Set environment variable to use Qodo index
    os.environ['RETRIEVER_INDEX'] = 'code_chunks_v5_qodo'
    
    # Example search using existing search_v4
    query = "authentication and user management"
    repo_ids = ["test-repo"]
    plan = {"repos": repo_ids}
    
    print(f"Searching with search_v4: '{query}'")
    print("Using Qodo index via RETRIEVER_INDEX environment variable")
    
    try:
        results = search_v4(query, repo_ids, plan, fetch_all_texts=True)
        print(f"✅ search_v4 returned {len(results.get('results', []))} results")
        
        # Show first result
        if results.get('results'):
            first_result = results['results'][0]
            print(f"Top result: {first_result.get('rel_path', 'unknown')}")
            print(f"Score: {first_result.get('_score', 0.0):.3f}")
            
    except Exception as e:
        print(f"❌ search_v4 failed: {e}")
        print("Note: This requires the Qodo index to be populated first")


def example_ab_testing():
    """Example: A/B testing between MiniLM and Qodo embeddings."""
    print("\n📊 Example: A/B Testing Setup")
    print("=" * 60)
    
    print("To run A/B testing between MiniLM (v4) and Qodo (v5) embeddings:")
    print()
    print("1. Create queries file:")
    print("   echo 'How does authentication work?' > queries.txt")
    print("   echo 'database connection pooling' >> queries.txt")
    print()
    print("2. Run evaluation:")
    print("   python -m rag_qodo_embed.evaluate \\")
    print("     --queries queries.txt \\")
    print("     --index-a code_chunks_v4 \\")
    print("     --index-b code_chunks_v5_qodo \\")
    print("     --k 10 \\")
    print("     --output results.json")
    print()
    print("3. Switch to Qodo in production:")
    print("   export RETRIEVER_INDEX=code_chunks_v5_qodo")
    print("   # Your existing search_v4 code will now use Qodo embeddings")


def main():
    """Run all examples."""
    print("🚀 Qodo Embedding Integration Examples")
    print("=" * 60)
    
    try:
        # Example 1: Index chunks
        example_index_chunks()
        
        # Example 2: Search chunks
        example_search_chunks()
        
        # Example 3: Integration with search_v4
        example_integration_with_search_v4()
        
        # Example 4: A/B testing setup
        example_ab_testing()
        
        print("\n✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Index your actual chunk data using the indexer CLI")
        print("2. Run A/B testing to compare against existing embeddings")
        print("3. Switch to Qodo by setting RETRIEVER_INDEX environment variable")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Make sure OpenSearch is running and dependencies are installed")


if __name__ == '__main__':
    main()
