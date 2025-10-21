#!/usr/bin/env python3
"""
Example: Integrating RAG v4 Answerer with search_v4

This shows how to:
1. Perform a search using search_v4 (if available)
2. Save results for reuse
3. Answer questions using the retrieved chunks
"""
import json
import os
import sys

def example_with_search_v4():
    """Example using live search_v4 integration"""
    print("="*80)
    print("Example 1: Using Live Search (search_v4)")
    print("="*80)
    
    try:
        from search_v4.service import search_v4
        
        # Perform a search
        query = "How does the weather forecast service handle caching?"
        results = search_v4(
            query=query,
            router_repo_ids=["weather_foreca_proxy_service"],
            plan={"clarified_query": query}
        )
        
        chunks = results.get("results", [])
        print(f"✅ Found {len(chunks)} chunks from search_v4")
        
        # Save for reuse
        output_file = "rag_v4/search_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
        
        print(f"✅ Saved to {output_file}")
        print("\nNow answer with:")
        print(f"  python -m rag_v4.answerer \"{query}\" --chunks-json {output_file}")
        
    except ImportError:
        print("⚠️  search_v4 not available - skipping this example")
    except Exception as e:
        print(f"❌ Error: {e}")

def example_with_json_file():
    """Example using pre-saved chunks"""
    print("\n" + "="*80)
    print("Example 2: Using Pre-saved Chunks")
    print("="*80)
    
    test_file = "rag_v4/test_chunks.json"
    
    if not os.path.exists(test_file):
        print(f"⚠️  {test_file} not found - run: python rag_v4/test_answerer.py")
        return
    
    # Load chunks
    with open(test_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"✅ Loaded {len(chunks)} chunks from {test_file}")
    
    # Show what we have
    print("\nChunks preview:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"  [{i}] {chunk.get('primary_symbol', 'N/A')} ({chunk.get('primary_kind', 'N/A')})")
    
    # Example questions
    questions = [
        "What does the main function do?",
        "How is the root command initialized?",
        "What gRPC interceptors are configured?"
    ]
    
    print("\nExample questions you can ask:")
    for q in questions:
        print(f"  • {q}")
    
    print("\nRun:")
    print(f"  python -m rag_v4.answerer \"QUESTION\" --chunks-json {test_file}")

def example_batch_questions():
    """Example: Answer multiple related questions"""
    print("\n" + "="*80)
    print("Example 3: Batch Questions (Reuse Chunks)")
    print("="*80)
    
    test_file = "rag_v4/test_chunks.json"
    
    if not os.path.exists(test_file):
        print(f"⚠️  {test_file} not found")
        return
    
    questions = [
        "What is the main entry point?",
        "What command-line flags are available?",
        "How is logging configured?"
    ]
    
    print("To answer multiple questions efficiently:")
    print(f"\n# Save chunks once")
    print(f"python rag_v4/test_answerer.py")
    print(f"\n# Reuse for multiple questions")
    for q in questions:
        print(f"python -m rag_v4.answerer \"{q}\" --chunks-json {test_file}")

def example_custom_extraction():
    """Example: Extract specific chunks by pattern"""
    print("\n" + "="*80)
    print("Example 4: Custom Chunk Extraction")
    print("="*80)
    
    jsonl_path = "ChunksV3/weather_foreca_proxy_service_v3_enriched.jsonl"
    
    if not os.path.exists(jsonl_path):
        print(f"⚠️  {jsonl_path} not found")
        return
    
    print(f"Extract chunks matching a pattern from {jsonl_path}:")
    print("\n# Python code to extract 'forecast' related chunks:")
    print("""
import json

chunks = []
with open('ChunksV3/weather_foreca_proxy_service_v3_enriched.jsonl', 'r') as f:
    for line in f:
        chunk = json.loads(line)
        if 'forecast' in chunk.get('primary_symbol', '').lower():
            chunks.append(chunk)
            if len(chunks) >= 10:
                break

with open('forecast_chunks.json', 'w') as f:
    json.dump(chunks, f, indent=2)
""")
    print("\nThen:")
    print("  python -m rag_v4.answerer \"How does forecast caching work?\" --chunks-json forecast_chunks.json")

def main():
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "RAG V4 ANSWERER - INTEGRATION EXAMPLES" + " "*20 + "║")
    print("╚" + "="*78 + "╝\n")
    
    example_with_json_file()
    example_batch_questions()
    example_custom_extraction()
    example_with_search_v4()
    
    print("\n" + "="*80)
    print("📚 For more examples, see:")
    print("  - rag_v4/README.md")
    print("  - rag_v4/USAGE_EXAMPLES.md")
    print("  - rag_v4/QUICK_REFERENCE.md")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

