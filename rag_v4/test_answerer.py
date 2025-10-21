#!/usr/bin/env python3
"""
Quick test for the RAG v4 answerer.
Extracts a few chunks from a JSONL file and formats them for testing.
"""
import json
import sys
import os

def extract_sample_chunks(jsonl_path: str, max_chunks: int = 5) -> list:
    """Extract first N chunks from a JSONL file"""
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_chunks:
                break
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return chunks

def main():
    # Use the weather service as a test case
    jsonl_path = "ChunksV3/weather_foreca_proxy_service_v3_enriched.jsonl"
    
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found")
        sys.exit(1)
    
    print("Extracting sample chunks...")
    chunks = extract_sample_chunks(jsonl_path, max_chunks=5)
    
    # Save to temp JSON for testing
    test_file = "rag_v4/test_chunks.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"✅ Extracted {len(chunks)} chunks to {test_file}")
    print("\nSample chunk structure:")
    if chunks:
        c = chunks[0]
        print(f"  - repo_id: {c.get('repo_id', 'N/A')}")
        print(f"  - rel_path: {c.get('rel_path', c.get('path', 'N/A'))}")
        print(f"  - primary_symbol: {c.get('primary_symbol', 'N/A')}")
        print(f"  - primary_kind: {c.get('primary_kind', 'N/A')}")
        print(f"  - all_roles: {c.get('all_roles', [])}")
        print(f"  - has summary_en: {'summary_en' in c}")
        print(f"  - has text: {'text' in c}")
    
    print("\n" + "="*80)
    print("TEST COMMAND:")
    print("="*80)
    print(f"\npython -m rag_v4.answerer \"What does the GetForecastForLocation method do?\" --chunks-json {test_file} --k 5\n")
    print("="*80)

if __name__ == "__main__":
    main()

