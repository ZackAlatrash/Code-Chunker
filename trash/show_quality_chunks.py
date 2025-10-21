#!/usr/bin/env python3
"""
Show Quality Chunks

This script shows specific examples of chunks from the quality check output.
"""

import json
import sys
from pathlib import Path

def load_jsonl(file_path: str):
    """Load JSONL file and return list of chunks."""
    chunks = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

def print_chunk_example(chunk, title):
    """Print a formatted chunk example."""
    print(f"\n{'='*80}")
    print(f"📄 {title}")
    print(f"{'='*80}")
    print(f"AST Path: {chunk['ast_path']}")
    print(f"File: {chunk['path']}")
    print(f"Lines: {chunk['start_line']}-{chunk['end_line']}")
    print(f"Language: {chunk['language']}")
    print(f"Tokens: {chunk['token_counts']['total']}")
    print(f"Summary: {chunk['summary_1l']}")
    print(f"QA Terms: {chunk['qa_terms']}")
    print(f"Symbols Defined: {chunk['symbols_defined']}")
    print(f"Symbols Referenced: {chunk['symbols_referenced'][:10]}...")  # First 10
    print(f"Imports Used: {chunk['imports_used']}")
    print(f"Neighbors: prev={chunk['neighbors']['prev'][:16] if chunk['neighbors']['prev'] else 'null'}..., next={chunk['neighbors']['next'][:16] if chunk['neighbors']['next'] else 'null'}...")
    print(f"\n📝 Content:")
    print("-" * 40)
    print(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
    print("-" * 40)

def main():
    """Show chunk examples."""
    chunks_file = "weather_forecast_quality_check.jsonl"
    
    if not Path(chunks_file).exists():
        print(f"❌ File not found: {chunks_file}")
        sys.exit(1)
    
    print("📖 Loading chunks from quality check file...")
    chunks = load_jsonl(chunks_file)
    print(f"📊 Loaded {len(chunks)} chunks")
    
    # Find examples of different chunk types
    examples = {}
    
    for chunk in chunks:
        ast_path = chunk['ast_path']
        language = chunk['language']
        
        # File header
        if ast_path == 'go:file_header' and 'file_header' not in examples:
            examples['file_header'] = chunk
        
        # Interface
        elif 'interface' in ast_path and 'interface' not in examples:
            examples['interface'] = chunk
        
        # Struct
        elif 'struct' in ast_path and 'struct' not in examples:
            examples['struct'] = chunk
        
        # Function
        elif ast_path.startswith('go:function:') and 'function' not in examples:
            examples['function'] = chunk
        
        # Method
        elif ast_path.startswith('go:method:') and 'method' not in examples:
            examples['method'] = chunk
        
        # Method part
        elif '#part' in ast_path and 'method_part' not in examples:
            examples['method_part'] = chunk
        
        # YAML chunk
        elif language == 'yaml' and 'yaml' not in examples:
            examples['yaml'] = chunk
        
        # Stop when we have all examples
        if len(examples) >= 7:
            break
    
    # Print examples
    if 'file_header' in examples:
        print_chunk_example(examples['file_header'], "FILE HEADER CHUNK")
    
    if 'interface' in examples:
        print_chunk_example(examples['interface'], "INTERFACE CHUNK")
    
    if 'struct' in examples:
        print_chunk_example(examples['struct'], "STRUCT CHUNK")
    
    if 'function' in examples:
        print_chunk_example(examples['function'], "FUNCTION CHUNK")
    
    if 'method' in examples:
        print_chunk_example(examples['method'], "METHOD CHUNK")
    
    if 'method_part' in examples:
        print_chunk_example(examples['method_part'], "METHOD PART CHUNK (Split)")
    
    if 'yaml' in examples:
        print_chunk_example(examples['yaml'], "YAML CHUNK")
    
    print(f"\n{'='*80}")
    print("📊 QUALITY CHECK SUMMARY")
    print(f"{'='*80}")
    print(f"Total chunks: {len(chunks)}")
    
    # Count by type
    type_counts = {}
    for chunk in chunks:
        ast_path = chunk['ast_path']
        language = chunk['language']
        
        if ast_path == 'go:file_header':
            type_counts['File Headers'] = type_counts.get('File Headers', 0) + 1
        elif 'interface' in ast_path:
            type_counts['Interfaces'] = type_counts.get('Interfaces', 0) + 1
        elif 'struct' in ast_path:
            type_counts['Structs'] = type_counts.get('Structs', 0) + 1
        elif ast_path.startswith('go:function:'):
            type_counts['Functions'] = type_counts.get('Functions', 0) + 1
        elif ast_path.startswith('go:method:'):
            type_counts['Methods'] = type_counts.get('Methods', 0) + 1
        elif '#part' in ast_path:
            type_counts['Method Parts'] = type_counts.get('Method Parts', 0) + 1
        elif language == 'yaml':
            type_counts['YAML'] = type_counts.get('YAML', 0) + 1
        elif language == 'markdown':
            type_counts['Markdown'] = type_counts.get('Markdown', 0) + 1
        elif language == 'json':
            type_counts['JSON'] = type_counts.get('JSON', 0) + 1
        else:
            type_counts['Other'] = type_counts.get('Other', 0) + 1
    
    for chunk_type, count in sorted(type_counts.items()):
        print(f"{chunk_type}: {count}")
    
    # Token distribution
    token_dist = {'under_50': 0, '50_100': 0, '100_200': 0, '200_300': 0, 'over_300': 0}
    total_tokens = 0
    
    for chunk in chunks:
        token_count = chunk['token_counts']['total']
        total_tokens += token_count
        
        if token_count < 50:
            token_dist['under_50'] += 1
        elif token_count < 100:
            token_dist['50_100'] += 1
        elif token_count < 200:
            token_dist['100_200'] += 1
        elif token_count < 300:
            token_dist['200_300'] += 1
        else:
            token_dist['over_300'] += 1
    
    print(f"\n📏 TOKEN DISTRIBUTION:")
    print(f"Under 50 tokens: {token_dist['under_50']} ({token_dist['under_50']/len(chunks)*100:.1f}%)")
    print(f"50-100 tokens: {token_dist['50_100']} ({token_dist['50_100']/len(chunks)*100:.1f}%)")
    print(f"100-200 tokens: {token_dist['100_200']} ({token_dist['100_200']/len(chunks)*100:.1f}%)")
    print(f"200-300 tokens: {token_dist['200_300']} ({token_dist['200_300']/len(chunks)*100:.1f}%)")
    print(f"Over 300 tokens: {token_dist['over_300']} ({token_dist['over_300']/len(chunks)*100:.1f}%)")
    print(f"Average tokens: {total_tokens/len(chunks):.1f}")

if __name__ == "__main__":
    main()
