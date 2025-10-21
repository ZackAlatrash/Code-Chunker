#!/usr/bin/env python3
"""
Standalone enrichment tool for adding Dutch LLM summaries to existing code chunks.

This tool reads existing chunks from JSONL and adds summary_nl, keywords_nl,
and enrich_provenance fields without modifying the original chunking logic.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from enrich.llm_enricher import LLMEnricher
from enrich.cache import EnrichmentCache


def load_chunks(input_path: str) -> List[Dict[str, Any]]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    return chunks


def save_chunks(chunks: List[Dict[str, Any]], output_path: str):
    """Save chunks to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


def group_chunks_by_file(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group chunks by file path."""
    file_groups = defaultdict(list)
    
    for chunk in chunks:
        # Use rel_path if available, otherwise use path
        file_path = chunk.get("rel_path") or chunk.get("path", "unknown")
        file_groups[file_path].append(chunk)
    
    return dict(file_groups)


def build_file_text(chunks: List[Dict[str, Any]]) -> str:
    """Build concatenated file text from chunks."""
    # Sort chunks by start_line if available
    sorted_chunks = sorted(chunks, key=lambda c: c.get("start_line", 0))
    
    file_text_parts = []
    for chunk in sorted_chunks:
        text = chunk.get("text", "")
        if text.strip():
            file_text_parts.append(text.strip())
    
    return "\n\n".join(file_text_parts)


def hash_file_content(file_text: str) -> str:
    """Generate hash of file content."""
    return hashlib.sha256(file_text.encode('utf-8')).hexdigest()


def enrich_chunks(
    input_path: str,
    output_path: str,
    model: str,
    max_workers: int,
    batch_size: int,
    max_retries: int,
    dry_run: bool,
    overwrite: bool,
    cache_path: str
) -> Dict[str, int]:
    """Main enrichment logic."""
    print(f"Loading chunks from {input_path}...")
    chunks = load_chunks(input_path)
    print(f"Loaded {len(chunks)} chunks")
    
    if not chunks:
        print("No chunks to process")
        return {"total": 0, "processed": 0, "cached": 0, "generated": 0, "errors": 0}
    
    # Group chunks by file
    file_groups = group_chunks_by_file(chunks)
    print(f"Grouped into {len(file_groups)} files")
    
    # Initialize components
    enricher = LLMEnricher(
        model=model,
        max_workers=max_workers,
        batch_size=batch_size,
        max_retries=max_retries
    )
    
    cache = EnrichmentCache(cache_path)
    
    stats = {
        "total": len(chunks),
        "processed": 0,
        "cached": 0,
        "generated": 0,
        "errors": 0
    }
    
    enriched_chunks = []
    
    for file_path, file_chunks in file_groups.items():
        print(f"Processing file: {file_path} ({len(file_chunks)} chunks)")
        
        # Build file text and synopsis
        file_text = build_file_text(file_chunks)
        file_sha = hash_file_content(file_text)
        
        # Generate file synopsis
        language = file_chunks[0].get("language", "go")  # Default to Go
        try:
            file_synopsis_nl = enricher.generate_file_synopsis(file_path, language, file_text)
            file_synopsis_hash = hashlib.sha256(file_synopsis_nl.encode('utf-8')).hexdigest()
        except Exception as e:
            print(f"Warning: Failed to generate synopsis for {file_path}: {e}")
            file_synopsis_nl = f"Bestand {file_path} kon niet worden samengevat."
            file_synopsis_hash = hashlib.sha256(file_synopsis_nl.encode('utf-8')).hexdigest()
        
        # Process chunks in this file
        for chunk in file_chunks:
            chunk_id = chunk.get("id", f"{file_path}#{chunk.get('chunk_number', 0)}")
            
            # Check if already enriched and not overwriting
            if not overwrite and all(key in chunk for key in ["summary_nl", "keywords_nl", "enrich_provenance"]):
                enriched_chunks.append(chunk)
                stats["cached"] += 1
                continue
            
            # Check cache
            chunk_text = chunk.get("text", "")
            cached_result = cache.get(chunk_id, file_sha, chunk_text)
            if cached_result:
                # Add cached enrichment to chunk
                enriched_chunk = chunk.copy()
                enriched_chunk.update(cached_result)
                enriched_chunks.append(enriched_chunk)
                stats["cached"] += 1
                continue
            
            if dry_run:
                print(f"  Would enrich chunk {chunk_id}")
                enriched_chunks.append(chunk)
                stats["processed"] += 1
                continue
            
            # Enrich chunk
            try:
                enrichment = enricher.enrich_chunk(chunk, file_synopsis_nl, file_synopsis_hash)
                
                # Add enrichment to chunk
                enriched_chunk = chunk.copy()
                enriched_chunk.update(enrichment)
                enriched_chunks.append(enriched_chunk)
                
                # Cache the result
                cache.set(
                    chunk_id, file_sha, chunk_text, file_synopsis_hash,
                    enrichment["summary_nl"], enrichment["keywords_nl"], model
                )
                
                if enrichment["enrich_provenance"]["skipped_reason"] == "generated":
                    stats["generated"] += 1
                else:
                    stats["processed"] += 1
                    
            except Exception as e:
                print(f"Error enriching chunk {chunk_id}: {e}")
                enriched_chunks.append(chunk)  # Keep original chunk
                stats["errors"] += 1
    
    if not dry_run:
        print(f"Saving enriched chunks to {output_path}...")
        save_chunks(enriched_chunks, output_path)
    
    return stats


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enrich existing code chunks with Dutch LLM summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic enrichment
  python tools/enrich_chunks_v3.py --in chunks.jsonl --out chunks.enriched.jsonl
  
  # With custom settings
  python tools/enrich_chunks_v3.py \\
    --in chunks.jsonl \\
    --out chunks.enriched.jsonl \\
    --model qwen2.5-coder:7b-instruct \\
    --max-workers 6 \\
    --batch-size 16 \\
    --retry 3
  
  # Dry run to see what would be processed
  python tools/enrich_chunks_v3.py --in chunks.jsonl --out chunks.enriched.jsonl --dry-run
        """
    )
    
    parser.add_argument(
        "--in", dest="input_path", required=True,
        help="Input JSONL file with chunks to enrich"
    )
    parser.add_argument(
        "--out", dest="output_path", required=True,
        help="Output JSONL file for enriched chunks"
    )
    parser.add_argument(
        "--model", default="qwen2.5-coder:7b-instruct",
        help="LLM model to use (default: qwen2.5-coder:7b-instruct)"
    )
    parser.add_argument(
        "--lang", default="nl",
        help="Language for summaries (default: nl)"
    )
    parser.add_argument(
        "--max-workers", type=int, default=os.cpu_count() or 4,
        help="Maximum number of worker threads (default: CPU count)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for processing (default: 16)"
    )
    parser.add_argument(
        "--retry", type=int, default=3,
        help="Maximum number of retries for failed requests (default: 3)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without making changes"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing enrichment fields"
    )
    parser.add_argument(
        "--cache-path", default=".cache/enrichment.sqlite",
        help="Path to SQLite cache file (default: .cache/enrichment.sqlite)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_path):
        print(f"Error: Input file {args.input_path} does not exist")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Enriching chunks with Dutch summaries...")
    print(f"Model: {args.model}")
    print(f"Max workers: {args.max_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dry run: {args.dry_run}")
    print(f"Overwrite: {args.overwrite}")
    print()
    
    try:
        stats = enrich_chunks(
            args.input_path,
            args.output_path,
            args.model,
            args.max_workers,
            args.batch_size,
            args.retry,
            args.dry_run,
            args.overwrite,
            args.cache_path
        )
        
        print("\nEnrichment completed!")
        print(f"Total chunks: {stats['total']}")
        print(f"Processed with LLM: {stats['processed']}")
        print(f"Used cache: {stats['cached']}")
        print(f"Generated files (templated): {stats['generated']}")
        print(f"Errors: {stats['errors']}")
        
        if args.dry_run:
            print("\nThis was a dry run. No files were modified.")
        
    except KeyboardInterrupt:
        print("\nEnrichment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during enrichment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
