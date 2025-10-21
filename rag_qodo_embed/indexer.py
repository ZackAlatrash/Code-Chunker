#!/usr/bin/env python3
"""
Qodo Indexer CLI

Index chunk JSON files into OpenSearch with Qodo embeddings.
Supports both array format and {hits: [...]} format.
"""
import argparse
import json
import logging
import sys
import gc
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import hashlib

from .config import get_settings
from .embedder import QodoEmbedder
from .opensearch_client import QodoOpenSearchClient
from .truncation import truncate_text

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_chunks_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Robustly load chunks from JSON (array or {hits: [...]}) or JSONL.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data_str = f.read().strip()

    # Try JSON first
    try:
        data = json.loads(data_str)
        if isinstance(data, list):
            logger.info(f"Loaded {len(data)} chunks from {file_path} (JSON array)")
            return data
        if isinstance(data, dict) and 'hits' in data:
            logger.info(f"Loaded {len(data['hits'])} chunks from {file_path} (JSON hits)")
            return data['hits']
        raise ValueError("JSON must be an array or {hits: [...]} format")
    except json.JSONDecodeError:
        pass

    # Fallback to JSONL
    chunks: List[Dict[str, Any]] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    logger.info(f"Loaded {len(chunks)} chunks from {file_path} (JSONL)")
    return chunks


def prepare_chunk_for_indexing(chunk: Dict[str, Any], settings) -> Dict[str, Any]:
    """
    Prepare a chunk for indexing with Qodo embeddings.
    
    Args:
        chunk: Original chunk dictionary
        settings: Configuration settings
        
    Returns:
        Prepared chunk with embedding metadata
    """
    # Extract text for embedding
    text = chunk.get('text', '')
    
    # Truncation for embedding only
    original_len = len(text)
    truncated_text = truncate_text(text, settings.TRUNCATE_CHARS)
    truncated_len = len(truncated_text)
    if truncated_len < original_len:
        logger.debug(f"Truncated text from {original_len} to {truncated_len} chars")

    # Hashes for audit
    raw_text = text or ''
    raw_text_hash = hashlib.sha256(raw_text.encode('utf-8')).hexdigest()[:16]
    truncated_text_hash = hashlib.sha256(truncated_text.encode('utf-8')).hexdigest()[:16]
    
    # Prepare document for indexing
    doc = {
        # Core chunk fields
        'id': chunk.get('id', ''),
        'repo_id': chunk.get('repo_id', ''),
        'rel_path': chunk.get('rel_path', ''),
        'path': chunk.get('path', ''),
        'abs_path': chunk.get('abs_path', ''),
        'ext': chunk.get('ext', ''),
        'language': chunk.get('language', ''),
        'package': chunk.get('package', ''),
        'chunk_number': chunk.get('chunk_number', 0),
        'start_line': chunk.get('start_line', 0),
        'end_line': chunk.get('end_line', 0),
        'text': raw_text,  # Keep original text in _source
        'summary_en': chunk.get('summary_en', ''),
        
        # Embedding metadata
        'model_name': settings.MODEL_ID,
        'model_version': '1.0.0',
        'embedding_type': 'code',
        'raw_text_hash': raw_text_hash,
        'truncated_text_hash': truncated_text_hash,
        'created_at': datetime.utcnow().isoformat() + 'Z'
    }
    
    return doc, truncated_text


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Index chunk JSON files with Qodo embeddings"
    )
    parser.add_argument(
        '--chunks-json',
        required=True,
        help='Path to chunk JSON file (array or {hits: [...]} format)'
    )
    parser.add_argument(
        '--index-name',
        help='OpenSearch index name (overrides config)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreate index if it exists'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for embedding (overrides config)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be indexed without actually indexing'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--start-offset',
        type=int,
        default=0,
        help='Resume indexing from this document offset (default: 0)'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load settings
    settings = get_settings()
    if args.index_name:
        settings.OPENSEARCH_INDEX = args.index_name
    if args.batch_size:
        settings.BATCH_SIZE = args.batch_size
    
    # Load chunks
    try:
        chunks = load_chunks_from_json(args.chunks_json)
    except Exception as e:
        logger.error(f"Failed to load chunks: {e}")
        sys.exit(1)
    
    if not chunks:
        logger.warning("No chunks to index")
        sys.exit(0)
    
    # Initialize components
    try:
        embedder = QodoEmbedder(settings)
        client = QodoOpenSearchClient(settings)
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)
    
    # Create index
    if not args.dry_run:
        if not client.create_index(force=args.force):
            logger.error("Failed to create index")
            sys.exit(1)
    
    # Prepare documents and collect texts for embedding
    docs = []
    texts_for_embedding = []
    
    logger.info("Preparing documents for indexing...")
    for i, chunk in enumerate(chunks):
        try:
            doc, truncated_text = prepare_chunk_for_indexing(chunk, settings)
            docs.append(doc)
            texts_for_embedding.append(truncated_text)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Prepared {i + 1}/{len(chunks)} documents")
                
        except Exception as e:
            logger.error(f"Failed to prepare chunk {i}: {e}")
            continue
    
    logger.info(f"Prepared {len(docs)} documents for indexing")
    
    if args.dry_run:
        logger.info("Dry run mode - would index the following documents:")
        for i, doc in enumerate(docs[:5]):  # Show first 5
            logger.info(f"  {i+1}. {doc['id']} - {doc['rel_path']}")
        if len(docs) > 5:
            logger.info(f"  ... and {len(docs) - 5} more")
        return
    
    # Stream embeddings and indexing in bounded batches to reduce memory
    embed_chunk_size = max(8, min(64, settings.BATCH_SIZE * 2))
    bulk_batch_size = 100
    total_indexed = 0
    
    # Resume support
    start_offset = max(0, args.start_offset)
    if start_offset > 0:
        logger.info(f"Resuming from document offset {start_offset}")
        # Align to window boundary
        start_offset = (start_offset // embed_chunk_size) * embed_chunk_size
        logger.info(f"Aligned to window boundary: {start_offset}")
    
    # Adaptive fallback truncation caps for memory errors
    fallback_truncations = [
        settings.TRUNCATE_CHARS,  # Original
        8000,                      # First fallback
        4000,                      # Second fallback
        2000                       # Last resort
    ]

    logger.info(f"Streaming indexing {len(docs)} documents (embed_chunk_size={embed_chunk_size}, bulk_batch_size={bulk_batch_size})...")
    for start in range(start_offset, len(docs), embed_chunk_size):
        end = min(start + embed_chunk_size, len(docs))
        sub_docs = docs[start:end]
        sub_texts = texts_for_embedding[start:end]

        # Adaptive retry with progressively smaller truncation caps
        sub_embeddings = None
        for attempt, trunc_cap in enumerate(fallback_truncations):
            try:
                # Re-truncate texts with current cap
                if attempt > 0:
                    logger.warning(f"Window {start}:{end} - retry {attempt} with truncation cap {trunc_cap}")
                    sub_texts = [truncate_text(texts_for_embedding[i], trunc_cap) for i in range(start, end)]
                
                sub_embeddings = embedder.embed_docs(sub_texts)
                logger.debug(f"Generated window embeddings: {sub_embeddings.shape}")
                
                if len(sub_embeddings) != len(sub_docs):
                    raise RuntimeError(
                        f"Embed/doc count mismatch in window {start}:{end}: embeddings={len(sub_embeddings)} docs={len(sub_docs)}"
                    )
                break  # Success
                
            except Exception as e:
                logger.warning(f"Window {start}:{end} failed with truncation cap {trunc_cap}: {e}")
                if attempt == len(fallback_truncations) - 1:
                    logger.error(f"❌ Giving up on window {start}:{end} after {len(fallback_truncations)} attempts")
                    logger.error(f"   To resume, run with: --start-offset {end}")
                    sub_embeddings = None
                    break
                # Clean up memory before retry
                gc.collect()
                if HAS_TORCH and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        
        # Skip window if all retries failed
        if sub_embeddings is None:
            logger.warning(f"⚠️  Skipping window {start}:{end} - could not generate embeddings")
            continue

        # Attach embeddings
        for d, emb in zip(sub_docs, sub_embeddings):
            d['embedding'] = emb.tolist()

        # Bulk index this window in smaller bulk batches
        for i in range(0, len(sub_docs), bulk_batch_size):
            batch = sub_docs[i:i + bulk_batch_size]
            try:
                if client.bulk_index(batch):
                    total_indexed += len(batch)
                    logger.info(f"Indexed {total_indexed}/{len(docs)} documents")
                else:
                    logger.error("Failed to index a bulk batch; aborting")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to index bulk batch: {e}")
                sys.exit(1)
        
        # Clean up memory after each window
        gc.collect()
        if HAS_TORCH and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    logger.info(f"✅ Successfully indexed {total_indexed} documents")
    
    # Post-index verification
    try:
        stats = client.get_index_stats()
        if stats:
            doc_count = stats.get('total', {}).get('docs', {}).get('count', 0)
            logger.info(f"Index {settings.OPENSEARCH_INDEX} now contains {doc_count} documents")
        sample = client.sample_one(fields=["id","rel_path","chunk_number","model_name","raw_text_hash","truncated_text_hash"])
        if sample:
            logger.info(f"Sample doc: {sample}")
    except Exception as e:
        logger.warning(f"Post-index verification failed: {e}")


if __name__ == '__main__':
    main()
