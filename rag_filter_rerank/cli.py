"""
CLI for RAG Filter-Rerank system.

Supports file-based and live search retrieval with trace and dry-run modes.
"""
import sys
import json
import argparse
import logging
from pathlib import Path
from .config import get_settings
from .pipeline import FilterRerankPipeline
from .retriever import get_retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Filter-Rerank: Two-stage relevance filtering for code Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a JSON file
  python -m rag_filter_rerank.cli "How does the scraper work?" --chunks-json hits.json
  
  # With custom top-k
  python -m rag_filter_rerank.cli "Explain the cache" --chunks-json hits.json --k 10
  
  # Trace mode (show intermediate results)
  python -m rag_filter_rerank.cli "What is the API?" --chunks-json hits.json --trace
  
  # Dry run (no answering, just show filtered/reranked chunks)
  python -m rag_filter_rerank.cli "How does auth work?" --chunks-json hits.json --dry-run
        """
    )
    
    parser.add_argument(
        "question",
        help="Developer question to answer"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of evidence chunks to use (default from config)"
    )
    
    parser.add_argument(
        "--chunks-json",
        help="Path to JSON/JSONL file with chunks (array or {hits:[...]})"
    )
    
    parser.add_argument(
        "--repo-ids",
        nargs="+",
        help="Repo IDs to search (for live search_v4 mode)"
    )
    
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable trace mode (show top-N at each stage)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: show filtered/reranked chunks without calling answerer"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for diagnostics JSON (default: /tmp/rag_filter_rerank_last.json)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        sys.exit(1)
    
    # Override k if provided
    if args.k:
        settings.PIPELINE_TOPK_EVIDENCE = args.k
    
    # Get retriever
    try:
        retriever = get_retriever(
            chunks_json=args.chunks_json,
            repo_ids=args.repo_ids
        )
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        logger.error("\nPlease provide --chunks-json with a valid JSON file,")
        logger.error("or ensure search_v4 is available for live search.")
        sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = FilterRerankPipeline(retriever, settings)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Print configuration
    logger.info("=" * 80)
    logger.info("RAG Filter-Rerank Pipeline")
    logger.info("=" * 80)
    logger.info(f"Question: {args.question}")
    logger.info(f"Mode: {settings.RAG_FILTER_RERANK}")
    logger.info(f"Recall top-N: {settings.PIPELINE_TOPN_RECALL}")
    logger.info(f"Filter threshold: {settings.FILTER_THRESHOLD}")
    logger.info(f"Evidence top-K: {settings.PIPELINE_TOPK_EVIDENCE}")
    logger.info(f"Rerank: {'disabled' if settings.DISABLE_RERANK else 'enabled'}")
    if args.dry_run:
        logger.info("DRY RUN MODE: Will not call answerer")
    logger.info("=" * 80)
    logger.info("")
    
    # Run pipeline
    try:
        if args.dry_run:
            # Run pipeline but skip answering
            result = pipeline.run(args.question, trace=args.trace)
            result.answer = "[DRY RUN - Answering skipped]"
        else:
            result = pipeline.run(args.question, trace=args.trace)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    
    # Print results
    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result.answer)
    print()
    
    # Print timing summary
    print("=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    timing = result.timing_ms
    print(f"Retrieval:  {timing.get('retrieval_ms', 0):>6} ms")
    print(f"Filter:     {timing.get('filter_ms', 0):>6} ms")
    print(f"Rerank:     {timing.get('rerank_ms', 0):>6} ms")
    print(f"Answer:     {timing.get('answer_ms', 0):>6} ms")
    print(f"{'─' * 30}")
    print(f"Total:      {timing.get('total_ms', 0):>6} ms")
    print()
    
    # Print cache stats
    print("=" * 80)
    print("CACHE STATS")
    print("=" * 80)
    stats = result.cache_stats
    print(f"Backend:    {stats.get('backend', 'unknown')}")
    print(f"Hits:       {stats.get('hits', 0)}")
    print(f"Misses:     {stats.get('misses', 0)}")
    print(f"Hit rate:   {stats.get('hit_rate', '0%')}")
    print(f"Size:       {stats.get('size_mb', 0):.2f} MB")
    print()
    
    # Print pipeline stats
    print("=" * 80)
    print("PIPELINE STATS")
    print("=" * 80)
    print(f"Recalled:   {result.recall_n} chunks")
    print(f"Filtered:   {result.filtered_m} chunks (threshold>={settings.FILTER_THRESHOLD})")
    print(f"Evidence:   {result.evidence_k} chunks")
    print()
    
    # Show top reranked chunks if trace
    if args.trace or args.dry_run:
        print("=" * 80)
        print(f"TOP {min(len(result.reranked), 10)} RERANKED CHUNKS")
        print("=" * 80)
        for i, item in enumerate(result.reranked[:10], 1):
            print(f"[{i}] {item['rel_path']}:{item['start_line']}-{item['end_line']}")
            print(f"    provenance: {item['provenance_id']}")
            if item.get('filter_score') is not None:
                print(f"    filter: {item['filter_score']}")
            if item.get('rerank_score') is not None:
                print(f"    rerank: {item['rerank_score']:.4f}")
        print()
    
    # Save diagnostics
    output_path = args.output or "/tmp/rag_filter_rerank_last.json"
    try:
        diagnostics = {
            "question": args.question,
            "recall_n": result.recall_n,
            "filtered_m": result.filtered_m,
            "evidence_k": result.evidence_k,
            "reranked": result.reranked,
            "timing_ms": result.timing_ms,
            "cache_stats": result.cache_stats,
            "flags": result.flags,
            "answer_length": len(result.answer)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostics, f, indent=2)
        
        logger.info(f"Diagnostics saved to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save diagnostics: {e}")
    
    print("=" * 80)
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()

