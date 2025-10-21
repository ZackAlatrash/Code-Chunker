#!/usr/bin/env python3
"""
Qodo Search CLI

Query the Qodo embedding index using natural language queries.
"""
import argparse
import json
import logging
import sys
from typing import List, Dict, Any

from .config import get_settings
from .embedder import QodoEmbedder
from .opensearch_client import QodoOpenSearchClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_result(result: Dict[str, Any], index: int) -> str:
    """
    Format a search result for display.
    
    Args:
        result: Search result dictionary
        index: Result index (1-based)
        
    Returns:
        Formatted string
    """
    lines = [
        f"[{index}] {result.get('repo_id', 'unknown')} | {result.get('rel_path', 'unknown')} | L{result.get('start_line', 0)}-{result.get('end_line', 0)}",
        f"    Score: {result.get('_score', 0.0):.4f}",
        f"    Language: {result.get('language', 'unknown')}",
        f"    Package: {result.get('package', 'unknown')}"
    ]
    
    if result.get('summary_en'):
        summary = result['summary_en'][:100]
        if len(result['summary_en']) > 100:
            summary += "..."
        lines.append(f"    Summary: {summary}")
    
    return '\n'.join(lines)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Search code chunks using Qodo embeddings"
    )
    parser.add_argument(
        '--query',
        required=True,
        help='Search query'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=25,
        help='Number of results to return (default: 25)'
    )
    parser.add_argument(
        '--index-name',
        help='OpenSearch index name (overrides config)'
    )
    parser.add_argument(
        '--output-format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--source-fields',
        nargs='+',
        help='Specific fields to return in results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load settings
    settings = get_settings()
    if args.index_name:
        settings.OPENSEARCH_INDEX = args.index_name
    
    # Validate k
    if args.k > settings.MAX_K:
        logger.warning(f"K limited to {settings.MAX_K}")
        args.k = settings.MAX_K
    
    # Initialize components
    try:
        embedder = QodoEmbedder(settings)
        client = QodoOpenSearchClient(settings)
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)
    
    # Generate query embedding
    logger.info(f"Embedding query: '{args.query}'")
    try:
        query_vec = embedder.embed_query(args.query)
        logger.info(f"Generated query embedding: {query_vec.shape}")
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        sys.exit(1)
    
    # Perform KNN search
    logger.info(f"Searching index {settings.OPENSEARCH_INDEX} for {args.k} results...")
    try:
        results = client.knn_search(
            query_vec.tolist(),
            k=args.k,
            _source=args.source_fields
        )
        logger.info(f"Found {len(results)} results")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        sys.exit(1)
    
    if not results:
        logger.warning("No results found")
        return
    
    # Format and display results
    if args.output_format == 'json':
        # JSON output
        output = {
            'query': args.query,
            'k': args.k,
            'results_count': len(results),
            'results': results
        }
        print(json.dumps(output, indent=2))
    else:
        # Text output
        print(f"\n🔍 Search Results for: '{args.query}'")
        print(f"📊 Found {len(results)} results from {settings.OPENSEARCH_INDEX}")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(format_result(result, i))
            print()
        
        # Show top 3 scores for reference
        if len(results) >= 3:
            scores = [r.get('_score', 0.0) for r in results[:3]]
            print(f"📈 Top 3 scores: {[f'{s:.4f}' for s in scores]}")


if __name__ == '__main__':
    main()
