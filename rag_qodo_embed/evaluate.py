#!/usr/bin/env python3
"""
Qodo Evaluation CLI

A/B test Qodo embeddings against existing embedding models.
Compares retrieval quality between different indices.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import csv

from .config import get_settings
from .embedder import QodoEmbedder
from .opensearch_client import QodoOpenSearchClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_queries(queries_file: str) -> List[str]:
    """
    Load queries from file.
    
    Args:
        queries_file: Path to queries file (one per line)
        
    Returns:
        List of query strings
    """
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(queries)} queries from {queries_file}")
    return queries


def load_relevant_ids(relevant_file: str) -> Dict[str, Set[str]]:
    """
    Load relevant document IDs for each query.
    
    Args:
        relevant_file: Path to CSV file with columns: query, relevant_id
        
    Returns:
        Dictionary mapping query to set of relevant IDs
    """
    relevant = {}
    
    if not Path(relevant_file).exists():
        logger.warning(f"Relevant IDs file {relevant_file} not found, skipping precision calculation")
        return relevant
    
    try:
        with open(relevant_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                query = row['query'].strip()
                relevant_id = row['relevant_id'].strip()
                
                if query not in relevant:
                    relevant[query] = set()
                relevant[query].add(relevant_id)
        
        logger.info(f"Loaded relevant IDs for {len(relevant)} queries")
    except Exception as e:
        logger.error(f"Failed to load relevant IDs: {e}")
    
    return relevant


def search_index(
    client: QodoOpenSearchClient,
    embedder: QodoEmbedder,
    query: str,
    k: int = 10
) -> List[str]:
    """
    Search an index and return document IDs.
    
    Args:
        client: OpenSearch client
        embedder: Embedder for query
        query: Search query
        k: Number of results
        
    Returns:
        List of document IDs
    """
    try:
        query_vec = embedder.embed_query(query)
        results = client.knn_search(
            query_vec.tolist(),
            k=k,
            _source=['id']
        )
        return [r['id'] for r in results]
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        return []


def calculate_overlap(ids_a: List[str], ids_b: List[str]) -> float:
    """
    Calculate overlap percentage between two result sets.
    
    Args:
        ids_a: First result set
        ids_b: Second result set
        
    Returns:
        Overlap percentage (0.0 to 1.0)
    """
    if not ids_a and not ids_b:
        return 1.0
    
    if not ids_a or not ids_b:
        return 0.0
    
    set_a = set(ids_a)
    set_b = set(ids_b)
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


def calculate_precision_at_k(ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate precision@k.
    
    Args:
        ids: Retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Cutoff for precision calculation
        
    Returns:
        Precision@k (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    top_k = ids[:k]
    if not top_k:
        return 0.0
    
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_count / len(top_k)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="A/B test Qodo embeddings against existing models"
    )
    parser.add_argument(
        '--queries',
        required=True,
        help='Path to queries file (one per line)'
    )
    parser.add_argument(
        '--index-a',
        required=True,
        help='First index name (e.g., code_chunks_v4)'
    )
    parser.add_argument(
        '--index-b',
        required=True,
        help='Second index name (e.g., code_chunks_v5_qodo)'
    )
    parser.add_argument(
        '--relevant-ids',
        help='Path to CSV file with relevant IDs (columns: query, relevant_id)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of results to compare (default: 10)'
    )
    parser.add_argument(
        '--output',
        help='Path to output JSON file for detailed results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load queries and relevant IDs
    try:
        queries = load_queries(args.queries)
        relevant_ids = load_relevant_ids(args.relevant_ids) if args.relevant_ids else {}
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    if not queries:
        logger.error("No queries loaded")
        sys.exit(1)
    
    # Load settings
    settings = get_settings()
    
    # Initialize components
    try:
        embedder = QodoEmbedder(settings)
        client = QodoOpenSearchClient(settings)
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)
    
    # Run evaluation
    logger.info(f"Starting A/B evaluation: {args.index_a} vs {args.index_b}")
    logger.info(f"Testing {len(queries)} queries with k={args.k}")
    
    results = []
    total_overlap = 0.0
    total_precision_a = 0.0
    total_precision_b = 0.0
    
    for i, query in enumerate(queries, 1):
        logger.info(f"Evaluating query {i}/{len(queries)}: '{query[:50]}...'")
        
        # Search both indices
        client.OPENSEARCH_INDEX = args.index_a
        ids_a = search_index(client, embedder, query, args.k)
        
        client.OPENSEARCH_INDEX = args.index_b
        ids_b = search_index(client, embedder, query, args.k)
        
        # Calculate metrics
        overlap = calculate_overlap(ids_a, ids_b)
        total_overlap += overlap
        
        precision_a = 0.0
        precision_b = 0.0
        
        if query in relevant_ids:
            precision_a = calculate_precision_at_k(ids_a, relevant_ids[query], args.k)
            precision_b = calculate_precision_at_k(ids_b, relevant_ids[query], args.k)
            total_precision_a += precision_a
            total_precision_b += precision_b
        
        # Store result
        result = {
            'query': query,
            'overlap': overlap,
            'precision_a': precision_a,
            'precision_b': precision_b,
            'ids_a': ids_a,
            'ids_b': ids_b
        }
        results.append(result)
        
        logger.info(f"  Overlap: {overlap:.3f}, Precision A: {precision_a:.3f}, Precision B: {precision_b:.3f}")
    
    # Calculate averages
    avg_overlap = total_overlap / len(queries)
    avg_precision_a = total_precision_a / len(queries) if relevant_ids else 0.0
    avg_precision_b = total_precision_b / len(queries) if relevant_ids else 0.0
    
    # Print summary
    print("\n" + "="*80)
    print("A/B EVALUATION RESULTS")
    print("="*80)
    print(f"Index A: {args.index_a}")
    print(f"Index B: {args.index_b}")
    print(f"Queries tested: {len(queries)}")
    print(f"Results per query: {args.k}")
    print()
    print(f"Average overlap: {avg_overlap:.3f}")
    if relevant_ids:
        print(f"Average precision@k (A): {avg_precision_a:.3f}")
        print(f"Average precision@k (B): {avg_precision_b:.3f}")
        print(f"Precision improvement: {avg_precision_b - avg_precision_a:+.3f}")
    print()
    
    # Show top queries by overlap
    results_by_overlap = sorted(results, key=lambda x: x['overlap'], reverse=True)
    print("Top 5 queries by overlap:")
    for i, result in enumerate(results_by_overlap[:5], 1):
        print(f"  {i}. {result['overlap']:.3f} - '{result['query'][:60]}...'")
    
    # Show queries with biggest precision differences
    if relevant_ids:
        results_by_precision_diff = sorted(
            results, 
            key=lambda x: x['precision_b'] - x['precision_a'], 
            reverse=True
        )
        print("\nTop 5 queries by precision improvement (B vs A):")
        for i, result in enumerate(results_by_precision_diff[:5], 1):
            diff = result['precision_b'] - result['precision_a']
            print(f"  {i}. {diff:+.3f} - '{result['query'][:60]}...'")
    
    # Save detailed results
    if args.output:
        detailed_results = {
            'config': {
                'index_a': args.index_a,
                'index_b': args.index_b,
                'k': args.k,
                'queries_count': len(queries)
            },
            'summary': {
                'avg_overlap': avg_overlap,
                'avg_precision_a': avg_precision_a,
                'avg_precision_b': avg_precision_b,
                'precision_improvement': avg_precision_b - avg_precision_a
            },
            'results': results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info(f"Detailed results saved to {args.output}")


if __name__ == '__main__':
    main()
