#!/usr/bin/env python3
"""
Test Chunk Doctor Pipeline

This script tests the complete chunking pipeline:
1. Run build_chunks_v3 on the weather forecast repository
2. Apply chunk_doctor --fix to the generated chunks
3. Compare against a golden file
4. Report quality improvements
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd: List[str], description: str) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    logger.info(f"🔨 {description}")
    logger.info(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"❌ {description} failed")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        sys.exit(1)
    
    logger.info(f"✅ {description} completed successfully")
    return result

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of chunks."""
    chunks = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

def save_jsonl(chunks: List[Dict[str, Any]], file_path: str):
    """Save chunks to JSONL file."""
    with open(file_path, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')

def normalize_chunk_for_comparison(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize chunk for comparison by ignoring variable fields."""
    normalized = chunk.copy()
    
    # Ignore fields that change between runs
    ignore_fields = ['chunk_id', 'created_at', 'file_sha']
    for field in ignore_fields:
        if field in normalized:
            del normalized[field]
    
    # Normalize token_counts to 0 for comparison
    if 'token_counts' in normalized:
        normalized['token_counts'] = {
            'header': 0,
            'core': 0,
            'footer': 0,
            'total': 0
        }
    
    # Sort lists for consistent comparison
    for field in ['symbols_defined', 'symbols_referenced', 'imports_used']:
        if field in normalized and isinstance(normalized[field], list):
            normalized[field] = sorted(normalized[field])
    
    # Sort qa_terms string
    if 'qa_terms' in normalized and isinstance(normalized['qa_terms'], str):
        terms = [term.strip() for term in normalized['qa_terms'].split(',')]
        normalized['qa_terms'] = ', '.join(sorted(terms))
    
    return normalized

def compare_chunks(actual_chunks: List[Dict[str, Any]], expected_chunks: List[Dict[str, Any]]) -> bool:
    """Compare actual chunks against expected chunks."""
    logger.info(f"🔍 Comparing {len(actual_chunks)} actual chunks against {len(expected_chunks)} expected chunks")
    
    if len(actual_chunks) != len(expected_chunks):
        logger.error(f"❌ Length mismatch: actual={len(actual_chunks)}, expected={len(expected_chunks)}")
        return False
    
    # Sort chunks by path and start_line for consistent comparison
    def chunk_sort_key(chunk):
        return (chunk.get('path', ''), chunk.get('start_line', 0))
    
    actual_sorted = sorted(actual_chunks, key=chunk_sort_key)
    expected_sorted = sorted(expected_chunks, key=chunk_sort_key)
    
    for i, (actual, expected) in enumerate(zip(actual_sorted, expected_sorted)):
        actual_norm = normalize_chunk_for_comparison(actual)
        expected_norm = normalize_chunk_for_comparison(expected)
        
        if actual_norm != expected_norm:
            logger.error(f"❌ Chunk {i} mismatch:")
            logger.error(f"  Actual path: {actual.get('path', 'unknown')} line {actual.get('start_line', 'unknown')}")
            logger.error(f"  Expected path: {expected.get('path', 'unknown')} line {expected.get('start_line', 'unknown')}")
            logger.error(f"  Actual: {json.dumps(actual_norm, indent=2)}")
            logger.error(f"  Expected: {json.dumps(expected_norm, indent=2)}")
            return False
    
    logger.info("✅ All chunks match golden file!")
    return True

def analyze_chunk_quality(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze chunk quality and return statistics."""
    stats = {
        'total_chunks': len(chunks),
        'go_chunks': 0,
        'file_headers': 0,
        'interfaces': 0,
        'structs': 0,
        'functions': 0,
        'methods': 0,
        'avg_tokens': 0,
        'chunks_with_imports': 0,
        'chunks_with_symbols': 0,
        'neighbor_chains_complete': 0
    }
    
    total_tokens = 0
    neighbor_chains = 0
    
    for chunk in chunks:
        # Count by language
        if chunk.get('language') == 'go':
            stats['go_chunks'] += 1
        
        # Count by ast_path
        ast_path = chunk.get('ast_path', '')
        if ast_path == 'go:file_header':
            stats['file_headers'] += 1
        elif 'interface' in ast_path:
            stats['interfaces'] += 1
        elif 'struct' in ast_path:
            stats['structs'] += 1
        elif 'function' in ast_path:
            stats['functions'] += 1
        elif 'method' in ast_path:
            stats['methods'] += 1
        
        # Count tokens
        token_counts = chunk.get('token_counts', {})
        if isinstance(token_counts, dict):
            total_tokens += token_counts.get('total', 0)
        
        # Count chunks with imports
        if chunk.get('imports_used'):
            stats['chunks_with_imports'] += 1
        
        # Count chunks with symbols
        if chunk.get('symbols_referenced'):
            stats['chunks_with_symbols'] += 1
        
        # Check neighbor chains
        neighbors = chunk.get('neighbors', {})
        if neighbors.get('prev') is not None or neighbors.get('next') is not None:
            neighbor_chains += 1
    
    stats['avg_tokens'] = total_tokens / len(chunks) if chunks else 0
    stats['neighbor_chains_complete'] = neighbor_chains
    
    return stats

def main():
    """Main test pipeline."""
    logger.info("🧪 Starting Chunk Doctor Pipeline Test")
    
    # Repository path
    repo_path = "/Users/zack.alatrash/CompanyRepos/crunding-weather_foreca_proxy_service-8aae91880849"
    
    # Check if repository exists
    if not Path(repo_path).exists():
        logger.error(f"❌ Repository not found: {repo_path}")
        sys.exit(1)
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        raw_chunks_file = temp_path / "raw_chunks.jsonl"
        fixed_chunks_file = temp_path / "fixed_chunks.jsonl"
        golden_file = temp_path / "golden_chunks.jsonl"
        
        logger.info(f"📁 Using temporary directory: {temp_dir}")
        
        # Step 1: Run build_chunks_v3
        build_cmd = [
            "python", "build_chunks_v3.py",
            "--root", repo_path,
            "--out", str(raw_chunks_file),
            "--repo", "weather-forecast-service",
            "--max-total", "250",
            "--log-level", "INFO",
            "--force"
        ]
        
        run_command(build_cmd, "Running build_chunks_v3 on weather forecast repository")
        
        # Load raw chunks
        raw_chunks = load_jsonl(str(raw_chunks_file))
        logger.info(f"📊 Generated {len(raw_chunks)} raw chunks")
        
        # Step 2: Apply chunk_doctor --fix
        doctor_cmd = [
            "python", "-m", "tools.chunk_doctor",
            "--fix", str(raw_chunks_file)
        ]
        
        result = run_command(doctor_cmd, "Running chunk_doctor --fix")
        
        # Save fixed chunks
        with open(fixed_chunks_file, 'w') as f:
            f.write(result.stdout)
        
        # Load fixed chunks
        fixed_chunks = load_jsonl(str(fixed_chunks_file))
        logger.info(f"📊 Generated {len(fixed_chunks)} fixed chunks")
        
        # Step 3: Analyze quality
        raw_stats = analyze_chunk_quality(raw_chunks)
        fixed_stats = analyze_chunk_quality(fixed_chunks)
        
        logger.info("📈 Quality Analysis:")
        logger.info(f"  Raw chunks: {raw_stats}")
        logger.info(f"  Fixed chunks: {fixed_stats}")
        
        # Step 4: Create golden file (first run)
        golden_file_path = Path("tests/goldens/go/weather_forecast_service.jsonl")
        golden_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not golden_file_path.exists():
            logger.info("📝 Creating golden file (first run)")
            # Sort chunks by path and start_line for consistent golden file
            def chunk_sort_key(chunk):
                return (chunk.get('path', ''), chunk.get('start_line', 0))
            sorted_chunks = sorted(fixed_chunks, key=chunk_sort_key)
            save_jsonl(sorted_chunks, str(golden_file_path))
            logger.info(f"✅ Golden file created: {golden_file_path}")
        else:
            # Step 5: Compare against golden file
            expected_chunks = load_jsonl(str(golden_file_path))
            logger.info(f"📖 Loaded {len(expected_chunks)} expected chunks from golden file")
            
            if compare_chunks(fixed_chunks, expected_chunks):
                logger.info("🎉 All tests passed! Chunk Doctor pipeline is working correctly.")
            else:
                logger.error("❌ Tests failed! Chunks do not match golden file.")
                sys.exit(1)
        
        # Step 6: Report improvements
        logger.info("🎯 Chunk Doctor Pipeline Results:")
        logger.info(f"  ✅ Generated {len(raw_chunks)} raw chunks")
        logger.info(f"  ✅ Fixed {len(fixed_chunks)} chunks with chunk_doctor")
        logger.info(f"  ✅ Go chunks: {fixed_stats['go_chunks']}")
        logger.info(f"  ✅ File headers: {fixed_stats['file_headers']}")
        logger.info(f"  ✅ Interfaces: {fixed_stats['interfaces']}")
        logger.info(f"  ✅ Structs: {fixed_stats['structs']}")
        logger.info(f"  ✅ Functions: {fixed_stats['functions']}")
        logger.info(f"  ✅ Methods: {fixed_stats['methods']}")
        logger.info(f"  ✅ Average tokens: {fixed_stats['avg_tokens']:.1f}")
        logger.info(f"  ✅ Chunks with imports: {fixed_stats['chunks_with_imports']}")
        logger.info(f"  ✅ Chunks with symbols: {fixed_stats['chunks_with_symbols']}")
        logger.info(f"  ✅ Neighbor chains: {fixed_stats['neighbor_chains_complete']}")

if __name__ == "__main__":
    main()
