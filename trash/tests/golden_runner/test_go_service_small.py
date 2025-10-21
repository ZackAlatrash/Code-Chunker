#!/usr/bin/env python3
"""
Golden test runner for Go service chunking.

This test:
1. Runs build_chunks_v3 on the service.go fixture
2. Pipes the result through tools.chunk_doctor --fix
3. Compares to the golden file (field-by-field, ignoring chunk_id and created_at)
4. Prints a readable diff on failure
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import difflib


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of JSON objects."""
    chunks = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def save_jsonl(chunks: List[Dict[str, Any]], file_path: Path) -> None:
    """Save list of JSON objects to JSONL file."""
    with open(file_path, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')


def normalize_chunk_for_comparison(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize chunk by removing fields that should be ignored in comparison."""
    normalized = chunk.copy()
    
    # Remove fields that are expected to differ
    normalized.pop('chunk_id', None)
    normalized.pop('created_at', None)
    
    # Normalize token_counts to 0 for comparison (since we don't care about exact counts)
    if 'token_counts' in normalized:
        normalized['token_counts'] = {
            'header': 0,
            'core': 0,
            'footer': 0,
            'total': 0
        }
    
    # Normalize file_sha to placeholder
    if 'file_sha' in normalized:
        normalized['file_sha'] = 'PLACEHOLDER'
    
    # Normalize neighbor chunk_ids to placeholders
    if 'neighbors' in normalized:
        neighbors = normalized['neighbors']
        if neighbors.get('prev'):
            neighbors['prev'] = 'PLACEHOLDER'
        if neighbors.get('next'):
            neighbors['next'] = 'PLACEHOLDER'
    
    # Sort lists for consistent comparison
    if 'symbols_defined' in normalized and isinstance(normalized['symbols_defined'], list):
        normalized['symbols_defined'] = sorted(normalized['symbols_defined'])
    if 'symbols_referenced' in normalized and isinstance(normalized['symbols_referenced'], list):
        normalized['symbols_referenced'] = sorted(normalized['symbols_referenced'])
    if 'imports_used' in normalized and isinstance(normalized['imports_used'], list):
        normalized['imports_used'] = sorted(normalized['imports_used'])
    
    # Sort qa_terms for consistent comparison
    if 'qa_terms' in normalized and isinstance(normalized['qa_terms'], str):
        terms = [term.strip() for term in normalized['qa_terms'].split(',')]
        normalized['qa_terms'] = ', '.join(sorted(terms))
    
    return normalized


def compare_chunks(actual: List[Dict[str, Any]], expected: List[Dict[str, Any]]) -> bool:
    """Compare actual and expected chunks, returning True if they match."""
    if len(actual) != len(expected):
        print(f"❌ Length mismatch: actual={len(actual)}, expected={len(expected)}")
        return False
    
    for i, (act_chunk, exp_chunk) in enumerate(zip(actual, expected)):
        act_norm = normalize_chunk_for_comparison(act_chunk)
        exp_norm = normalize_chunk_for_comparison(exp_chunk)
        
        if act_norm != exp_norm:
            print(f"❌ Chunk {i} mismatch:")
            print(f"   AST Path: {act_chunk.get('ast_path', 'N/A')}")
            
            # Print detailed diff
            act_str = json.dumps(act_norm, indent=2, sort_keys=True)
            exp_str = json.dumps(exp_norm, indent=2, sort_keys=True)
            
            diff = difflib.unified_diff(
                exp_str.splitlines(keepends=True),
                act_str.splitlines(keepends=True),
                fromfile=f'expected_chunk_{i}',
                tofile=f'actual_chunk_{i}',
                lineterm=''
            )
            
            for line in diff:
                print(f"   {line.rstrip()}")
            
            return False
    
    return True


def run_build_chunks_v3(fixture_path: Path, output_path: Path) -> bool:
    """Run build_chunks_v3 on the fixture and save output."""
    try:
        cmd = [
            sys.executable, 'build_chunks_v3.py',
            '--file', str(fixture_path),
            '--out', str(output_path),
            '--repo', 'test-repo',
            '--max-total', '500',  # Higher limit for this test
            '--log-level', 'INFO',  # Reduce noise
            '--force'  # Force rebuild
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            print(f"❌ build_chunks_v3 failed with return code {result.returncode}")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to run build_chunks_v3: {e}")
        return False


def run_chunk_doctor_fix(input_path: Path, output_path: Path) -> bool:
    """Run chunk_doctor --fix on the input and save output."""
    try:
        cmd = [
            sys.executable, '-m', 'tools.chunk_doctor',
            '--fix', str(input_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            print(f"❌ chunk_doctor --fix failed with return code {result.returncode}")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False
        
        # Save the fixed output
        with open(output_path, 'w') as f:
            f.write(result.stdout)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to run chunk_doctor --fix: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Running Go service small golden test...")
    
    # Paths
    fixture_path = Path('tests/fixtures/go/service_small/service.go')
    golden_path = Path('tests/goldens/go/service_small.jsonl')
    
    if not fixture_path.exists():
        print(f"❌ Fixture not found: {fixture_path}")
        return 1
    
    if not golden_path.exists():
        print(f"❌ Golden file not found: {golden_path}")
        return 1
    
    # Load expected chunks
    print("📖 Loading golden chunks...")
    try:
        expected_chunks = load_jsonl(golden_path)
        print(f"   Loaded {len(expected_chunks)} expected chunks")
    except Exception as e:
        print(f"❌ Failed to load golden file: {e}")
        return 1
    
    # Run build_chunks_v3
    print("🔨 Running build_chunks_v3...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    
    try:
        if not run_build_chunks_v3(fixture_path, tmp_path):
            return 1
        
        # Run chunk_doctor --fix
        print("🔧 Running chunk_doctor --fix...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as fixed_file:
            fixed_path = Path(fixed_file.name)
        
        try:
            if not run_chunk_doctor_fix(tmp_path, fixed_path):
                return 1
            
            # Load actual chunks
            print("📖 Loading actual chunks...")
            actual_chunks = load_jsonl(fixed_path)
            print(f"   Loaded {len(actual_chunks)} actual chunks")
            
            # Compare chunks
            print("🔍 Comparing chunks...")
            if compare_chunks(actual_chunks, expected_chunks):
                print("✅ All chunks match golden file!")
                return 0
            else:
                print("❌ Chunks do not match golden file")
                return 1
                
        finally:
            if fixed_path.exists():
                fixed_path.unlink()
                
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == '__main__':
    sys.exit(main())
