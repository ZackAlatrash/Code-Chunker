#!/usr/bin/env python3
"""
Test runner for build_chunks_v3.py tests.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests for build_chunks_v3.py."""
    test_dir = Path(__file__).parent / "tests" / "chunking_v3"
    
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return False
    
    # Find all test files
    test_files = list(test_dir.glob("test_*.py"))
    
    if not test_files:
        print("No test files found")
        return False
    
    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file.name}")
    
    # Run tests
    cmd = [sys.executable, "-m", "pytest", str(test_dir), "-v"]
    
    print(f"\nRunning tests with command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("\n❌ pytest not found. Install with: pip install pytest")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
