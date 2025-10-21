"""
CLI for search_v4: Test hybrid search end-to-end.

Usage:
    python -m search_v4.cli \\
        --query "where is GetForecastForLocation implemented?" \\
        --repos foreca \\
        --planner-out plan.json
"""
import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict

from .service import search_v4


def read_plan(path: str) -> Dict[str, Any]:
    """Read query plan from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_planner(query: str, guides_path: str = "") -> Dict[str, Any]:
    """
    Call query_planner.py to generate a query plan.
    
    Args:
        query: User query
        guides_path: Optional path to repo guides JSON
    
    Returns:
        Query plan dict
    
    Raises:
        RuntimeError: If planner fails
    """
    cmd = [
        "python",
        "scripts/query_planner.py",
        query,
        "--model", "qwen2.5-coder:7b-instruct"
    ]
    
    if guides_path and os.path.exists(guides_path):
        cmd += ["--repo-guides", guides_path]
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    if proc.returncode != 0:
        print("Query planner failed:", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("query_planner failed")
    
    return json.loads(proc.stdout)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid search for code_chunks_v4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With existing plan file:
  python -m search_v4.cli \\
      --query "where is GetForecastForLocation?" \\
      --repos foreca \\
      --planner-out plan.json
  
  # Let CLI call planner:
  python -m search_v4.cli \\
      --query "how does caching work?" \\
      --repos foreca weather \\
      --guides repo_guides.json \\
      --out results.json
        """
    )
    
    parser.add_argument(
        "--query",
        required=True,
        help="User query"
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        default=[],
        help="Repo IDs from router (1-3 repos)"
    )
    parser.add_argument(
        "--planner-out",
        default="",
        help="Path to existing plan JSON file (skip planner call)"
    )
    parser.add_argument(
        "--guides",
        default="",
        help="Path to repo guides JSON file (for planner)"
    )
    parser.add_argument(
        "--out",
        default="",
        help="Write results to file (default: stdout)"
    )
    
    args = parser.parse_args()
    
    # Get or generate plan
    if args.planner_out:
        print(f"Loading plan from: {args.planner_out}", file=sys.stderr)
        plan = read_plan(args.planner_out)
    else:
        print("Calling query_planner.py...", file=sys.stderr)
        plan = call_planner(args.query, guides_path=args.guides)
    
    # Execute search
    print(f"Searching repos: {args.repos}", file=sys.stderr)
    results = search_v4(args.query, args.repos, plan)
    
    # Output
    json_str = json.dumps(results, ensure_ascii=False, indent=2)
    
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"✅ Wrote results → {args.out}", file=sys.stderr)
        print(f"   Found {len(results['results'])} results", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()

