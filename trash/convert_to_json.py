#!/usr/bin/env python3
"""
Convert JSONL to JSON

This script converts the JSONL chunks file to a proper JSON file for easier viewing.
"""

import json
import sys
from pathlib import Path

def convert_jsonl_to_json(jsonl_file: str, json_file: str):
    """Convert JSONL file to JSON file."""
    chunks = []
    
    print(f"📖 Reading {jsonl_file}...")
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing line {line_num}: {e}")
                    continue
    
    print(f"📊 Loaded {len(chunks)} chunks")
    
    print(f"💾 Writing to {json_file}...")
    with open(json_file, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"✅ Successfully converted {len(chunks)} chunks to {json_file}")

def main():
    """Main function."""
    jsonl_file = "weather_forecast_quality_check.jsonl"
    json_file = "weather_forecast_quality_check.json"
    
    if not Path(jsonl_file).exists():
        print(f"❌ File not found: {jsonl_file}")
        sys.exit(1)
    
    convert_jsonl_to_json(jsonl_file, json_file)

if __name__ == "__main__":
    main()
