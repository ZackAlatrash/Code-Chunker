#!/usr/bin/env python3
"""
Chunk Quality Inspector

Analyzes and displays chunk quality metrics, text snippets, and LLM enrichment
in an easy-to-read format. Helps identify issues with chunking and enrichment.

Usage:
    python tools/inspect_chunk_quality.py <chunks.jsonl> [options]
    
    Options:
        --file FILE          Filter by specific file path
        --min-tokens N       Only show chunks with N+ tokens
        --max-tokens N       Only show chunks with N- tokens
        --kind KIND          Filter by primary_kind (function, type, method, etc.)
        --no-summary         Show chunks without LLM summary
        --sample N           Show only N random samples
        --interactive        Launch interactive browser
        --output-html FILE   Generate HTML report
"""

import json
import sys
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict, Counter


class ChunkQualityInspector:
    def __init__(self, chunks_file: str):
        self.chunks_file = chunks_file
        self.chunks = []
        self.load_chunks()
    
    def load_chunks(self):
        """Load chunks from JSONL file"""
        print(f"📂 Loading chunks from {self.chunks_file}...")
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line)
                    self.chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Failed to parse line {line_num}: {e}")
        print(f"✅ Loaded {len(self.chunks)} chunks\n")
    
    def filter_chunks(self, **filters) -> List[Dict[str, Any]]:
        """Filter chunks based on criteria"""
        filtered = self.chunks
        
        if filters.get('file'):
            file_pattern = filters['file'].lower()
            filtered = [c for c in filtered if file_pattern in c.get('rel_path', '').lower()]
        
        if filters.get('min_tokens'):
            filtered = [c for c in filtered if c.get('n_tokens', 0) >= filters['min_tokens']]
        
        if filters.get('max_tokens'):
            filtered = [c for c in filtered if c.get('n_tokens', 0) <= filters['max_tokens']]
        
        if filters.get('kind'):
            filtered = [c for c in filtered if c.get('primary_kind') == filters['kind']]
        
        if filters.get('no_summary'):
            filtered = [c for c in filtered if not c.get('summary_en')]
        
        if filters.get('sample'):
            filtered = random.sample(filtered, min(filters['sample'], len(filtered)))
        
        return filtered
    
    def print_summary_stats(self):
        """Print overall quality statistics"""
        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║                         CHUNK QUALITY SUMMARY                              ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print()
        
        total = len(self.chunks)
        
        # Token statistics
        tokens = [c.get('n_tokens', 0) for c in self.chunks]
        avg_tokens = sum(tokens) / len(tokens) if tokens else 0
        max_tokens = max(tokens) if tokens else 0
        min_tokens = min(tokens) if tokens else 0
        
        print(f"📊 Basic Statistics:")
        print(f"   Total chunks:        {total:,}")
        print(f"   Avg tokens/chunk:    {avg_tokens:.1f}")
        print(f"   Token range:         {min_tokens} - {max_tokens}")
        print()
        
        # Kind distribution
        kinds = Counter(c.get('primary_kind', 'unknown') for c in self.chunks)
        print(f"🔍 Chunk Types:")
        for kind, count in kinds.most_common(10):
            pct = count / total * 100
            print(f"   {kind:15s}: {count:5d} ({pct:5.1f}%)")
        print()
        
        # LLM Enrichment stats
        with_summary = sum(1 for c in self.chunks if c.get('summary_en'))
        with_keywords = sum(1 for c in self.chunks if c.get('keywords_en'))
        
        print(f"✨ LLM Enrichment:")
        print(f"   With summary:        {with_summary:,} ({with_summary/total*100:.1f}%)")
        print(f"   With keywords:       {with_keywords:,} ({with_keywords/total*100:.1f}%)")
        print()
        
        # Quality issues
        issues = self.detect_quality_issues()
        if issues:
            print(f"⚠️  Quality Issues Detected:")
            for issue_type, count in issues.items():
                print(f"   {issue_type:30s}: {count}")
            print()
        else:
            print(f"✅ No quality issues detected!")
            print()
    
    def detect_quality_issues(self) -> Dict[str, int]:
        """Detect potential quality issues"""
        issues = {}
        
        # Very small chunks (< 10 tokens)
        tiny = sum(1 for c in self.chunks if c.get('n_tokens', 0) < 10)
        if tiny > 0:
            issues['Tiny chunks (< 10 tokens)'] = tiny
        
        # Very large chunks (> 1000 tokens)
        huge = sum(1 for c in self.chunks if c.get('n_tokens', 0) > 1000)
        if huge > 0:
            issues['Huge chunks (> 1000 tokens)'] = huge
        
        # Empty text
        empty_text = sum(1 for c in self.chunks if not c.get('text', '').strip())
        if empty_text > 0:
            issues['Empty text'] = empty_text
        
        # Incomplete code (doesn't end with }, ), ;, or newline)
        incomplete = sum(1 for c in self.chunks 
                        if c.get('text', '').strip() 
                        and not c.get('text', '').strip()[-1] in '});"\n')
        if incomplete > 0:
            issues['Potentially incomplete code'] = incomplete
        
        # Missing LLM enrichment (if any chunks have it)
        has_enrichment = any(c.get('summary_en') for c in self.chunks)
        if has_enrichment:
            missing = sum(1 for c in self.chunks if not c.get('summary_en'))
            if missing > 0:
                issues['Missing LLM summary'] = missing
        
        # Very short summaries (< 20 chars)
        short_summary = sum(1 for c in self.chunks 
                           if c.get('summary_en') and len(c.get('summary_en', '')) < 20)
        if short_summary > 0:
            issues['Very short summaries (< 20 chars)'] = short_summary
        
        # Few keywords (< 3)
        few_keywords = sum(1 for c in self.chunks 
                          if c.get('keywords_en') and len(c.get('keywords_en', [])) < 3)
        if few_keywords > 0:
            issues['Few keywords (< 3)'] = few_keywords
        
        return issues
    
    def display_chunk_detailed(self, chunk: Dict[str, Any], index: int = 0):
        """Display detailed view of a single chunk"""
        print("=" * 80)
        print(f"CHUNK #{index + 1}")
        print("=" * 80)
        
        # Basic info
        print(f"\n📄 File: {chunk.get('rel_path', 'N/A')}")
        print(f"   Lines: {chunk.get('start_line', '?')}-{chunk.get('end_line', '?')}")
        print(f"   Tokens: {chunk.get('n_tokens', 0)}")
        print(f"   Kind: {chunk.get('primary_kind', 'unknown')}")
        print(f"   Symbol: {chunk.get('primary_symbol', '(none)')}")
        
        # Metadata
        if chunk.get('is_multi_declaration'):
            symbols = chunk.get('all_symbols', [])
            print(f"   Multi-declaration: {len(symbols)} symbols: {', '.join(symbols[:5])}")
        
        # Text preview
        text = chunk.get('text', '')
        print(f"\n📝 Code Text ({len(text)} chars):")
        print("─" * 80)
        lines = text.split('\n')
        if len(lines) <= 20:
            print(text)
        else:
            print('\n'.join(lines[:10]))
            print(f"\n... ({len(lines) - 20} lines omitted) ...\n")
            print('\n'.join(lines[-10:]))
        print("─" * 80)
        
        # LLM Enrichment
        if chunk.get('summary_en'):
            print(f"\n✨ LLM Summary:")
            print(f"   {chunk['summary_en']}")
        
        if chunk.get('keywords_en'):
            keywords = ', '.join(chunk['keywords_en'])
            print(f"\n🏷️  Keywords:")
            print(f"   {keywords}")
        
        if chunk.get('enrich_provenance'):
            prov = chunk['enrich_provenance']
            print(f"\n🔍 Enrichment Info:")
            print(f"   Model: {prov.get('model', 'N/A')}")
            print(f"   Created: {prov.get('created_at', 'N/A')}")
            if prov.get('skipped_reason'):
                print(f"   ⚠️  Skipped: {prov['skipped_reason']}")
        
        # Quality checks
        issues = []
        if chunk.get('n_tokens', 0) < 10:
            issues.append("Very small (< 10 tokens)")
        if chunk.get('n_tokens', 0) > 1000:
            issues.append("Very large (> 1000 tokens)")
        if not text.strip():
            issues.append("Empty text")
        if text.strip() and text.strip()[-1] not in '});"\n':
            issues.append("Potentially incomplete code")
        if chunk.get('summary_en') and len(chunk.get('summary_en', '')) < 20:
            issues.append("Very short summary")
        if chunk.get('keywords_en') and len(chunk.get('keywords_en', [])) < 3:
            issues.append("Few keywords")
        
        if issues:
            print(f"\n⚠️  Quality Issues:")
            for issue in issues:
                print(f"   - {issue}")
        
        print()
    
    def generate_html_report(self, output_file: str, filtered_chunks: List[Dict[str, Any]] = None):
        """Generate interactive HTML report"""
        chunks_to_show = filtered_chunks if filtered_chunks is not None else self.chunks
        
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Chunk Quality Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
            text-transform: uppercase;
        }
        .stat-card .value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }
        .chunk-card {
            background: white;
            margin-bottom: 20px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chunk-header {
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .chunk-title {
            font-weight: 600;
            color: #333;
        }
        .chunk-meta {
            display: flex;
            gap: 15px;
            font-size: 13px;
            color: #666;
        }
        .chunk-body {
            padding: 20px;
        }
        .code-block {
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            line-height: 1.5;
            margin: 15px 0;
        }
        .enrichment {
            background: #f0f7ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        .enrichment h4 {
            margin: 0 0 10px 0;
            color: #667eea;
        }
        .keywords {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .keyword {
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-function { background: #e3f2fd; color: #1976d2; }
        .badge-method { background: #f3e5f5; color: #7b1fa2; }
        .badge-type { background: #e8f5e9; color: #388e3c; }
        .badge-header { background: #fff3e0; color: #f57c00; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-error { background: #f8d7da; color: #721c24; }
        .filter-bar {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .filter-bar input, .filter-bar select {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-right: 10px;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background: #5568d3;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Chunk Quality Report</h1>
            <p>Generated from: """ + self.chunks_file + """</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total Chunks</h3>
                <div class="value">""" + str(len(chunks_to_show)) + """</div>
            </div>
            <div class="stat-card">
                <h3>Avg Tokens</h3>
                <div class="value">""" + str(int(sum(c.get('n_tokens', 0) for c in chunks_to_show) / len(chunks_to_show))) + """</div>
            </div>
            <div class="stat-card">
                <h3>With LLM Summary</h3>
                <div class="value">""" + str(sum(1 for c in chunks_to_show if c.get('summary_en'))) + """</div>
            </div>
            <div class="stat-card">
                <h3>With Keywords</h3>
                <div class="value">""" + str(sum(1 for c in chunks_to_show if c.get('keywords_en'))) + """</div>
            </div>
        </div>
        
        <div class="filter-bar">
            <input type="text" id="searchInput" placeholder="Search file path..." onkeyup="filterChunks()">
            <select id="kindFilter" onchange="filterChunks()">
                <option value="">All Types</option>
                <option value="function">Functions</option>
                <option value="method">Methods</option>
                <option value="type">Types</option>
                <option value="header">Headers</option>
            </select>
            <button onclick="showAll()">Show All</button>
        </div>
        
        <div id="chunks">
"""
        
        # Add individual chunks
        for i, chunk in enumerate(chunks_to_show[:100]):  # Limit to first 100 for performance
            kind = chunk.get('primary_kind', 'unknown')
            badge_class = f"badge-{kind}" if kind in ['function', 'method', 'type', 'header'] else 'badge-header'
            
            html += f"""
            <div class="chunk-card" data-kind="{kind}" data-file="{chunk.get('rel_path', '')}">
                <div class="chunk-header">
                    <div class="chunk-title">
                        <span class="badge {badge_class}">{kind}</span>
                        {chunk.get('primary_symbol') or '(header)'}
                    </div>
                    <div class="chunk-meta">
                        <span>📄 {chunk.get('rel_path', 'N/A').split('/')[-1]}</span>
                        <span>📏 {chunk.get('start_line')}-{chunk.get('end_line')}</span>
                        <span>🔢 {chunk.get('n_tokens', 0)} tokens</span>
                    </div>
                </div>
                <div class="chunk-body">
            """
            
            # Code preview
            text = chunk.get('text', '')
            lines = text.split('\n')
            preview = '\n'.join(lines[:15])
            if len(lines) > 15:
                preview += f"\n\n... ({len(lines) - 15} more lines) ..."
            
            html += f"""
                    <div class="code-block">{self._escape_html(preview)}</div>
            """
            
            # LLM Enrichment
            if chunk.get('summary_en') or chunk.get('keywords_en'):
                html += """<div class="enrichment">"""
                if chunk.get('summary_en'):
                    html += f"""
                        <h4>✨ Summary</h4>
                        <p>{chunk['summary_en']}</p>
                    """
                if chunk.get('keywords_en'):
                    keywords_html = ''.join(f'<span class="keyword">{k}</span>' for k in chunk['keywords_en'])
                    html += f"""
                        <h4>🏷️ Keywords</h4>
                        <div class="keywords">{keywords_html}</div>
                    """
                html += """</div>"""
            
            html += """
                </div>
            </div>
            """
        
        if len(chunks_to_show) > 100:
            html += f"""
            <div class="stat-card" style="text-align: center; padding: 30px;">
                <p>Showing first 100 of {len(chunks_to_show)} chunks</p>
                <p>Use filters or command-line options to see specific chunks</p>
            </div>
            """
        
        html += """
        </div>
    </div>
    
    <script>
        function filterChunks() {
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            const kindFilter = document.getElementById('kindFilter').value;
            const chunks = document.querySelectorAll('.chunk-card');
            
            chunks.forEach(chunk => {
                const file = chunk.dataset.file.toLowerCase();
                const kind = chunk.dataset.kind;
                
                const matchesSearch = !searchTerm || file.includes(searchTerm);
                const matchesKind = !kindFilter || kind === kindFilter;
                
                if (matchesSearch && matchesKind) {
                    chunk.classList.remove('hidden');
                } else {
                    chunk.classList.add('hidden');
                }
            });
        }
        
        function showAll() {
            document.getElementById('searchInput').value = '';
            document.getElementById('kindFilter').value = '';
            filterChunks();
        }
    </script>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ HTML report generated: {output_file}")
        print(f"   Open in browser: file://{Path(output_file).absolute()}")
    
    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters"""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#39;'))


def main():
    parser = argparse.ArgumentParser(description='Inspect chunk quality and LLM enrichment')
    parser.add_argument('chunks_file', help='Path to chunks JSONL file')
    parser.add_argument('--file', help='Filter by file path')
    parser.add_argument('--min-tokens', type=int, help='Minimum token count')
    parser.add_argument('--max-tokens', type=int, help='Maximum token count')
    parser.add_argument('--kind', help='Filter by primary_kind')
    parser.add_argument('--no-summary', action='store_true', help='Show chunks without summary')
    parser.add_argument('--sample', type=int, help='Show N random samples')
    parser.add_argument('--output-html', help='Generate HTML report')
    parser.add_argument('--detailed', action='store_true', help='Show detailed view of each chunk')
    
    args = parser.parse_args()
    
    if not Path(args.chunks_file).exists():
        print(f"❌ Error: File not found: {args.chunks_file}")
        sys.exit(1)
    
    inspector = ChunkQualityInspector(args.chunks_file)
    inspector.print_summary_stats()
    
    # Apply filters
    filters = {
        'file': args.file,
        'min_tokens': args.min_tokens,
        'max_tokens': args.max_tokens,
        'kind': args.kind,
        'no_summary': args.no_summary,
        'sample': args.sample,
    }
    filters = {k: v for k, v in filters.items() if v is not None and v is not False}
    
    if filters:
        print(f"🔍 Applying filters: {filters}\n")
        filtered = inspector.filter_chunks(**filters)
        print(f"📊 Filtered Results: {len(filtered)} chunks\n")
    else:
        filtered = inspector.chunks
    
    # Display detailed view
    if args.detailed and filtered:
        num_to_show = min(5, len(filtered))
        print(f"\n{'='*80}")
        print(f"DETAILED VIEW - Showing {num_to_show} of {len(filtered)} chunks")
        print(f"{'='*80}\n")
        
        for i, chunk in enumerate(filtered[:num_to_show]):
            inspector.display_chunk_detailed(chunk, i)
    
    # Generate HTML report
    if args.output_html:
        inspector.generate_html_report(args.output_html, filtered)
    

if __name__ == '__main__':
    main()

