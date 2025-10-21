# Chunk Quality Monitoring Guide

This guide explains how to monitor and analyze chunk quality, especially for text content and LLM enrichment.

## Quick Start

### 1. Basic Quality Report

```bash
python tools/inspect_chunk_quality.py "path/to/chunks.jsonl"
```

This shows:
- Total chunks and token statistics
- Distribution of chunk types (function, method, type, etc.)
- LLM enrichment coverage (summaries and keywords)
- Quality issues detected

### 2. Generate Interactive HTML Report

```bash
python tools/inspect_chunk_quality.py "path/to/chunks.jsonl" \
  --output-html "quality_report.html"
```

Open the HTML file in your browser to:
- Browse chunks with syntax-highlighted code
- Search and filter by file or type
- See LLM summaries and keywords inline
- Identify quality issues visually

### 3. Detailed Text Inspection

```bash
python tools/inspect_chunk_quality.py "path/to/chunks.jsonl" \
  --detailed \
  --sample 10
```

Shows detailed view of 10 random chunks including:
- Full code text (or preview if too long)
- LLM summary and keywords
- Enrichment provenance
- Quality warnings

## Filtering Options

### Filter by File
```bash
python tools/inspect_chunk_quality.py "chunks.jsonl" \
  --file "export.go" \
  --detailed
```

### Filter by Token Count
```bash
# Find large chunks (> 800 tokens)
python tools/inspect_chunk_quality.py "chunks.jsonl" \
  --min-tokens 800 \
  --detailed

# Find tiny chunks (< 20 tokens)
python tools/inspect_chunk_quality.py "chunks.jsonl" \
  --max-tokens 20 \
  --detailed
```

### Filter by Chunk Type
```bash
# Only show functions
python tools/inspect_chunk_quality.py "chunks.jsonl" \
  --kind function \
  --detailed

# Only show types
python tools/inspect_chunk_quality.py "chunks.jsonl" \
  --kind type \
  --detailed
```

### Find Chunks Without LLM Enrichment
```bash
python tools/inspect_chunk_quality.py "chunks.jsonl" \
  --no-summary \
  --detailed
```

## Quality Metrics Explained

### Token Statistics
- **Avg tokens/chunk**: Should be reasonable for your token limit (e.g., 150-300 for 600 limit)
- **Token range**: Min-max helps identify outliers

### Chunk Type Distribution
- **method**: Instance methods (e.g., `func (s *Service) Method()`)
- **function**: Free functions (e.g., `func MyFunc()`)
- **type**: Type declarations (e.g., `type Config struct {}`)
- **header**: Package and imports

### LLM Enrichment Coverage
- **With summary**: % of chunks with `summary_en` field
- **With keywords**: % of chunks with `keywords_en` field

### Quality Issues

#### Tiny Chunks (< 10 tokens)
- **Cause**: Very small type definitions or minimal code
- **Action**: Usually OK if syntactically complete (e.g., `type ID string`)
- **Red Flag**: If text is incomplete or fragment

#### Huge Chunks (> 1000 tokens)
- **Cause**: Very large functions or types with many fields
- **Action**: Check if splitting would make sense
- **Usually OK**: If it's a complete, valid function

#### Potentially Incomplete Code
- **Cause**: Code doesn't end with `}`, `)`, `;`, or newline
- **Action**: Verify the chunk is complete in detailed view
- **Common**: False positives for valid small chunks

#### Missing LLM Summary
- **Cause**: Chunks were processed without `--llm-enrich`
- **Action**: Re-run with LLM enrichment if needed

#### Very Short Summaries (< 20 chars)
- **Cause**: LLM generated minimal summary
- **Action**: Check if summary is meaningful

#### Few Keywords (< 3)
- **Cause**: LLM couldn't extract enough keywords
- **Action**: Check if chunk content is minimal

## HTML Report Features

### Overview Dashboard
- Key metrics at a glance
- Quick filters for file path and chunk type
- Responsive grid layout

### Chunk Cards
- Color-coded badges for chunk types
- File path and line numbers
- Token count
- Syntax-highlighted code preview
- LLM enrichment displayed prominently

### Interactive Filters
- **Search**: Type to filter by file path
- **Type Dropdown**: Filter by function/method/type/header
- **Show All**: Clear all filters

### Performance
- Shows first 100 chunks for performance
- Use command-line filters to focus on specific areas

## Use Cases

### 1. Verify Nested Declaration Fixes

Check if structs with nested fields are complete:

```bash
python tools/inspect_chunk_quality.py "chunks.jsonl" \
  --kind type \
  --file "config" \
  --detailed
```

Look for:
- Complete opening and closing braces
- All nested fields included
- No "Potentially incomplete code" warnings

### 2. Audit LLM Enrichment Quality

Generate report of enriched chunks:

```bash
python tools/inspect_chunk_quality.py "enriched_chunks.jsonl" \
  --output-html "enrichment_audit.html" \
  --sample 50
```

Review in browser:
- Are summaries meaningful and concise?
- Do keywords match the code content?
- Are summaries in correct language (English)?
- No hallucinations (e.g., "exponential backoff" when not present)?

### 3. Find Problem Chunks

Find chunks that might be fragments:

```bash
python tools/inspect_chunk_quality.py "chunks.jsonl" \
  --max-tokens 15 \
  --kind type \
  --detailed
```

Verify each is complete:
- Has opening and closing braces
- Contains full type definition
- Not cut off mid-field

### 4. Compare Before/After Fixes

```bash
# Before
python tools/inspect_chunk_quality.py "before_fix.jsonl" \
  --output-html "before.html"

# After
python tools/inspect_chunk_quality.py "after_fix.jsonl" \
  --output-html "after.html"
```

Compare:
- Total chunk count (should decrease with fixes)
- Quality issues (should be fewer)
- Chunk completeness (spot check types and functions)

### 5. Sample Random Chunks for Manual Review

```bash
python tools/inspect_chunk_quality.py "chunks.jsonl" \
  --sample 20 \
  --detailed > quality_sample.txt
```

Manually review:
- Is code syntactically valid?
- Are summaries accurate?
- Are keywords relevant?

## Monitoring Workflow

### For Every Chunking Run

1. **Generate Basic Report**
   ```bash
   python tools/inspect_chunk_quality.py "chunks.jsonl"
   ```
   
2. **Check Quality Issues**
   - Review warnings in summary
   - If issues detected, investigate with filters

3. **Generate HTML Report**
   ```bash
   python tools/inspect_chunk_quality.py "chunks.jsonl" \
     --output-html "quality_$(date +%Y%m%d).html"
   ```

4. **Spot Check Samples**
   ```bash
   python tools/inspect_chunk_quality.py "chunks.jsonl" \
     --detailed \
     --sample 10
   ```

### For LLM-Enriched Chunks

1. **Check Enrichment Coverage**
   ```bash
   python tools/inspect_chunk_quality.py "enriched_chunks.jsonl"
   ```
   - Should be 100% or explain missing chunks

2. **Audit Summary Quality**
   ```bash
   python tools/inspect_chunk_quality.py "enriched_chunks.jsonl" \
     --sample 25 \
     --detailed
   ```
   - Read summaries - are they accurate and concise?
   - Check for hallucinations

3. **Audit Keyword Quality**
   - Are they lowercase and search-friendly?
   - Do they match the code content?
   - 5-10 keywords per chunk?

4. **Check for English-Only**
   - Summaries should be in English
   - No Dutch/German/other languages (unless configured)

## Advanced: Custom Analysis

The tool loads all chunks into memory as Python dictionaries. You can modify it to add custom analysis:

```python
# In ChunkQualityInspector class, add:

def find_hallucinations(self):
    """Find potential hallucinations in summaries"""
    forbidden = ['exponential backoff', 'circuit breaker', 'retry logic']
    issues = []
    
    for chunk in self.chunks:
        summary = chunk.get('summary_en', '').lower()
        text = chunk.get('text', '').lower()
        
        for term in forbidden:
            if term in summary and term not in text:
                issues.append({
                    'file': chunk['rel_path'],
                    'line': chunk['start_line'],
                    'term': term,
                    'summary': chunk['summary_en']
                })
    
    return issues
```

## Tips

### Performance
- For large JSONL files (> 10K chunks), use `--sample` to limit output
- HTML reports show first 100 chunks - use filters to focus
- Use `--file` or `--kind` to narrow scope before `--detailed`

### Automation
- Add quality checks to CI/CD pipeline
- Set thresholds (e.g., fail if > 5% incomplete chunks)
- Generate HTML reports for each release

### Best Practices
- Always run quality check after chunking
- Keep HTML reports for historical comparison
- Spot check random samples regularly
- Compare metrics across repositories

## Output Examples

### Terminal Output
```
📊 Basic Statistics:
   Total chunks:        2,046
   Avg tokens/chunk:    174.1
   Token range:         2 - 680

🔍 Chunk Types:
   method         :  1287 ( 62.9%)
   type           :   305 ( 14.9%)
   function       :   227 ( 11.1%)

✨ LLM Enrichment:
   With summary:        2,046 (100.0%)
   With keywords:       2,046 (100.0%)

✅ No quality issues detected!
```

### HTML Report
Opens in browser with:
- Dashboard showing key metrics
- Filterable list of chunks
- Code with syntax highlighting
- LLM summaries and keywords inline
- Visual indicators for quality issues

### Detailed View
```
================================================================================
CHUNK #1
================================================================================

📄 File: cmd/ssim/export.go
   Lines: 77-126
   Tokens: 377
   Kind: function
   Symbol: export

📝 Code Text (1348 chars):
────────────────────────────────────────────────────────────────────────────────
func export(ctx context.Context, cfg *exportCfg) error {
    log.Debug(ctx, "Starting export process")
    ...
}
────────────────────────────────────────────────────────────────────────────────

✨ LLM Summary:
   Exports data for specified airlines and schemas, handling database connections and errors

🏷️  Keywords:
   airline filtering, atomic counters, data export, database connection, error handling
```

## Troubleshooting

### "Failed to parse line X"
- JSONL file might be corrupted
- Check line X in the file
- Re-generate chunks if needed

### "No chunks found"
- Check file path
- Verify JSONL format
- Use `--sample 1` to test

### HTML report doesn't open
- Use `file://` URL shown in output
- Check browser security settings
- Try different browser

### Filters return no results
- Verify filter values match data
- Use `--kind function` not `--kind functions`
- File filter is case-insensitive partial match

## See Also

- `CHUNK_SCHEMA_V3.md` - Complete chunk format documentation
- `NESTED_FIXES_REPORT.md` - Details on quality fixes applied
- `nwe_chunks_v3.py` - Main chunking script

