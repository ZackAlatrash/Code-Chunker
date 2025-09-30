# Search v3 Features & Enhancements

## Overview
`search_into_json_v3.py` is an enhanced search script that leverages the new v3 stack:
- **code_chunks_v3**: Symbol-aware chunking with enhanced metadata
- **repo_router_v2**: LLM-enriched repository router
- **repo_guide_v1**: LLM-generated repository guides

## Key Enhancements Over Original `search_into_json.py`

### 1. Symbol-Aware Search
**New Fields from code_chunks_v3:**
- `primary_symbol`: Main function/class name in chunk
- `primary_kind`: Type of symbol (function, class, etc.)  
- `primary_span`: Line range where primary symbol is defined
- `def_symbols`: All symbols defined in this chunk
- `symbols`: All symbols (definitions + references) in chunk
- `doc_head`: Documentation/docstring content

**Enhanced BM25 Query:**
```python
"fields": [
    "text^3",              # Code content
    "primary_symbol^6",    # Main symbol (highest weight)
    "def_symbols^5",       # Defined symbols
    "symbols^4",           # All symbols
    "doc_head^3",          # Documentation
    "rel_path^2",          # File path
    "primary_kind^2"       # Symbol type
]
```

### 2. Enhanced Router (v2)
**New Router Fields:**
- `short_title`: Descriptive repository title
- `summary`: LLM-generated 120-200 word summary
- `key_symbols`: Important function/class names
- `keywords`: Curated search terms
- `synonyms`: Alternative terminology
- `tech_stack`: Technology stack information
- `modules`: Key components/modules
- `important_files`: Most critical files
- `sample_queries`: Example queries

**Enhanced Router Query:**
```python
"fields": [
    "short_title^4",        # Enhanced title
    "summary^3",            # LLM-generated summary
    "key_symbols^5",        # Important symbols (highest weight)
    "keywords^4",           # Curated keywords
    "synonyms^3",           # Alternative terms
    "tech_stack^2",         # Technology stack
    "modules^2",            # Key modules
    "important_files^1.5",  # Important files
    "sample_queries^2"      # Example queries
]
```

### 3. Enhanced Search Results Display
**New Information Shown:**
- Symbol type and name: `(function: getUserById)`
- Defined symbols: `[defines: getUserById, validateUser, hashPassword]`
- Documentation preview from `doc_head`
- Better formatted output with symbol context

### 4. Enhanced LLM Bundle (v1.3)
**New Fields in JSON Output:**
```json
{
  "version": "1.3",
  "sources": [
    {
      "idx": 1,
      "primary_symbol": "authenticate",
      "primary_kind": "function", 
      "primary_span": [15, 28],
      "def_symbols": ["authenticate", "validateToken"],
      "symbols": ["authenticate", "jwt", "decode", "User", "find"],
      "doc_head": "Authenticates user with JWT token..."
    }
  ]
}
```

## Usage Examples

### Basic Search
```bash
python scripts/search_into_json_v3.py "JWT authentication"
```

### Target Specific Repository  
```bash
python scripts/search_into_json_v3.py "user validation" --explicit-repo myapp
```

### Custom Output
```bash
python scripts/search_into_json_v3.py "database queries" --out my_search.json --final-k 15
```

## Benefits

1. **Better Symbol Matching**: Searches can now match specific function/class names with high precision
2. **Context-Aware Results**: Results include symbol definitions and documentation context
3. **Enhanced Router**: More accurate repository routing using LLM-generated metadata
4. **Richer LLM Bundles**: Output includes comprehensive symbol information for better LLM reasoning
5. **Improved Relevance**: Symbol-weighted scoring provides more relevant code search results

## Migration from v1

Simply replace:
```bash
python scripts/search_into_json.py "query"
```

With:
```bash  
python scripts/search_into_json_v3.py "query"
```

The script maintains backward compatibility with all existing arguments while providing enhanced results through the new v3 stack.
