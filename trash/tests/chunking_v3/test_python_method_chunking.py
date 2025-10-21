#!/usr/bin/env python3
"""
Test Python method chunking functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import (
    chunk_code_file, compute_chunk_id, generate_summary_1l, 
    generate_qa_terms, extract_imports_from_code
)


class TestPythonMethodChunking:
    """Test Python method chunking with AST analysis."""
    
    def test_simple_function_chunking(self):
        """Test chunking a simple Python function."""
        code = '''
import os
from typing import List

class UserService:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_user_by_id(self, user_id: int) -> dict:
        """Get user by ID from database."""
        query = "SELECT * FROM users WHERE id = %s"
        result = self.db.execute(query, (user_id,))
        return result.fetchone()
    
    def create_user(self, user_data: dict) -> int:
        """Create a new user."""
        query = "INSERT INTO users (name, email) VALUES (%s, %s)"
        return self.db.execute(query, (user_data['name'], user_data['email']))
'''
        
        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_code_file(temp_path, "test_repo", "gpt-4", 150, 380, None)
            
            # Should create multiple chunks
            assert len(chunks) > 0
            
            # Check chunk structure
            for chunk in chunks:
                assert chunk.repo == "test_repo"
                assert chunk.language == "python"
                assert chunk.start_line >= 1
                assert chunk.end_line >= chunk.start_line
                assert chunk.chunk_id is not None
                assert chunk.text is not None
                assert chunk.v == 3
                
        finally:
            temp_path.unlink()
    
    def test_chunk_id_deterministic(self):
        """Test that chunk IDs are deterministic."""
        chunk_id1 = compute_chunk_id("repo", "/path/file.py", "sha123", 10, 20, "function_name")
        chunk_id2 = compute_chunk_id("repo", "/path/file.py", "sha123", 10, 20, "function_name")
        
        assert chunk_id1 == chunk_id2
        
        # Different inputs should produce different IDs
        chunk_id3 = compute_chunk_id("repo", "/path/file.py", "sha123", 10, 21, "function_name")
        assert chunk_id1 != chunk_id3
    
    def test_summary_generation(self):
        """Test one-line summary generation."""
        code = '''
def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user with username and password."""
    # Implementation here
    pass
'''
        
        summary = generate_summary_1l(code, "python", "authenticate_user")
        assert "Python function" in summary
        assert "authenticate_user" in summary
        assert "lines of code" in summary
    
    def test_qa_terms_generation(self):
        """Test QA terms generation."""
        code = '''
from fastapi import FastAPI, HTTPException
from typing import List

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if user_id < 1:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    return {"user_id": user_id, "name": "John Doe"}
'''
        
        qa_terms = generate_qa_terms(code, "python")
        terms_list = qa_terms.split(', ')
        
        # Should contain framework and HTTP terms
        assert any('FastAPI' in term for term in terms_list)
        assert any('GET' in term for term in terms_list)
        assert any('400' in term for term in terms_list)
    
    def test_import_extraction(self):
        """Test import statement extraction."""
        code = '''
import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException
import requests as req
'''
        
        imports = extract_imports_from_code(code, "python")
        
        assert len(imports) >= 4
        assert any('import os' in imp for imp in imports)
        assert any('from typing import' in imp for imp in imports)
        assert any('from fastapi import' in imp for imp in imports)
    
    def test_neighbors_wiring(self):
        """Test that neighbors are properly wired between chunks."""
        code = '''
def function1():
    pass

def function2():
    pass

def function3():
    pass
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_code_file(temp_path, "test_repo", "gpt-4", 50, 100, None)
            
            if len(chunks) > 1:
                # Check neighbors
                for i, chunk in enumerate(chunks):
                    if i > 0:
                        assert chunk.neighbors.prev == chunks[i-1].chunk_id
                    if i < len(chunks) - 1:
                        assert chunk.neighbors.next == chunks[i+1].chunk_id
                        
        finally:
            temp_path.unlink()
    
    def test_token_budget_enforcement(self):
        """Test that token budgets are respected."""
        # Create a large function that should be split
        large_code = '''
def large_function():
    """This is a very large function with many lines."""
    var1 = "This is a very long string that takes up many tokens"
    var2 = "Another long string for token counting"
    var3 = "Yet another long string to exceed token limits"
    var4 = "More content to make this function large"
    var5 = "Even more content to ensure token limit is exceeded"
    var6 = "Additional content for token counting"
    var7 = "More strings to increase token count"
    var8 = "Final string to ensure we exceed the limit"
    return var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_code)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_code_file(temp_path, "test_repo", "gpt-4", 50, 100, None)
            
            # Check token counts
            for chunk in chunks:
                assert chunk.token_counts.total <= 100  # max_total
                
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__])
