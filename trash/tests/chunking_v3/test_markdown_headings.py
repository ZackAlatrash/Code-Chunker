#!/usr/bin/env python3
"""
Test Markdown chunking by headings with fenced code block preservation.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_special_file


class TestMarkdownHeadings:
    """Test Markdown chunking by heading levels."""
    
    def test_markdown_heading_chunking(self):
        """Test chunking Markdown by heading levels."""
        markdown_content = '''
# API Documentation

This document describes the REST API endpoints for our application.

## Authentication

All API requests require authentication using a Bearer token.

### Getting a Token

To get an authentication token, send a POST request to `/auth/login`:

```json
{
  "username": "your_username",
  "password": "your_password"
}
```

The response will include a token:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600
}
```

## User Management

The user management endpoints allow you to create, read, update, and delete users.

### Create User

Create a new user with a POST request to `/users`:

```python
import requests

response = requests.post('https://api.example.com/users', 
    json={
        'name': 'John Doe',
        'email': 'john@example.com'
    },
    headers={'Authorization': 'Bearer your_token'}
)
```

### Get User

Retrieve user information with a GET request:

```bash
curl -H "Authorization: Bearer your_token" \
     https://api.example.com/users/123
```

## Error Handling

The API returns standard HTTP status codes:

- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

### Error Response Format

All error responses follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "email",
      "reason": "Invalid email format"
    }
  }
}
```
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(markdown_content)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_special_file(
                markdown_content, temp_path, "test_repo", "sha123",
                "markdown", "test.md", None
            )
            
            # Should create multiple chunks for different sections
            assert len(chunks) > 1
            
            # Check chunk structure
            for chunk in chunks:
                assert chunk.language == "markdown"
                assert chunk.repo == "test_repo"
                assert "markdown" in chunk.summary_1l.lower() or "documentation" in chunk.summary_1l.lower()
                
        finally:
            temp_path.unlink()
    
    def test_fenced_code_blocks_preserved(self):
        """Test that fenced code blocks are preserved intact."""
        markdown_with_code = '''
# Code Examples

## Python Example

Here's a Python function:

```python
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Usage
result = calculate_fibonacci(10)
print(f"Fibonacci(10) = {result}")
```

## JavaScript Example

And here's the JavaScript equivalent:

```javascript
function calculateFibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return calculateFibonacci(n - 1) + calculateFibonacci(n - 2);
}

// Usage
const result = calculateFibonacci(10);
console.log(`Fibonacci(10) = ${result}`);
```

## SQL Example

Database query example:

```sql
SELECT 
    u.name,
    u.email,
    COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name, u.email
HAVING COUNT(o.id) > 5
ORDER BY order_count DESC;
```
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(markdown_with_code)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_special_file(
                markdown_with_code, temp_path, "test_repo", "sha123",
                "markdown", "test.md", None
            )
            
            # Should create chunks that preserve code blocks
            assert len(chunks) > 0
            
            # Check that code blocks are preserved
            python_chunks = [chunk for chunk in chunks if "```python" in chunk.text]
            js_chunks = [chunk for chunk in chunks if "```javascript" in chunk.text]
            sql_chunks = [chunk for chunk in chunks if "```sql" in chunk.text]
            
            # Should have chunks with preserved code blocks
            assert len(python_chunks) > 0
            assert len(js_chunks) > 0
            assert len(sql_chunks) > 0
            
            # Check that code blocks are complete
            for chunk in python_chunks:
                assert "```python" in chunk.text
                assert "```" in chunk.text[chunk.text.find("```python") + 9:]  # Closing ```
                
        finally:
            temp_path.unlink()
    
    def test_nested_headings_chunking(self):
        """Test chunking with nested heading levels."""
        nested_markdown = '''
# Main Topic

Introduction to the main topic.

## Subtopic A

Content for subtopic A.

### Sub-subtopic A1

Detailed content for A1.

### Sub-subtopic A2

Detailed content for A2.

## Subtopic B

Content for subtopic B.

### Sub-subtopic B1

Detailed content for B1.

#### Sub-sub-subtopic B1a

Very detailed content for B1a.

#### Sub-sub-subtopic B1b

Very detailed content for B1b.

### Sub-subtopic B2

Detailed content for B2.

## Subtopic C

Content for subtopic C.
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(nested_markdown)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_special_file(
                nested_markdown, temp_path, "test_repo", "sha123",
                "markdown", "test.md", None
            )
            
            # Should create chunks for different heading levels
            assert len(chunks) > 1
            
            # Check that chunks contain appropriate heading content
            for chunk in chunks:
                text = chunk.text
                # Each chunk should contain at least one heading
                assert any(text.startswith(f"{'#' * i} ") for i in range(1, 7))
                
        finally:
            temp_path.unlink()
    
    def test_markdown_with_mixed_content(self):
        """Test chunking Markdown with mixed content types."""
        mixed_markdown = '''
# Configuration Guide

This guide explains how to configure the application.

## Environment Variables

Set the following environment variables:

```bash
export DATABASE_URL="postgresql://user:pass@localhost/db"
export REDIS_URL="redis://localhost:6379"
export SECRET_KEY="your-secret-key"
```

## Configuration File

Create a `config.yaml` file:

```yaml
database:
  host: localhost
  port: 5432
  name: myapp
  user: myuser
  password: mypass

redis:
  host: localhost
  port: 6379
  db: 0

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## API Configuration

Configure the API settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `api.host` | `0.0.0.0` | API server host |
| `api.port` | `8000` | API server port |
| `api.debug` | `false` | Enable debug mode |

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check database URL
   - Verify database is running
   - Check network connectivity

2. **Redis Connection Error**
   - Check Redis URL
   - Verify Redis is running
   - Check firewall settings

### Logs

Check application logs for errors:

```bash
tail -f /var/log/myapp/app.log
```
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(mixed_markdown)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_special_file(
                mixed_markdown, temp_path, "test_repo", "sha123",
                "markdown", "test.md", None
            )
            
            # Should create chunks for different sections
            assert len(chunks) > 1
            
            # Check that different content types are preserved
            yaml_chunks = [chunk for chunk in chunks if "```yaml" in chunk.text]
            bash_chunks = [chunk for chunk in chunks if "```bash" in chunk.text]
            table_chunks = [chunk for chunk in chunks if "|" in chunk.text and "---" in chunk.text]
            
            # Should have chunks with different content types
            assert len(yaml_chunks) > 0
            assert len(bash_chunks) > 0
            assert len(table_chunks) > 0
            
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__])
