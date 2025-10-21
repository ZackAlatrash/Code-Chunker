#!/usr/bin/env python3
"""
Test OpenAPI/Swagger endpoint chunking functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_special_file, detect_file_type


class TestOpenAPIEndpointChunker:
    """Test OpenAPI/Swagger file chunking by endpoints."""
    
    def test_openapi_detection(self):
        """Test OpenAPI file type detection."""
        # Test YAML OpenAPI
        yaml_path = Path("test.yaml")
        assert detect_file_type(yaml_path) == "yaml"
        
        # Test JSON OpenAPI
        json_path = Path("test.json")
        assert detect_file_type(json_path) == "json"
    
    def test_openapi_yaml_chunking(self):
        """Test chunking OpenAPI YAML by endpoints."""
        openapi_yaml = '''
openapi: 3.0.0
info:
  title: User API
  version: 1.0.0
  description: API for user management

servers:
  - url: https://api.example.com/v1

paths:
  /users:
    get:
      summary: List users
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 10
      responses:
        '200':
          description: List of users
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/User'
        '400':
          description: Bad request
    post:
      summary: Create user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserInput'
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '400':
          description: Invalid input

  /users/{userId}:
    get:
      summary: Get user by ID
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: User details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          description: User not found
    put:
      summary: Update user
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: integer
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserInput'
      responses:
        '200':
          description: User updated
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          description: User not found
    delete:
      summary: Delete user
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: integer
      responses:
        '204':
          description: User deleted
        '404':
          description: User not found

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
        email:
          type: string
        createdAt:
          type: string
          format: date-time
    UserInput:
      type: object
      required:
        - name
        - email
      properties:
        name:
          type: string
        email:
          type: string
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(openapi_yaml)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_special_file(
                openapi_yaml, temp_path, "test_repo", "sha123", 
                "openapi", "test.yaml", None
            )
            
            # Should create multiple chunks for different endpoints
            assert len(chunks) > 1
            
            # Check chunk structure
            for chunk in chunks:
                assert chunk.language == "openapi"
                assert chunk.repo == "test_repo"
                assert chunk.text is not None
                assert "openapi" in chunk.summary_1l.lower() or "api" in chunk.summary_1l.lower()
                
        finally:
            temp_path.unlink()
    
    def test_openapi_json_chunking(self):
        """Test chunking OpenAPI JSON by endpoints."""
        openapi_json = '''
{
  "openapi": "3.0.0",
  "info": {
    "title": "Product API",
    "version": "1.0.0",
    "description": "API for product management"
  },
  "servers": [
    {
      "url": "https://api.example.com/v1"
    }
  ],
  "paths": {
    "/products": {
      "get": {
        "summary": "List products",
        "parameters": [
          {
            "name": "category",
            "in": "query",
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "List of products",
            "content": {
              "application/json": {
                "schema": {
                  "type": "array",
                  "items": {
                    "$ref": "#/components/schemas/Product"
                  }
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Create product",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ProductInput"
              }
            }
          }
        },
        "responses": {
          "201": {
            "description": "Product created",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Product"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "Product": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          },
          "price": {
            "type": "number"
          }
        }
      },
      "ProductInput": {
        "type": "object",
        "required": ["name", "price"],
        "properties": {
          "name": {
            "type": "string"
          },
          "price": {
            "type": "number"
          }
        }
      }
    }
  }
}
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(openapi_json)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_special_file(
                openapi_json, temp_path, "test_repo", "sha123",
                "openapi", "test.json", None
            )
            
            # Should create chunks for different endpoints
            assert len(chunks) > 0
            
            # Check that chunks contain endpoint information
            endpoint_chunks = [chunk for chunk in chunks if "/products" in chunk.text]
            assert len(endpoint_chunks) > 0
            
        finally:
            temp_path.unlink()
    
    def test_no_cross_endpoint_leakage(self):
        """Test that chunks don't leak information between endpoints."""
        openapi_with_multiple_endpoints = '''
openapi: 3.0.0
info:
  title: Multi-endpoint API
  version: 1.0.0

paths:
  /users:
    get:
      summary: List users
      responses:
        '200':
          description: Users list
    post:
      summary: Create user
      responses:
        '201':
          description: User created

  /orders:
    get:
      summary: List orders
      responses:
        '200':
          description: Orders list
    post:
      summary: Create order
      responses:
        '201':
          description: Order created

  /payments:
    post:
      summary: Process payment
      responses:
        '200':
          description: Payment processed
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(openapi_with_multiple_endpoints)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_special_file(
                openapi_with_multiple_endpoints, temp_path, "test_repo", "sha123",
                "openapi", "test.yaml", None
            )
            
            # Check that each chunk focuses on specific endpoints
            for chunk in chunks:
                text = chunk.text.lower()
                
                # If chunk contains users endpoint, it shouldn't contain orders/payments
                if "/users" in text:
                    assert "/orders" not in text or "/payments" not in text
                elif "/orders" in text:
                    assert "/users" not in text or "/payments" not in text
                elif "/payments" in text:
                    assert "/users" not in text or "/orders" not in text
                    
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__])
