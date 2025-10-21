#!/usr/bin/env python3
"""
Test JavaScript function splitting into multiple chunks.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_code_file


class TestJavaScriptSplitBlocks:
    """Test JavaScript function splitting when oversized."""
    
    def test_oversized_function_splitting(self):
        """Test that oversized functions are split into multiple chunks."""
        large_js_code = '''
import React, { useState, useEffect } from 'react';
import axios from 'axios';

class UserComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            users: [],
            loading: false,
            error: null
        };
    }
    
    async fetchUsers() {
        this.setState({ loading: true, error: null });
        try {
            const response = await axios.get('/api/users');
            this.setState({ 
                users: response.data,
                loading: false 
            });
        } catch (error) {
            this.setState({ 
                error: error.message,
                loading: false 
            });
        }
    }
    
    async createUser(userData) {
        this.setState({ loading: true });
        try {
            const response = await axios.post('/api/users', userData);
            this.setState(prevState => ({
                users: [...prevState.users, response.data],
                loading: false
            }));
            return response.data;
        } catch (error) {
            this.setState({ 
                error: error.message,
                loading: false 
            });
            throw error;
        }
    }
    
    async updateUser(userId, userData) {
        this.setState({ loading: true });
        try {
            const response = await axios.put(`/api/users/${userId}`, userData);
            this.setState(prevState => ({
                users: prevState.users.map(user => 
                    user.id === userId ? response.data : user
                ),
                loading: false
            }));
            return response.data;
        } catch (error) {
            this.setState({ 
                error: error.message,
                loading: false 
            });
            throw error;
        }
    }
    
    async deleteUser(userId) {
        this.setState({ loading: true });
        try {
            await axios.delete(`/api/users/${userId}`);
            this.setState(prevState => ({
                users: prevState.users.filter(user => user.id !== userId),
                loading: false
            }));
        } catch (error) {
            this.setState({ 
                error: error.message,
                loading: false 
            });
            throw error;
        }
    }
    
    render() {
        const { users, loading, error } = this.state;
        
        if (loading) return <div>Loading...</div>;
        if (error) return <div>Error: {error}</div>;
        
        return (
            <div>
                <h1>Users</h1>
                <button onClick={() => this.fetchUsers()}>Refresh</button>
                <ul>
                    {users.map(user => (
                        <li key={user.id}>
                            {user.name} - {user.email}
                            <button onClick={() => this.deleteUser(user.id)}>Delete</button>
                        </li>
                    ))}
                </ul>
            </div>
        );
    }
}

export default UserComponent;
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(large_js_code)
            temp_path = Path(f.name)
        
        try:
            # Use small token limits to force splitting
            chunks = chunk_code_file(temp_path, "test_repo", "gpt-4", 50, 100, None)
            
            # Should create multiple chunks due to large function
            assert len(chunks) > 1
            
            # Check that chunks are properly structured
            for chunk in chunks:
                assert chunk.language == "javascript"
                assert chunk.token_counts.total <= 100  # max_total
                assert chunk.text is not None
                assert len(chunk.text) > 0
                
        finally:
            temp_path.unlink()
    
    def test_header_context_repetition(self):
        """Test that header context is repeated for split chunks."""
        js_code_with_imports = '''
import React from 'react';
import { useState, useEffect } from 'react';
import axios from 'axios';

function LargeComponent() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    
    useEffect(() => {
        fetchData();
    }, []);
    
    const fetchData = async () => {
        setLoading(true);
        try {
            const response = await axios.get('/api/data');
            setData(response.data);
        } catch (error) {
            console.error('Error fetching data:', error);
        } finally {
            setLoading(false);
        }
    };
    
    const processData = (rawData) => {
        // Large processing logic here
        const processed = rawData.map(item => ({
            id: item.id,
            name: item.name.toUpperCase(),
            category: item.category,
            timestamp: new Date(item.created_at),
            status: item.active ? 'active' : 'inactive'
        }));
        
        return processed.sort((a, b) => a.timestamp - b.timestamp);
    };
    
    const handleSubmit = (formData) => {
        // Large submit logic
        const validation = validateFormData(formData);
        if (!validation.isValid) {
            setErrors(validation.errors);
            return;
        }
        
        const payload = transformFormData(formData);
        submitToAPI(payload);
    };
    
    return (
        <div>
            {loading ? <div>Loading...</div> : <div>{JSON.stringify(data)}</div>}
        </div>
    );
}
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(js_code_with_imports)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_code_file(temp_path, "test_repo", "gpt-4", 30, 60, None)
            
            # Check that imports are preserved in chunks
            for chunk in chunks:
                # Each chunk should have access to the imports
                assert chunk.imports_used is not None
                
        finally:
            temp_path.unlink()
    
    def test_arrow_function_chunking(self):
        """Test chunking of arrow functions."""
        arrow_function_code = '''
import React from 'react';

const MyComponent = () => {
    const handleClick = () => {
        console.log('Button clicked');
    };
    
    const handleSubmit = (event) => {
        event.preventDefault();
        const formData = new FormData(event.target);
        console.log('Form submitted:', formData);
    };
    
    return (
        <div>
            <button onClick={handleClick}>Click me</button>
            <form onSubmit={handleSubmit}>
                <input name="username" type="text" />
                <button type="submit">Submit</button>
            </form>
        </div>
    );
};

export default MyComponent;
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(arrow_function_code)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_code_file(temp_path, "test_repo", "gpt-4", 50, 100, None)
            
            assert len(chunks) > 0
            
            for chunk in chunks:
                assert chunk.language == "javascript"
                assert "arrow" in chunk.summary_1l.lower() or "function" in chunk.summary_1l.lower()
                
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__])
