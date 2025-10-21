#!/usr/bin/env python3
"""
Test SQL CTE (Common Table Expression) inclusion in chunks.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_chunks_v3 import chunk_special_file


class TestSQLCTEInclusion:
    """Test SQL chunking with CTE inclusion."""
    
    def test_sql_with_cte_chunking(self):
        """Test chunking SQL with CTEs."""
        sql_with_cte = '''
-- User analytics query with CTEs
WITH user_stats AS (
    SELECT 
        user_id,
        COUNT(*) as total_orders,
        SUM(amount) as total_spent,
        AVG(amount) as avg_order_value
    FROM orders 
    WHERE created_at >= '2024-01-01'
    GROUP BY user_id
),
active_users AS (
    SELECT 
        u.id,
        u.name,
        u.email,
        us.total_orders,
        us.total_spent,
        us.avg_order_value
    FROM users u
    INNER JOIN user_stats us ON u.id = us.user_id
    WHERE u.status = 'active'
)
SELECT 
    au.name,
    au.email,
    au.total_orders,
    au.total_spent,
    au.avg_order_value,
    CASE 
        WHEN au.total_spent > 1000 THEN 'high_value'
        WHEN au.total_spent > 500 THEN 'medium_value'
        ELSE 'low_value'
    END as customer_tier
FROM active_users au
ORDER BY au.total_spent DESC;

-- Another query with different CTEs
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', created_at) as month,
        SUM(amount) as monthly_revenue
    FROM orders
    GROUP BY DATE_TRUNC('month', created_at)
),
yearly_totals AS (
    SELECT 
        EXTRACT(YEAR FROM month) as year,
        SUM(monthly_revenue) as yearly_revenue
    FROM monthly_sales
    GROUP BY EXTRACT(YEAR FROM month)
)
SELECT 
    year,
    yearly_revenue,
    LAG(yearly_revenue) OVER (ORDER BY year) as prev_year_revenue,
    yearly_revenue - LAG(yearly_revenue) OVER (ORDER BY year) as revenue_growth
FROM yearly_totals
ORDER BY year;
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write(sql_with_cte)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_special_file(
                sql_with_cte, temp_path, "test_repo", "sha123",
                "sql", "test.sql", None
            )
            
            # Should create chunks for different SQL statements
            assert len(chunks) > 1
            
            # Check that CTEs are included with their referencing queries
            cte_chunks = [chunk for chunk in chunks if "WITH" in chunk.text.upper()]
            assert len(cte_chunks) > 0
            
            # Check chunk structure
            for chunk in chunks:
                assert chunk.language == "sql"
                assert chunk.repo == "test_repo"
                assert "sql" in chunk.summary_1l.lower() or "query" in chunk.summary_1l.lower()
                
        finally:
            temp_path.unlink()
    
    def test_sql_cte_referenced_in_select(self):
        """Test that CTEs are included when referenced in SELECT."""
        sql_with_referenced_cte = '''
-- Complex query with multiple CTEs
WITH 
    user_metrics AS (
        SELECT 
            user_id,
            COUNT(DISTINCT order_id) as order_count,
            SUM(amount) as total_amount,
            MIN(created_at) as first_order_date,
            MAX(created_at) as last_order_date
        FROM orders
        GROUP BY user_id
    ),
    product_popularity AS (
        SELECT 
            product_id,
            COUNT(*) as purchase_count,
            SUM(quantity) as total_quantity
        FROM order_items
        GROUP BY product_id
    )
SELECT 
    u.name as user_name,
    u.email,
    um.order_count,
    um.total_amount,
    um.first_order_date,
    um.last_order_date,
    pp.product_name,
    pp.purchase_count
FROM users u
INNER JOIN user_metrics um ON u.id = um.user_id
INNER JOIN (
    SELECT 
        p.id as product_id,
        p.name as product_name,
        pp.purchase_count
    FROM products p
    INNER JOIN product_popularity pp ON p.id = pp.product_id
) pp ON 1=1  -- This is a simplified join for the example
WHERE um.total_amount > 100
ORDER BY um.total_amount DESC;
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write(sql_with_referenced_cte)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_special_file(
                sql_with_referenced_cte, temp_path, "test_repo", "sha123",
                "sql", "test.sql", None
            )
            
            # Should create chunks that include CTEs with their references
            assert len(chunks) > 0
            
            # Find chunks that contain CTEs
            cte_chunks = []
            for chunk in chunks:
                if "WITH" in chunk.text.upper() and "user_metrics" in chunk.text:
                    cte_chunks.append(chunk)
            
            # Should have chunks that include the CTE definitions
            assert len(cte_chunks) > 0
            
        finally:
            temp_path.unlink()
    
    def test_sql_multiple_statements(self):
        """Test chunking multiple SQL statements."""
        multiple_sql_statements = '''
-- Create table statement
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO users (name, email) VALUES 
    ('John Doe', 'john@example.com'),
    ('Jane Smith', 'jane@example.com'),
    ('Bob Johnson', 'bob@example.com');

-- Select with CTE
WITH user_count AS (
    SELECT COUNT(*) as total_users FROM users
)
SELECT 
    total_users,
    'Total users in system' as description
FROM user_count;

-- Update statement
UPDATE users 
SET name = 'John Updated'
WHERE email = 'john@example.com';

-- Delete statement
DELETE FROM users 
WHERE email = 'bob@example.com';
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write(multiple_sql_statements)
            temp_path = Path(f.name)
        
        try:
            chunks = chunk_special_file(
                multiple_sql_statements, temp_path, "test_repo", "sha123",
                "sql", "test.sql", None
            )
            
            # Should create separate chunks for different statement types
            assert len(chunks) > 1
            
            # Check that different statement types are chunked separately
            create_chunks = [chunk for chunk in chunks if "CREATE TABLE" in chunk.text.upper()]
            insert_chunks = [chunk for chunk in chunks if "INSERT INTO" in chunk.text.upper()]
            select_chunks = [chunk for chunk in chunks if "SELECT" in chunk.text.upper()]
            update_chunks = [chunk for chunk in chunks if "UPDATE" in chunk.text.upper()]
            delete_chunks = [chunk for chunk in chunks if "DELETE FROM" in chunk.text.upper()]
            
            # Should have chunks for different statement types
            assert len(create_chunks) > 0
            assert len(insert_chunks) > 0
            assert len(select_chunks) > 0
            assert len(update_chunks) > 0
            assert len(delete_chunks) > 0
            
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__])
