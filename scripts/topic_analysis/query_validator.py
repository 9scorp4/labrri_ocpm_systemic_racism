"""SQL query validation and safety checks for topic analysis."""
from typing import Dict, Any, Optional, List, Union
from loguru import logger
import re
import sqlalchemy
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError

class TopicQueryValidator:
    """Validates and sanitizes SQL queries for topic analysis pipeline."""
    
    # Patterns for SQL injection detection
    UNSAFE_PATTERNS = [
        r';\s*DROP',
        r';\s*DELETE',
        r';\s*INSERT',
        r';\s*UPDATE',
        r';\s*ALTER',
        r';\s*CREATE',
        r'--',
        r'/\*'
    ]
    
    # Allowed tables for topic analysis
    ALLOWED_TABLES = {
        'documents',
        'content',
        'topics',
        'document_topics'
    }
    
    # Allowed columns per table
    ALLOWED_COLUMNS = {
        'documents': {'id', 'organization', 'document_type', 'category', 
                     'clientele', 'knowledge_type', 'language', 'filepath'},
        'content': {'id', 'doc_id', 'content'},
        'topics': {'id', 'label', 'words', 'coherence_score'},
        'document_topics': {'doc_id', 'topic_id', 'relevance_score'}
    }

    @classmethod
    def validate_query(cls, 
                      query: str, 
                      params: Optional[Dict[str, Any]] = None,
                      allow_write: bool = False) -> bool:
        """
        Validate SQL query for safety and correctness.
        
        Args:
            query: SQL query string
            params: Query parameters
            allow_write: Whether to allow write operations
            
        Returns:
            bool: Whether query is valid
            
        Raises:
            ValueError: If query is invalid
        """
        try:
            # Check for SQL injection patterns
            query_lower = query.lower()
            for pattern in cls.UNSAFE_PATTERNS:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    raise ValueError(f"Potentially unsafe SQL pattern detected: {pattern}")

            # Parse query to extract tables and validate
            tables = cls._extract_tables(query_lower)
            if not tables.issubset(cls.ALLOWED_TABLES):
                invalid_tables = tables - cls.ALLOWED_TABLES
                raise ValueError(f"Invalid tables referenced: {invalid_tables}")

            # Validate columns if we can parse them
            columns = cls._extract_columns(query_lower)
            for table, cols in columns.items():
                if table in cls.ALLOWED_COLUMNS:
                    invalid_cols = cols - cls.ALLOWED_COLUMNS[table]
                    if invalid_cols:
                        raise ValueError(f"Invalid columns for table {table}: {invalid_cols}")

            # Check for write operations if not allowed
            if not allow_write:
                write_operations = {'insert', 'update', 'delete', 'drop', 'alter', 'create'}
                first_word = query_lower.strip().split()[0]
                if first_word in write_operations:
                    raise ValueError(f"Write operation '{first_word}' not allowed")

            # Validate parameters if provided
            if params:
                cls._validate_params(query, params)

            return True

        except ValueError as e:
            logger.error(f"Query validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during query validation: {e}")
            raise ValueError(f"Query validation error: {str(e)}")

    @classmethod
    def prepare_query(cls,
                     query: str,
                     params: Optional[Dict[str, Any]] = None,
                     allow_write: bool = False) -> sqlalchemy.sql.elements.TextClause:
        """
        Prepare and validate SQL query for execution.
        
        Args:
            query: SQL query string
            params: Query parameters
            allow_write: Whether to allow write operations
            
        Returns:
            SQLAlchemy TextClause
            
        Raises:
            ValueError: If query is invalid
        """
        # Validate query first
        cls.validate_query(query, params, allow_write)
        
        # Create SQLAlchemy text object
        try:
            return text(query)
        except SQLAlchemyError as e:
            logger.error(f"Error preparing query: {e}")
            raise ValueError(f"Query preparation failed: {str(e)}")

    @staticmethod
    def _extract_tables(query: str) -> set:
        """Extract table names from query."""
        # Simple regex to find table names after FROM and JOIN
        table_pattern = r'(?:from|join)\s+([a-z_][a-z0-9_]*)'
        return set(re.findall(table_pattern, query))

    @staticmethod
    def _extract_columns(query: str) -> Dict[str, set]:
        """Extract column names and their tables from query."""
        columns = {}
        # Match patterns like "table.column" or "column"
        col_pattern = r'(?:([a-z_][a-z0-9_]*)\.)?([a-z_][a-z0-9_]*)'
        
        # Find all column references
        for table, col in re.findall(col_pattern, query):
            table = table or 'unknown'
            if table not in columns:
                columns[table] = set()
            columns[table].add(col)
            
        return columns

    @staticmethod
    def _validate_params(query: str, params: Dict[str, Any]):
        """Validate query parameters."""
        # Check that all parameters in query have values
        param_pattern = r':([a-z_][a-z0-9_]*)'
        query_params = set(re.findall(param_pattern, query))
        missing_params = query_params - set(params.keys())
        if missing_params:
            raise ValueError(f"Missing parameters: {missing_params}")
            
        # Validate parameter types
        for name, value in params.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(f"Invalid parameter type for {name}: {type(value)}")