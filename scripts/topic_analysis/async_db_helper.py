"""Async database helper with improved task management."""
from typing import List, Optional, Tuple, Dict, Any
import asyncio
from loguru import logger
import asyncpg
from asyncpg.pool import Pool
from contextlib import asynccontextmanager
import functools

class AsyncDatabaseHelper:
    """Helper class for async database operations using single connection per request."""
    
    def __init__(self, db_url: str):
        """Initialize async database connection."""
        try:
            # Parse database URL
            if db_url.startswith('postgresql://'):
                self.db_url = db_url
            elif db_url.startswith('postgresql+asyncpg://'):
                self.db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')
            else:
                raise ValueError("Invalid database URL format")
            
            # Initialize state
            self._connections: Dict[int, asyncpg.Connection] = {}
            self._lock = asyncio.Lock()
            
            logger.info("Async database helper initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database helper: {e}")
            raise

    @asynccontextmanager
    async def connection(self):
        """Get a database connection for current task."""
        task_id = id(asyncio.current_task())
        connection = None
        
        try:
            async with self._lock:
                if task_id in self._connections:
                    connection = self._connections[task_id]
                else:
                    connection = await asyncpg.connect(self.db_url)
                    self._connections[task_id] = connection
            
            yield connection
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                try:
                    async with self._lock:
                        if task_id in self._connections:
                            await connection.close()
                            del self._connections[task_id]
                except Exception as e:
                    logger.warning(f"Error cleaning up connection: {e}")

    async def fetch_all_documents(self) -> List[Tuple[int, str]]:
        """Fetch all documents with content."""
        try:
            async with self.connection() as conn:
                # Query to get documents with content
                query = """
                    SELECT d.id, c.content
                    FROM documents d
                    JOIN content c ON d.id = c.doc_id
                    WHERE c.content IS NOT NULL
                    AND length(c.content) > 0
                """
                
                rows = await conn.fetch(query)
                
                if not rows:
                    logger.warning("No documents found in database")
                    return []
                
                # Process results
                docs = [
                    (row['id'], row['content'])
                    for row in rows
                    if row['content'] and isinstance(row['content'], str) and row['content'].strip()
                ]
                
                logger.info(f"Successfully fetched {len(docs)} documents")
                return docs
                
        except Exception as e:
            logger.error(f"Error fetching documents: {e}")
            return []

    async def store_topics(
        self,
        topics: List[Tuple[str, List[str], float]],
        doc_ids: List[int]
    ):
        """Store topic analysis results."""
        try:
            async with self.connection() as conn:
                # Start transaction
                async with conn.transaction():
                    # Clear existing topics
                    await conn.execute(
                        "DELETE FROM document_topics WHERE doc_id = ANY($1::int[])",
                        doc_ids
                    )
                    
                    # Store new topics
                    for label, words, score in topics:
                        # Insert topic
                        topic_id = await conn.fetchval(
                            """
                            INSERT INTO topics (label, words, coherence_score)
                            VALUES ($1, $2, $3)
                            RETURNING id
                            """,
                            label,
                            ','.join(words),
                            score
                        )
                        
                        # Create document associations
                        await conn.executemany(
                            """
                            INSERT INTO document_topics (doc_id, topic_id)
                            VALUES ($1, $2)
                            """,
                            [(doc_id, topic_id) for doc_id in doc_ids]
                        )
            
            logger.info(f"Stored {len(topics)} topics for {len(doc_ids)} documents")
            
        except Exception as e:
            logger.error(f"Error storing topics: {e}")
            raise

    async def cleanup(self):
        """Clean up database resources."""
        try:
            async with self._lock:
                for conn in self._connections.values():
                    await conn.close()
                self._connections.clear()
                
            logger.info("Database resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()