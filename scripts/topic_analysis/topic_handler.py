"""Handles topic storage and retrieval operations with async support."""
from loguru import logger
import pandas as pd
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from sqlalchemy import text, select, insert, delete, update
import asyncio
from collections import defaultdict

from scripts.models import Topic, DocumentTopic

class TopicHandler:
    """Handles topic-related database operations with async support."""
    
    def __init__(self, db_helper):
        """Initialize with async database helper."""
        self.db = db_helper
        self._cache_lock = asyncio.Lock()
        self._topic_cache: Dict[int, Dict] = {}
        self._similarity_cache: Dict[str, float] = {}
        
    async def find_similar_topic(
        self,
        words: List[str],
        threshold: float = 0.7,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """Find similar existing topic asynchronously."""
        try:
            # Create cache key
            cache_key = ','.join(sorted(words))
            
            # Check cache if enabled
            if use_cache:
                async with self._cache_lock:
                    if cache_key in self._similarity_cache:
                        similarity = self._similarity_cache[cache_key]
                        if similarity >= threshold:
                            topic_id = next(
                                (tid for tid, score in self._similarity_cache.items() 
                                 if isinstance(tid, int) and score >= threshold),
                                None
                            )
                            if topic_id and topic_id in self._topic_cache:
                                return self._topic_cache[topic_id]
            
            # Query database for topics
            async with self.db.get_session() as session:
                stmt = select(Topic).order_by(Topic.coherence_score.desc())
                result = await session.execute(stmt)
                topics = await result.scalars().all()

                for topic in topics:
                    existing_words = set(topic.words.split(','))
                    new_words = set(words)
                    
                    # Calculate Jaccard similarity
                    intersection = len(existing_words.intersection(new_words))
                    union = len(existing_words.union(new_words))
                    similarity = intersection / union if union > 0 else 0
                    
                    # Cache similarity score
                    async with self._cache_lock:
                        self._similarity_cache[cache_key] = similarity
                        
                    if similarity >= threshold:
                        topic_dict = {
                            'id': topic.id,
                            'label': topic.label,
                            'words': topic.words,
                            'coherence_score': topic.coherence_score
                        }
                        
                        # Cache topic
                        async with self._cache_lock:
                            self._topic_cache[topic.id] = topic_dict
                            
                        return topic_dict
                        
                return None
                
        except Exception as e:
            logger.error(f"Error finding similar topic: {e}")
            return None

    async def add_or_update_topic(
        self,
        label: str,
        words: List[str],
        coherence_score: float
    ) -> Optional[int]:
        """Add new topic or update existing similar one."""
        try:
            # Find similar topic
            similar = await self.find_similar_topic(words)
            
            async with self.db.get_session() as session:
                if similar:
                    if coherence_score > similar['coherence_score']:
                        # Update existing topic
                        stmt = (
                            update(Topic)
                            .where(Topic.id == similar['id'])
                            .values(
                                words=','.join(words),
                                coherence_score=coherence_score
                            )
                            .returning(Topic.id)
                        )
                        result = await session.execute(stmt)
                        await session.commit()
                        
                        # Update cache
                        async with self._cache_lock:
                            self._topic_cache[similar['id']].update({
                                'words': ','.join(words),
                                'coherence_score': coherence_score
                            })
                            
                        return similar['id']
                    return similar['id']
                else:
                    # Insert new topic
                    stmt = (
                        insert(Topic)
                        .values(
                            label=label,
                            words=','.join(words),
                            coherence_score=coherence_score
                        )
                        .returning(Topic.id)
                    )
                    result = await session.execute(stmt)
                    topic_id = (await result.first())[0]
                    await session.commit()
                    
                    # Cache new topic
                    async with self._cache_lock:
                        self._topic_cache[topic_id] = {
                            'id': topic_id,
                            'label': label,
                            'words': ','.join(words),
                            'coherence_score': coherence_score
                        }
                        
                    return topic_id

        except Exception as e:
            logger.error(f"Error adding/updating topic: {e}")
            return None

    async def store_topic_results(
        self,
        results: List[tuple],
        doc_ids: List[int]
    ) -> pd.DataFrame:
        """Store topic analysis results atomically."""
        try:
            # Prepare topics DataFrame
            topics_df = pd.DataFrame([
                {
                    'topic_id': i + 1,
                    'label': label,
                    'words': words,
                    'coherence_score': score
                }
                for i, (label, words, score) in enumerate(results)
            ])

            # Store topics and create document associations
            async with self.db.get_session() as session:
                # Start transaction
                async with session.begin():
                    # Clear existing topics for documents
                    await self.clear_document_topics(doc_ids, session)
                    
                    # Store each topic and its document associations
                    for doc_id in doc_ids:
                        for label, words, score in results:
                            topic_id = await self.add_or_update_topic(
                                label=label,
                                words=words,
                                coherence_score=score
                            )
                            
                            if topic_id:
                                # Create document-topic association
                                stmt = (
                                    insert(DocumentTopic)
                                    .values(doc_id=doc_id, topic_id=topic_id)
                                    .on_conflict_do_nothing()
                                )
                                await session.execute(stmt)
                    
                    # Commit transaction
                    await session.commit()

            return topics_df

        except Exception as e:
            logger.error(f"Error storing topic results: {e}")
            return pd.DataFrame()

    async def clear_document_topics(
        self,
        doc_ids: List[int],
        session = None
    ) -> None:
        """Clear topics for specified documents."""
        try:
            stmt = delete(DocumentTopic).where(DocumentTopic.doc_id.in_(doc_ids))
            
            if session:
                await session.execute(stmt)
            else:
                async with self.db.get_session() as session:
                    await session.execute(stmt)
                    await session.commit()
                    
            logger.info(f"Cleared topics for {len(doc_ids)} documents")
            
        except Exception as e:
            logger.error(f"Error clearing document topics: {e}")
            raise

    async def get_document_topics(
        self,
        doc_id: int,
        use_cache: bool = True
    ) -> List[Dict]:
        """Get topics for a document with caching."""
        try:
            # Check cache if enabled
            if use_cache:
                async with self._cache_lock:
                    cached_topics = [
                        topic for topic in self._topic_cache.values()
                        if doc_id in self._get_topic_docs(topic['id'])
                    ]
                    if cached_topics:
                        return cached_topics
            
            # Query database
            async with self.db.get_session() as session:
                stmt = (
                    select(Topic)
                    .join(DocumentTopic)
                    .where(DocumentTopic.doc_id == doc_id)
                    .order_by(Topic.coherence_score.desc())
                )
                
                result = await session.execute(stmt)
                topics = await result.scalars().all()
                
                topic_list = [{
                    'id': topic.id,
                    'label': topic.label,
                    'words': topic.words.split(',') if topic.words else [],
                    'coherence_score': topic.coherence_score
                } for topic in topics]
                
                # Update cache
                async with self._cache_lock:
                    for topic in topic_list:
                        self._topic_cache[topic['id']] = topic
                        self._add_topic_doc(topic['id'], doc_id)
                
                return topic_list
                
        except Exception as e:
            logger.error(f"Error getting document topics: {e}")
            return []

    def _get_topic_docs(self, topic_id: int) -> Set[int]:
        """Get document IDs associated with a topic from cache."""
        return set(
            doc_id for doc_id, topics in self._doc_topic_cache.items()
            if topic_id in topics
        )

    def _add_topic_doc(self, topic_id: int, doc_id: int):
        """Add document-topic association to cache."""
        if not hasattr(self, '_doc_topic_cache'):
            self._doc_topic_cache = defaultdict(set)
        self._doc_topic_cache[doc_id].add(topic_id)

    async def clear_cache(self):
        """Clear all caches."""
        async with self._cache_lock:
            self._topic_cache.clear()
            self._similarity_cache.clear()
            if hasattr(self, '_doc_topic_cache'):
                self._doc_topic_cache.clear()
            
    async def cleanup(self):
        """Clean up resources."""
        await self.clear_cache()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()