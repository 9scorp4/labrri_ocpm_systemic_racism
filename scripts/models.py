from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    organization = Column(String(255))
    document_type = Column(String(100))
    category = Column(String(100))
    clientele = Column(String(100))
    knowledge_type = Column(String(100))
    language = Column(String(50))
    filepath = Column(String(255), unique=True)
    content = relationship("Content", uselist=False, back_populates="document")
    last_processed = Column(DateTime, nullable=True)

class Content(Base):
    __tablename__ = 'content'

    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, ForeignKey('documents.id'))
    content = Column(Text)
    document = relationship("Document", back_populates="content")

class DatabaseUpdate(Base):
    __tablename__ = 'database_updates'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    new_documents = Column(Integer)
    updated_documents = Column(Integer)

class Topic(Base):
    __tablename__ = 'topics'

    id = Column(Integer, primary_key=True)
    label = Column(String)
    words = Column(String)
    coherence_score = Column(Float)
    documents = relationship('Document', secondary='document_topics', back_populates='topics')

class DocumentTopic(Base):
    __tablename__ = 'document_topics'

    doc_id = Column(Integer, ForeignKey('documents.id'), primary_key=True)
    topic_id = Column(Integer, ForeignKey('topics.id'), primary_key=True)
    relevance_score = Column(Float)

Document.topics = relationship('Topic', secondary='document_topics', back_populates='documents')