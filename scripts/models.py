from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    organization = Column(String)
    document_type = Column(String)
    category = Column(String)
    clientele = Column(String)
    knowledge_type = Column(String)
    language = Column(String)
    filepath = Column(String, unique=True)
    content = relationship("Content", uselist=False, back_populates="document")

class Content(Base):
    __tablename__ = 'content'

    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, ForeignKey('documents.id'))
    content = Column(String)
    document = relationship("Document", back_populates="content")

class DatabaseUpdate(Base):
    __tablename__ = 'database_updates'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    new_documents = Column(Integer)
    updated_documents = Column(Integer)