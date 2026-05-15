from sqlalchemy import create_engine, Column, String, Integer, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()
engine = create_engine("sqlite:///./data/rag_metadata.db", echo=False)
Session = sessionmaker(bind=engine)

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    title = Column(String)
    file_path = Column(String)
    file_type = Column(String)
    language = Column(String)
    has_diacritics = Column(Boolean)
    total_chars = Column(Integer)
    num_chunks = Column(Integer)
    strategy = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(String, primary_key=True)
    document_id = Column(String)
    chunk_index = Column(Integer)
    text = Column(Text)
    text_normalized = Column(Text)
    heading = Column(String)
    language = Column(String)
    has_diacritics = Column(Boolean)
    char_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

def save_document(data):
    with Session() as s:
        s.merge(Document(**data))
        s.commit()

def save_chunks(chunks):
    with Session() as s:
        for c in chunks:
            s.merge(Chunk(**c))
        s.commit()

def get_document(doc_id):
    with Session() as s:
        doc = s.get(Document, doc_id)
        if not doc:
            return None
        return {
            "id": doc.id,
            "title": doc.title,
            "file_path": doc.file_path,
            "file_type": doc.file_type,
            "language": doc.language,
            "has_diacritics": doc.has_diacritics,
            "total_chars": doc.total_chars,
            "num_chunks": doc.num_chunks,
            "strategy": doc.strategy,
            "created_at": str(doc.created_at),
        }

def list_documents():
    with Session() as s:
        docs = s.query(Document).all()
        return [get_document(d.id) for d in docs]

def get_chunks(doc_id):
    with Session() as s:
        chunks = s.query(Chunk).filter(Chunk.document_id == doc_id).order_by(Chunk.chunk_index).all()
        return [
            {
                "id": c.id,
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "heading": c.heading,
                "language": c.language,
                "has_diacritics": c.has_diacritics,
                "char_count": c.char_count,
            }
            for c in chunks
        ]

def delete_document(doc_id):
    with Session() as s:
        chunks = s.query(Chunk).filter(Chunk.document_id == doc_id).all()
        for c in chunks:
            s.delete(c)
        doc = s.get(Document, doc_id)
        if doc:
            s.delete(doc)
        s.commit()
        return True

def get_stats():
    with Session() as s:
        return {
            "total_documents": s.query(Document).count(),
            "total_chunks": s.query(Chunk).count(),
            "arabic_documents": s.query(Document).filter(Document.language == "arabic").count(),
        }