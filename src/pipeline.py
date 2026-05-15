import hashlib
import time
from src.loaders.document_loader import load_document
from src.processing.chunker import chunk_document
from src.processing.arabic_processor import strip_diacritics
from src.storage import vector_store, sql_store
from pathlib import Path

def process_file(file_path, force_strategy=None, original_name=None):
    start = time.time()

    doc = load_document(file_path)
    doc_id = hashlib.md5(f"{file_path}{doc['full_text'][:200]}".encode()).hexdigest()[:16]

    if force_strategy:
        from src.processing.chunker import fixed_chunk, dynamic_chunk
        if force_strategy == "fixed":
            raw = fixed_chunk(doc["full_text"])
            chunks = [{"heading": "", "text": c} for c in raw]
            strategy = "fixed"
        else:
            chunks = dynamic_chunk(doc["full_text"])
            strategy = "dynamic"
    else:
        strategy, chunks = chunk_document(
            doc["full_text"],
            num_pages=len(doc["pages"])
        )

    chunk_records = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        normalized = strip_diacritics(text) if doc["has_diacritics"] else text
        chunk_records.append({
            "id": f"{doc_id}_chunk_{i:04d}",
            "document_id": doc_id,
            "chunk_index": i,
            "text": text,
            "text_normalized": normalized,
            "heading": chunk.get("heading", ""),
            "language": doc["language"],
            "has_diacritics": doc["has_diacritics"],
            "char_count": len(text),
        })

    sql_store.save_document({
        "id": doc_id,
        "title": original_name or doc["meta"]["title"] or Path(file_path).stem,
        "file_path": str(file_path),
        "file_type": doc["file_type"],
        "language": doc["language"],
        "has_diacritics": doc["has_diacritics"],
        "total_chars": doc["total_chars"],
        "num_chunks": len(chunk_records),
        "strategy": strategy,
    })

    sql_store.save_chunks(chunk_records)
    vector_store.add_chunks(chunk_records)

    elapsed = round((time.time() - start) * 1000, 2)

    return {
        "document_id": doc_id,
        "title": doc["meta"]["title"],
        "language": doc["language"],
        "has_diacritics": doc["has_diacritics"],
        "strategy": strategy,
        "num_chunks": len(chunk_records),
        "total_chars": doc["total_chars"],
        "processing_time_ms": elapsed,
        "sample_chunks": [
            {
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "char_count": c["char_count"],
                "heading": c.get("heading", ""),
            }
            for c in chunk_records[:10]
        ],
    }

def search(query, n=5, language=None, hybrid=True):
    if hybrid:
        results = vector_store.hybrid_search(query, n=n, language=language)
    else:
        results = vector_store.search(query, n=n, language=language)
    return {
        "query": query,
        "results": results,
        "total": len(results),
    }

def list_documents():
    return sql_store.list_documents()

def get_stats():
    return sql_store.get_stats()

def delete_document(doc_id):
    vector_store.delete_document(doc_id)
    sql_store.delete_document(doc_id)