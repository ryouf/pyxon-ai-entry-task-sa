import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from src.processing.arabic_processor import strip_diacritics, is_arabic

model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("intfloat/multilingual-e5-large")
    return model

def embed(texts):
    m = get_model()
    prefixed = [f"passage: {t}" for t in texts]
    return m.encode(prefixed, normalize_embeddings=True).tolist()

def embed_query(query):
    m = get_model()
    if is_arabic(query):
        query = strip_diacritics(query)
    return m.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()

def get_collection():
    client = chromadb.PersistentClient(
        path="./data/chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )

def add_chunks(chunks):
    collection = get_collection()
    ids = [c["id"] for c in chunks]
    texts = [c["text_normalized"] for c in chunks]
    embeddings = embed(texts)
    metadatas = [
        {
            "document_id": c["document_id"],
            "chunk_index": c["chunk_index"],
            "language": c["language"],
            "has_diacritics": str(c["has_diacritics"]),
            "heading": c.get("heading", ""),
            "original_text": c["text"][:500],
        }
        for c in chunks
    ]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts
    )

def search(query, n=5, language=None):
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []
    query_vec = embed_query(query)
    where = {"language": language} if language else None
    kwargs = {
        "query_embeddings": [query_vec],
        "n_results": min(n, total),
    }
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    output = []
    if results and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            score = 1 - results["distances"][0][i]
            meta = results["metadatas"][0][i]
            output.append({
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "text": meta.get("original_text", ""),
                "document_id": meta.get("document_id"),
                "language": meta.get("language"),
                "heading": meta.get("heading", ""),
            })
    return output

def hybrid_search(query, n=5, language=None):
    results = search(query, n=n * 2, language=language)
    query_terms = set(query.lower().split())
    for r in results:
        text_lower = r["text"].lower()
        keyword_hits = sum(1 for term in query_terms if term in text_lower)
        keyword_score = keyword_hits / max(len(query_terms), 1)
        r["score"] = round(0.7 * r["score"] + 0.3 * keyword_score, 4)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:n]

def delete_document(document_id):
    collection = get_collection()
    results = collection.get(where={"document_id": document_id})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])