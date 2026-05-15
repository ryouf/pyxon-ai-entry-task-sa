from src.loaders.document_loader import load_document
from src.processing.chunker import chunk_document
from src.storage.vector_store import VectorStore


text = load_document(
    "data/sample_docs/structured_arabic.txt"
)

chunk_result = chunk_document(text)

chunks = chunk_result["chunks"]

vector_store = VectorStore()

vector_store.add_chunks(chunks)

query = "كيف يختار النظام استراتيجية التقسيم؟"

results = vector_store.search(query)

print(results["documents"])