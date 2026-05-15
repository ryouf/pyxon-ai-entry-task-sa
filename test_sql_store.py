from src.loaders.document_loader import load_document
from src.processing.chunker import chunk_document
from src.storage.sql_store import SQLStore


file_path = "data/sample_docs/structured_arabic.txt"

text = load_document(file_path)

chunk_result = chunk_document(text)

sql_store = SQLStore()

document_id = sql_store.add_document(
    file_name=file_path,
    chunking_strategy=chunk_result["strategy"]
)

sql_store.add_chunks(
    document_id=document_id,
    chunks=chunk_result["chunks"]
)

print("Document saved with ID:", document_id)
print("Documents:", sql_store.get_documents())
print("Chunks:", sql_store.get_chunks())