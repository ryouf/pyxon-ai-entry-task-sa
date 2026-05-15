from src.loaders.document_loader import load_document
from src.processing.chunker import chunk_document


text = load_document("data/sample_docs/structured_arabic.txt")

result = chunk_document(text)

print("Strategy:", result["strategy"])
print("Number of chunks:", len(result["chunks"]))

for index, chunk in enumerate(result["chunks"], start=1):
    print(f"\nChunk {index}:")
    print(chunk)