from openai import OpenAI
from src.pipeline import search
from src.processing.arabic_processor import is_arabic

GROQ_KEY = "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"

client = OpenAI(
    api_key=GROQ_KEY,
    base_url="https://api.groq.com/openai/v1"
)

def answer(question, n_chunks=5, document_id=None, language=None):
    import time

    retrieval_start = time.time()
    results = search(question, n=n_chunks, language=language, hybrid=True)
    retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)

    chunks = results["results"]

    if not chunks:
        return {
            "question": question,
            "answer": "لم أجد معلومات ذات صلة." if is_arabic(question) else "No relevant information found.",
            "sources": [],
            "retrieval_ms": retrieval_ms,
            "generation_ms": 0,
        }

    context = "\n\n---\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
    )

    if is_arabic(question):
        system = "أنت مساعد ذكي متخصص في تحليل المستندات. استخدم المقاطع المقدمة للإجابة على السؤال بدقة. إذا لم تجد الإجابة في المقاطع قل ذلك بوضوح."
        user = f"المقاطع:\n{context}\n\nالسؤال: {question}\n\nالإجابة:"
    else:
        system = "You are an intelligent document analysis assistant. Use the provided excerpts to answer the question accurately. If the answer is not in the excerpts, say so clearly."
        user = f"Excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    generation_start = time.time()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    generation_ms = round((time.time() - generation_start) * 1000, 2)

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": [
            {
                "chunk_id": c["chunk_id"],
                "score": c["score"],
                "text_preview": c["text"][:150],
                "heading": c.get("heading", ""),
            }
            for c in chunks
        ],
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
    }