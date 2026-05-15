import re
from src.processing.arabic_processor import is_arabic

def split_sentences(text, language):
    if language == "arabic":
        parts = re.split(r'(?<=[.!?؟\n])\s+', text)
    else:
        parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def fixed_chunk(text, chunk_size=512, overlap=64):
    language = "arabic" if is_arabic(text) else "english"
    sentences = split_sentences(text, language)
    chunks = []
    current = []
    current_size = 0

    for sentence in sentences:
        if current_size + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_text = []
            overlap_size = 0
            for s in reversed(current):
                if overlap_size + len(s) <= overlap:
                    overlap_text.insert(0, s)
                    overlap_size += len(s)
                else:
                    break
            current = overlap_text
            current_size = sum(len(s) for s in current)
        current.append(sentence)
        current_size += len(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks

def dynamic_chunk(text):
    lines = text.split("\n")
    segments = []
    current_heading = ""
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        is_heading = (
            len(stripped) < 60 and
            not stripped.endswith(".") and
            not stripped.endswith("،") and
            not stripped.endswith("ً") and
            not stripped.endswith("اً")
        )

        if is_heading and current_lines:
            segments.append({
                "heading": current_heading,
                "text": "\n".join(current_lines).strip()
            })
            current_lines = []
            current_heading = stripped
        elif is_heading:
            current_heading = stripped
        else:
            current_lines.append(stripped)

    if current_lines:
        segments.append({
            "heading": current_heading,
            "text": "\n".join(current_lines).strip()
        })

    if not segments:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        segments = [{"heading": "", "text": p} for p in paragraphs]

    return segments

def select_strategy(text, num_headings=0, num_lists=0, num_pages=1):
    if len(text) < 1000:
        return "dynamic"
    if num_headings >= 3:
        return "dynamic"
    if num_pages > 2:
        return "dynamic"
    if num_lists > 10:
        return "dynamic"
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 5:
        lengths = [len(p) for p in paragraphs]
        avg = sum(lengths) / len(lengths)
        variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
        cv = (variance ** 0.5) / max(avg, 1)
        if cv < 0.3:
            return "fixed"
    return "dynamic"

def chunk_document(text, num_headings=0, num_lists=0, num_pages=1):
    strategy = select_strategy(text, num_headings, num_lists, num_pages)
    if strategy == "fixed":
        chunks = [{"heading": "", "text": c} for c in fixed_chunk(text)]
    else:
        chunks = dynamic_chunk(text)
    result = []
    for c in chunks:
        if isinstance(c, dict) and c.get("text", "").strip():
            result.append(c)
        elif isinstance(c, str) and c.strip():
            result.append({"heading": "", "text": c})
    return strategy, result