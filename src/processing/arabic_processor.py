import re

ARABIC = re.compile(r'[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]')
DIACRITICS = re.compile(r'[\u064B-\u065F\u0670]')

def is_arabic(text):
    if not text:
        return False
    arabic = len(ARABIC.findall(text))
    alpha = sum(1 for c in text if c.isalpha())
    return (arabic / max(alpha, 1)) > 0.3

def strip_diacritics(text):
    return DIACRITICS.sub('', text)

def normalize(text):
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'[ؤئ]', 'ء', text)
    text = re.sub(r'\u0640', '', text)
    return text

def clean(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    return text.strip()

def get_stats(text):
    arabic = len(ARABIC.findall(text))
    diacritics = len(DIACRITICS.findall(text))
    return {
        "total_chars": len(text),
        "is_arabic": is_arabic(text),
        "has_diacritics": diacritics > 0,
        "arabic_ratio": arabic / max(len(text), 1),
    }

def prepare_for_embedding(text):
    text = clean(text)
    text = normalize(text)
    text = strip_diacritics(text)
    return text

def clean_ocr(text):
    text = re.sub(r'[a-zA-Z]+', '', text)
    text = re.sub(r'[،,]{2,}', '،', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()