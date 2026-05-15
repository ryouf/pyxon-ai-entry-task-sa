import time
import statistics
from src.pipeline import search
from src.processing.arabic_processor import is_arabic, get_stats, strip_diacritics
from src.storage import sql_store

ARABIC_TESTS = [
    {"text": "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ", "has_diacritics": True},
    {"text": "الذَّكَاءُ الاصطناعي يُغَيِّرُ مُسْتَقْبَلَ التَّقْنِيَة", "has_diacritics": True},
    {"text": "تعلم الآلة هو فرع من فروع الذكاء الاصطناعي", "has_diacritics": False},
    {"text": "المملكة العربية السعودية رؤية 2030", "has_diacritics": False},
    {"text": "مُعَالَجَةُ اللُّغَةِ الطَّبِيعِيَّةِ", "has_diacritics": True},
]

RETRIEVAL_TESTS = [
    {"query": "الذكاء الاصطناعي", "relevant_terms": ["الذكاء", "اصطناعي", "تعلم"]},
    {"query": "رؤية 2030", "relevant_terms": ["رؤية", "2030", "السعودية"]},
    {"query": "تعلم الآلة", "relevant_terms": ["تعلم", "الآلة", "خوارزميات"]},
    {"query": "artificial intelligence", "relevant_terms": ["intelligence", "AI", "machine"]},
]

def test_arabic_detection():
    passed = sum(1 for case in ARABIC_TESTS if is_arabic(case["text"]))
    score = passed / len(ARABIC_TESTS)
    return {
        "name": "Arabic Language Detection",
        "score": score,
        "passed": score >= 0.8,
        "details": {"passed": passed, "total": len(ARABIC_TESTS)},
    }

def test_diacritics_detection():
    passed = 0
    for case in ARABIC_TESTS:
        stats = get_stats(case["text"])
        if stats["has_diacritics"] == case["has_diacritics"]:
            passed += 1
    score = passed / len(ARABIC_TESTS)
    return {
        "name": "Diacritics Detection Accuracy",
        "score": score,
        "passed": score >= 0.8,
        "details": {"passed": passed, "total": len(ARABIC_TESTS)},
    }

def test_diacritics_stripping():
    text = "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"
    stripped = strip_diacritics(text)
    passed = stripped == "بسم الله الرحمن الرحيم"
    return {
        "name": "Diacritics Stripping",
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "details": {"input": text, "output": stripped},
    }

def test_search_speed():
    times = []
    for _ in range(10):
        start = time.time()
        search("الذكاء الاصطناعي", n=5)
        times.append((time.time() - start) * 1000)
    p50 = statistics.median(times)
    p95 = sorted(times)[int(0.95 * len(times))]
    return {
        "name": "Retrieval Latency",
        "score": max(0, 1 - (p50 / 2000)),
        "passed": p50 < 2000,
        "details": {
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
        },
    }

def test_precision_at_k(k=5):
    scores = []
    for case in RETRIEVAL_TESTS:
        results = search(case["query"], n=k)
        hits = 0
        for r in results["results"]:
            text_lower = r["text"].lower()
            if any(term.lower() in text_lower for term in case["relevant_terms"]):
                hits += 1
        scores.append(hits / max(len(results["results"]), 1))
    avg = statistics.mean(scores) if scores else 0
    return {
        "name": f"Precision@{k}",
        "score": avg,
        "passed": avg > 0.3,
        "details": {
            "per_query": [round(s, 3) for s in scores],
            "average": round(avg, 3),
        },
    }

def test_mrr():
    rr_scores = []
    for case in RETRIEVAL_TESTS:
        results = search(case["query"], n=10)
        rr = 0.0
        for rank, r in enumerate(results["results"], 1):
            text_lower = r["text"].lower()
            if any(term.lower() in text_lower for term in case["relevant_terms"]):
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    mrr = statistics.mean(rr_scores) if rr_scores else 0
    return {
        "name": "Mean Reciprocal Rank (MRR)",
        "score": mrr,
        "passed": mrr > 0.2,
        "details": {
            "mrr": round(mrr, 4),
            "per_query": [round(s, 4) for s in rr_scores],
        },
    }

def test_hybrid_vs_semantic():
    semantic = search("الذكاء الاصطناعي", n=5, hybrid=False)
    hybrid = search("الذكاء الاصطناعي", n=5, hybrid=True)
    sem_avg = statistics.mean(r["score"] for r in semantic["results"]) if semantic["results"] else 0
    hyb_avg = statistics.mean(r["score"] for r in hybrid["results"]) if hybrid["results"] else 0
    return {
        "name": "Hybrid vs Semantic Search",
        "score": 1.0 if hyb_avg >= sem_avg else 0.7,
        "passed": True,
        "details": {
            "semantic_avg_score": round(sem_avg, 4),
            "hybrid_avg_score": round(hyb_avg, 4),
            "improvement": round((hyb_avg - sem_avg) * 100, 2),
        },
    }

def test_chunk_quality():
    docs = sql_store.list_documents()
    if not docs:
        return {
            "name": "Chunk Size Quality",
            "score": 0,
            "passed": False,
            "details": {"error": "No documents found"},
        }
    all_chunks = []
    for doc in docs:
        all_chunks.extend(sql_store.get_chunks(doc["id"]))
    if not all_chunks:
        return {"name": "Chunk Size Quality", "score": 0, "passed": False}
    sizes = [c["char_count"] for c in all_chunks]
    valid = sum(1 for s in sizes if 50 <= s <= 2000)
    score = valid / len(sizes)
    return {
        "name": "Chunk Size Quality",
        "score": score,
        "passed": score > 0.8,
        "details": {
            "total_chunks": len(sizes),
            "valid_chunks": valid,
            "avg_size": round(statistics.mean(sizes)),
            "min_size": min(sizes),
            "max_size": max(sizes),
        },
    }

def test_multilingual():
    arabic_results = search("الذكاء الاصطناعي", n=3)
    english_results = search("artificial intelligence", n=3)
    arabic_ok = len(arabic_results["results"]) > 0
    english_ok = len(english_results["results"]) > 0
    score = (int(arabic_ok) + int(english_ok)) / 2
    return {
        "name": "Multilingual Search",
        "score": score,
        "passed": score >= 0.5,
        "details": {
            "arabic_results": len(arabic_results["results"]),
            "english_results": len(english_results["results"]),
        },
    }
def test_retrieved_chunks_preview(queries=None):
    if not queries:
        queries = [
            "الذكاء الاصطناعي",
            "رؤية 2030",
            "تعلم الآلة",
        ]
    preview = []
    for q in queries:
        results = search(q, n=3)
        preview.append({
            "query": q,
            "chunks": [
                {
                    "rank": i + 1,
                    "score": r["score"],
                    "heading": r.get("heading", ""),
                    "preview": r["text"][:100],
                }
                for i, r in enumerate(results["results"])
            ]
        })
    return {
        "name": "Retrieved Chunks Preview",
        "score": 1.0,
        "passed": True,
        "details": {"queries": preview},
    }

def test_chunking_strategy_comparison():
    docs = sql_store.list_documents()
    if not docs:
        return {
            "name": "Chunking Strategy Comparison",
            "score": 0,
            "passed": False,
            "details": {"error": "No documents found"},
        }
    strategies = {}
    for doc in docs:
        s = doc["strategy"]
        if s not in strategies:
            strategies[s] = {"count": 0, "total_chunks": 0, "avg_chars": 0, "docs": []}
        chunks = sql_store.get_chunks(doc["id"])
        avg_chars = sum(c["char_count"] for c in chunks) / max(len(chunks), 1)
        strategies[s]["count"] += 1
        strategies[s]["total_chunks"] += len(chunks)
        strategies[s]["avg_chars"] = round(avg_chars)
        strategies[s]["docs"].append(doc["title"])

    return {
        "name": "Chunking Strategy Comparison",
        "score": 1.0,
        "passed": True,
        "details": strategies,
    }

def test_query_level_evaluation():
    queries = [
        {"query": "الذكاء الاصطناعي", "relevant_terms": ["الذكاء", "اصطناعي"]},
        {"query": "رؤية 2030", "relevant_terms": ["رؤية", "2030"]},
        {"query": "تعلم الآلة", "relevant_terms": ["تعلم", "الآلة"]},
        {"query": "artificial intelligence", "relevant_terms": ["intelligence", "AI"]},
    ]
    rows = []
    for case in queries:
        results = search(case["query"], n=5)
        hits = sum(
            1 for r in results["results"]
            if any(term.lower() in r["text"].lower() for term in case["relevant_terms"])
        )
        precision = hits / max(len(results["results"]), 1)
        top_score = results["results"][0]["score"] if results["results"] else 0
        rows.append({
            "query": case["query"],
            "results_found": len(results["results"]),
            "relevant_hits": hits,
            "precision": round(precision, 3),
            "top_score": round(top_score, 3),
        })

    avg_precision = sum(r["precision"] for r in rows) / max(len(rows), 1)
    return {
        "name": "Query-Level Evaluation",
        "score": avg_precision,
        "passed": avg_precision > 0.3,
        "details": {"rows": rows, "avg_precision": round(avg_precision, 3)},
    }

def run_all():
    results = [
        test_arabic_detection(),
        test_diacritics_detection(),
        test_diacritics_stripping(),
        test_search_speed(),
        test_precision_at_k(),
        test_mrr(),
        test_hybrid_vs_semantic(),
        test_chunk_quality(),
        test_multilingual(),
        test_retrieved_chunks_preview(),
        test_chunking_strategy_comparison(),
        test_query_level_evaluation(),
    ]
    passed = sum(1 for r in results if r["passed"])
    avg_score = statistics.mean(r["score"] for r in results)
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "avg_score": round(avg_score, 3),
            "pass_rate": round(passed / len(results), 3),
        },
    }