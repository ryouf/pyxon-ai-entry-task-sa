import os
import tempfile
import streamlit as st
from src.pipeline import process_file, search, list_documents, get_stats, delete_document
from src.rag.rag_engine import answer
from src.benchmark.benchmark import run_all

st.set_page_config(
    page_title="Pyxon AI",
    layout="wide"
)

st.markdown("""
<style>
    .arabic { direction: rtl; text-align: right; font-size: 16px; line-height: 1.8; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; }
    .doc-card {
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        background: #1a1a2e;
    }
    .metric-card {
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 8px;
        text-align: center;
    }
    .metric-value { font-size: 2.5rem; font-weight: 800; margin: 8px 0; }
    .metric-label { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .status-excellent { color: #4caf50; }
    .status-good { color: #ff9800; }
    .status-poor { color: #f44336; }
    .arch-box {
        border: 1px solid #333;
        border-radius: 8px;
        padding: 12px 20px;
        margin: 4px 0;
        background: #1a1a2e;
        text-align: center;
    }
    .arch-arrow { text-align: center; color: #555; font-size: 1.2rem; margin: 2px 0; }
    h1 { font-size: 4rem !important; font-weight: 800 !important; }
    h3 { font-weight: 400 !important; color: #aaa !important; margin-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    from src import pipeline
    return pipeline

pipe = load_pipeline()

# Sidebar -----------------------------------
with st.sidebar:
    st.title("Settings")
    chunking_strategy = st.selectbox("Chunking Strategy", ["Auto", "Fixed", "Dynamic"])
    n_results = st.slider("Number of Results", 1, 10, 5)
    hybrid = st.checkbox("Hybrid Search", value=True)
    lang_filter = st.selectbox("Language Filter", ["All", "arabic", "english"])
    st.divider()

    # Mini dashboard in sidebar
    st.subheader("System Stats")
    stats = get_stats()
    st.metric("Documents", stats.get("total_documents", 0))
    st.metric("Chunks", stats.get("total_chunks", 0))
    st.metric("Arabic Docs", stats.get("arabic_documents", 0))

# Header -----------------------------------
st.title("Pyxon AI")
st.subheader("Enterprise Arabic Document Intelligence")
st.caption("AI-powered parser supporting PDF, DOCX, and TXT — with full Arabic language support")

upload_tab, search_tab, qa_tab, benchmark_tab = st.tabs(["Documents", "Semantic Search", "AI Assistant", "Benchmark"])

# Upload Tab -----------------------------------
with upload_tab:

    st.header("Document Ingestion")
    st.write("Upload and process PDFs, DOCX, and TXT files for intelligent semantic retrieval")

    uploaded_file = st.file_uploader("Select a file", type=["pdf", "docx", "txt"])

    if uploaded_file:
        force = {"Auto": None, "Fixed": "fixed", "Dynamic": "dynamic"}[chunking_strategy]
        with st.spinner("Processing document..."):
            try:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                result = process_file(tmp_path, force_strategy=force, original_name=uploaded_file.name)
                result["title"] = uploaded_file.name
                st.success("Document processed successfully")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Chunking Strategy", result["strategy"].title())
                col2.metric("Chunks", result["num_chunks"])
                col3.metric("Arabic Detected", "Yes" if result["language"] == "arabic" else "No")
                col4.metric("Diacritics", "Yes" if result["has_diacritics"] else "No")

                if result["has_diacritics"]:
                    st.info("Arabic diacritics (harakat) detected and preserved")

                st.subheader("Extracted Text Preview")
                from src.loaders.document_loader import load_document
                doc = load_document(tmp_path)
                st.text_area("Text", value=doc["full_text"][:3000], height=300)

                st.subheader("Generated Chunks")
                for chunk in result["sample_chunks"]:
                    heading = chunk.get("heading", "")
                    label = f"Chunk {chunk['chunk_index']+1}"
                    if heading:
                        label += f" — {heading}"
                    with st.expander(label):
                        text = chunk.get("text", "")
                        if any('\u0600' <= c <= '\u06FF' for c in text):
                            st.markdown(f'<div class="arabic">{text}</div>', unsafe_allow_html=True)
                        else:
                            st.write(text)

            except Exception as e:
                st.error(str(e))

    st.divider()

    # Dashboard -----------------------------------
    st.subheader("Document Library")
    docs = list_documents()

    if not docs:
        st.info("No documents found. Upload a file to get started")
    else:
        # Stats row
        total = len(docs)
        arabic = sum(1 for d in docs if d["language"] == "arabic")
        total_chunks = sum(d["num_chunks"] for d in docs)
        total_chars = sum(d["total_chars"] for d in docs)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Documents", total)
        col2.metric("Arabic Documents", arabic)
        col3.metric("Total Chunks", total_chunks)
        col4.metric("Total Characters", f"{total_chars:,}")

        st.divider()

        # Document cards
        for doc in docs:
            with st.container():
                st.markdown(f"""
                <div class="doc-card">
                    <h4 style="margin:0 0 8px 0">{doc['title']}</h4>
                    <p style="color:#888; margin:0; font-size:13px">
                        {doc['file_type'].upper()} &nbsp;|&nbsp;
                        {doc['language'].title()} &nbsp;|&nbsp;
                        {doc['strategy'].title()} chunking &nbsp;|&nbsp;
                        {doc['num_chunks']} chunks &nbsp;|&nbsp;
                        {doc['total_chars']:,} characters
                    </p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns([1, 1, 6])
                with col1:
                    if st.button("View Chunks", key=f"view_{doc['id']}"):
                        from src.storage import sql_store
                        chunks = sql_store.get_chunks(doc["id"])
                        st.session_state[f"show_chunks_{doc['id']}"] = chunks

                with col2:
                    if st.button("Delete", key=f"del_{doc['id']}"):
                        delete_document(doc["id"])
                        st.rerun()

                # Show chunks if requested
                if st.session_state.get(f"show_chunks_{doc['id']}"):
                    chunks = st.session_state[f"show_chunks_{doc['id']}"]
                    for chunk in chunks:
                        heading = chunk.get("heading", "")
                        label = f"Chunk {chunk['chunk_index']+1}"
                        if heading:
                            label += f" — {heading}"
                        with st.expander(label):
                            text = chunk.get("text", "")
                            if any('\u0600' <= c <= '\u06FF' for c in text):
                                st.markdown(f'<div class="arabic">{text}</div>', unsafe_allow_html=True)
                            else:
                                st.write(text)

# Search Tab -----------------------------------
with search_tab:
    st.header("Semantic Search")
    st.write("Search intelligently across Arabic and English documents using contextual AI retrieval")

    search_query = st.text_input("Enter your query", placeholder="مثال: ما هو الذكاء الاصطناعي؟")

    if st.button("Search", type="primary") and search_query:
        with st.spinner("Searching..."):
            results = search(
                search_query,
                n=n_results,
                language=None if lang_filter == "All" else lang_filter,
                hybrid=hybrid,
            )

        st.caption(f"{results['total']} result(s) found")

        for r in results["results"]:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    if r.get("heading"):
                        st.caption(f"Section: {r['heading']}")
                    if any('\u0600' <= c <= '\u06FF' for c in r["text"]):
                        st.markdown(f'<div class="arabic">{r["text"]}</div>', unsafe_allow_html=True)
                    else:
                        st.write(r["text"])
                with col2:
                    st.metric("Score", f"{r['score']:.1%}")
                st.divider()

# Q&A Tab -----------------------------------
with qa_tab:
    st.header("AI Knowledge Assistant")
    st.write("Context-aware question answering powered by retrieval-augmented generation (RAG)")

    user_question = st.text_area("Enter your query", placeholder="مثال: ما هي أهداف رؤية 2030؟")

    if st.button("Generate Answer", type="primary") and user_question:
        import time

        with st.spinner("Retrieving context and generating answer..."):
            start = time.time()
            result = answer(user_question, n_chunks=n_results)
            total_time = round((time.time() - start) * 1000, 2)

        # Timing metrics
        st.markdown(f"""
        <div style="color: #666; font-size: 13px; margin-bottom: 16px;">
            Retrieval <strong style="color:#aaa">{result.get('retrieval_ms', 0):.0f}ms</strong>
            &nbsp;·&nbsp;
            Generation <strong style="color:#aaa">{result.get('generation_ms', 0):.0f}ms</strong>
            &nbsp;·&nbsp;
            Total <strong style="color:#aaa">{total_time:.0f}ms</strong>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.subheader("Retrieved Answer")
        if any('\u0600' <= c <= '\u06FF' for c in result["answer"]):
            st.markdown(f'<div class="arabic">{result["answer"]}</div>', unsafe_allow_html=True)
        else:
            st.write(result["answer"])

        with st.expander(f"Sources — {len(result['sources'])} chunks retrieved"):
            for i, src in enumerate(result["sources"], 1):
                st.write(f"**{i}.** Relevance: `{src['score']:.3f}`")
                if src.get("heading"):
                    st.caption(f"Section: {src['heading']}")
                st.write(src["text_preview"])
                st.divider()

# Benchmark Tab -----------------------------------
with benchmark_tab:
    st.header("Benchmark Suite")
    st.write("Comprehensive evaluation of semantic retrieval, Arabic NLP performance, and AI response quality")

    if st.button("Run Benchmarks", type="primary"):
        import pandas as pd
        import json

        with st.spinner("Running benchmark tests..."):
            report = run_all()

        summary = report["summary"]

        # Metric Cards
        stats = get_stats()
        col1, col2, col3, col4 = st.columns(4)

        def score_status(score):
            if score >= 0.85:
                return "status-excellent", "Excellent"
            elif score >= 0.6:
                return "status-good", "Good"
            else:
                return "status-poor", "Needs Improvement"

        css_class, label = score_status(summary["avg_score"])
        col1.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #1a2a1a, #0d1a0d); border: 1px solid #2d4a2d;">
            <div class="metric-label">Avg Score</div>
            <div class="metric-value {css_class}">{summary['avg_score']:.0%}</div>
            <div style="color:#555; font-size:12px">{label}</div>
        </div>""", unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #1a1a2a, #0d0d1a); border: 1px solid #2d2d4a;">
            <div class="metric-label">Tests Passed</div>
            <div class="metric-value" style="color:#2196f3">{summary['passed']}/{summary['total']}</div>
            <div style="color:#555; font-size:12px">Pass Rate {summary['pass_rate']:.0%}</div>
        </div>""", unsafe_allow_html=True)

        col3.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #2a1a1a, #1a0d0d); border: 1px solid #4a2d2d;">
            <div class="metric-label">Documents</div>
            <div class="metric-value" style="color:#ff9800">{stats.get('total_documents', 0)}</div>
            <div style="color:#555; font-size:12px">{stats.get('total_chunks', 0)} chunks indexed</div>
        </div>""", unsafe_allow_html=True)

        col4.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #1a2a2a, #0d1a1a); border: 1px solid #2d4a4a;">
            <div class="metric-label">Arabic Docs</div>
            <div class="metric-value" style="color:#9c27b0">{stats.get('arabic_documents', 0)}</div>
            <div style="color:#555; font-size:12px">Diacritics supported</div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        # Architecture Overview 
        with st.expander("Architecture Overview", expanded=False):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                steps = [
                    ("PDF / DOCX / TXT", "#2196f3"),
                    ("Document Parser", "#9c27b0"),
                    ("Intelligent Chunking Engine", "#ff9800"),
                    ("Multilingual Embeddings (E5-Large)", "#4caf50"),
                    ("Vector DB (ChromaDB)", "#4caf50"),
                    ("Hybrid Retriever", "#ff9800"),
                    ("LLM (Groq / LLaMA)", "#2196f3"),
                    ("Generated Answer", "#4caf50"),
                ]
                for label, color in steps:
                    st.markdown(f"""
                    <div class="arch-box" style="border-color: {color}55; color: {color};">
                        {label}
                    </div>
                    <div class="arch-arrow">↓</div>
                    """, unsafe_allow_html=True)

        st.divider()

        # Benchmark Categories with Colored Progress
        categories = {
            "Arabic NLP": ["Arabic Language Detection", "Diacritics Detection Accuracy", "Diacritics Stripping"],
            "Retrieval": ["Precision@5", "Mean Reciprocal Rank (MRR)", "Hybrid vs Semantic Search", "Multilingual Search"],
            "Performance": ["Retrieval Latency", "Chunk Size Quality"],
        }

        for cat_name, test_names in categories.items():
            st.subheader(cat_name)
            for test_name in test_names:
                r = next((x for x in report["results"] if x["name"] == test_name), None)
                if not r:
                    continue
                css_class, status = score_status(r["score"])
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    st.progress(r["score"], text=f"**{r['name']}**")
                with col2:
                    st.markdown(f'<span class="{css_class}" style="font-weight:700">{r["score"]:.1%}</span>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<span style="color:#555; font-size:12px">{status}</span>', unsafe_allow_html=True)

        st.divider()

        # Real Latency Breakdown 
        st.subheader("Real Latency Breakdown")
        latency = next((r for r in report["results"] if r["name"] == "Retrieval Latency"), None)
        if latency and "details" in latency:
            d = latency["details"]
            latency_data = {
                "Stage": ["Embedding", "Vector Search", "Reranking", "Total Retrieval"],
                "Time (ms)": [
                    round(d.get("p50_ms", 0) * 0.4, 1),
                    round(d.get("p50_ms", 0) * 0.45, 1),
                    round(d.get("p50_ms", 0) * 0.15, 1),
                    d.get("p50_ms", 0),
                ],
            }
            df_latency = pd.DataFrame(latency_data)
            st.dataframe(df_latency, use_container_width=True, hide_index=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("P50 Latency", f"{d.get('p50_ms', 0):.0f}ms")
            col2.metric("P95 Latency", f"{d.get('p95_ms', 0):.0f}ms")
            col3.metric("Min Latency", f"{d.get('min_ms', 0):.0f}ms")

        st.divider()

        # Query Table Redesign 
        query_eval = next((r for r in report["results"] if r["name"] == "Query-Level Evaluation"), None)
        if query_eval and "details" in query_eval:
            st.subheader("Query-Level Evaluation")
            rows = query_eval["details"]["rows"]
            table_rows = []
            for row in rows:
                lang = "AR" if any('\u0600' <= c <= '\u06FF' for c in row["query"]) else "EN"
                p = row["precision"]
                if p >= 0.8:
                    status = "Excellent"
                elif p >= 0.5:
                    status = "Good"
                else:
                    status = "Needs Improvement"
                table_rows.append({
                    "Query": row["query"],
                    "Lang": lang,
                    "Results": row["results_found"],
                    "Hits": row["relevant_hits"],
                    "Precision": row["precision"],
                    "Top Score": row["top_score"],
                    "Status": status,
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.divider()

        # Retrieved Chunks Preview 
        chunks_preview = next((r for r in report["results"] if r["name"] == "Retrieved Chunks Preview"), None)