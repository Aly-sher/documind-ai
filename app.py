"""
DocuMind AI — Main Streamlit Application
Phase 1 features added:
  - Document Health Report  (sidebar, post-processing)
  - Complexity Tuner        (sidebar slider — Simple / Balanced / Expert)
  - Export Chat as PDF      (sidebar download button)
  - Multilingual Q&A        (auto-detects Urdu / Roman Urdu / Arabic)
"""

from __future__ import annotations

import os
import time

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from utils.logger import logger, setup_logger
from utils.validators import validate_api_key, validate_query, validate_uploaded_file

from src.pdf_processor import process_pdf
from src.chunker import create_chunks
from src.embedder import get_embedding_model
from src.vector_store import build_vector_store
from src.retriever import get_retriever
from src.llm_chain import build_rag_chain, run_query, GROQ_MODEL, MAX_HISTORY_TURNS
from src.prompt_builder import format_retrieved_chunks, COMPLEXITY_LEVELS

# ── Phase 1 imports ──────────────────────────────────────────────────────────
from src.document_health import analyse_document_health, health_emoji, health_label
from src.language_detector import detect_language, language_flag
from src.exporter import export_chat_to_pdf

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
setup_logger("INFO")

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── API key: local .env → Streamlit Cloud Secrets ────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
api_key: str | None = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
if api_key:
    os.environ["GROQ_API_KEY"] = api_key

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

def _init_session() -> None:
    defaults = {
        "chat_history":    [],
        "messages": [
            {
                "role":    "assistant",
                "content": "👋 Upload a PDF in the sidebar and ask me anything about it!",
            }
        ],
        "vectorstore":     None,
        "embedding_model": None,
        "doc_info":        None,
        "doc_health":      None,   # Phase 1 — Document Health Report
        "processing_done": False,
        "complexity_level": "Balanced",   # Phase 1 — Complexity Tuner
        "document_name":   "",            # Phase 1 — for export header
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session()

# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------
st.title("📄 DocuMind AI")
st.markdown(
    f"<span style='color:grey;font-size:0.9em;'>"
    f"Powered by <b>Llama 3.1</b> via Groq API &nbsp;·&nbsp; "
    f"FAISS Vector Search &nbsp;·&nbsp; Model: <code>{GROQ_MODEL}</code>"
    f"</span>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    # ── 1. Document Upload ───────────────────────────────────────────────
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type=["pdf"],
        help="Maximum file size: 50 MB. PDF format only.",
    )

    process_btn = st.button(
        "⚙️ Process Document",
        use_container_width=True,
        disabled=(uploaded_file is None),
    )

    if process_btn and uploaded_file is not None:
        is_valid, err_msg = validate_uploaded_file(uploaded_file)
        if not is_valid:
            st.error(err_msg)
        else:
            # Reset session for new document
            st.session_state.vectorstore     = None
            st.session_state.chat_history    = []
            st.session_state.doc_health      = None
            st.session_state.doc_info        = None
            st.session_state.processing_done = False
            st.session_state.document_name   = uploaded_file.name
            st.session_state.messages = [
                {"role": "assistant", "content": "👋 New document loaded! Ask me anything about it."}
            ]

            progress = st.progress(0, text="Starting…")
            try:
                # 1. Extract
                progress.progress(10, text="📖 Extracting text from PDF…")
                file_bytes = uploaded_file.read()
                documents, status = process_pdf(file_bytes, uploaded_file.name)

                if status == "empty_pdf":
                    st.error("This PDF appears to be empty (0 pages).")
                    progress.empty(); st.stop()
                elif status == "image_only_pdf":
                    st.error("This PDF is image-only (scanned) — no extractable text.")
                    progress.empty(); st.stop()
                elif status == "corrupt_pdf":
                    st.error("Could not read this PDF — the file may be corrupted.")
                    progress.empty(); st.stop()
                elif status == "unknown_error":
                    st.error("An unexpected error occurred while reading the PDF.")
                    progress.empty(); st.stop()

                # 2. Chunk
                progress.progress(30, text="✂️ Chunking text…")
                chunks = create_chunks(documents)

                # 3. Embedding model (cached in session)
                progress.progress(50, text="🧠 Loading embedding model…")
                if st.session_state.embedding_model is None:
                    st.session_state.embedding_model = get_embedding_model()

                # 4. Vector store
                progress.progress(70, text="📐 Building FAISS vector index…")
                vector_store = build_vector_store(chunks, st.session_state.embedding_model)
                st.session_state.vectorstore = vector_store

                # 5. Document metadata
                progress.progress(85, text="📊 Analysing document health…")
                st.session_state.doc_info = {
                    "filename": uploaded_file.name,
                    "pages":    documents[-1]["metadata"].get("total_pages", len(documents))
                                if isinstance(documents[-1], dict)
                                else getattr(getattr(documents[-1], "metadata", {}), "get",
                                             lambda k, d: d)("total_pages", len(documents)),
                    "chunks":   len(chunks),
                    "size_kb":  round(len(file_bytes) / 1024, 1),
                }

                # ── Phase 1: Document Health Report ──────────────────────
                st.session_state.doc_health = analyse_document_health(documents, chunks)

                progress.progress(100, text="Done!")
                st.session_state.processing_done = True
                time.sleep(0.4)
                progress.empty()

                st.success(f"✅ '{uploaded_file.name}' processed successfully!")
                logger.info(
                    f"Processed: {uploaded_file.name} | "
                    f"pages={st.session_state.doc_info['pages']} | "
                    f"chunks={len(chunks)}"
                )

            except (ValueError, RuntimeError) as exc:
                progress.empty()
                st.error(f"Processing error: {exc}")
                logger.error(f"Processing error: {exc}")

    # ── 2. Document Info Panel ───────────────────────────────────────────
    if st.session_state.doc_info:
        info = st.session_state.doc_info
        st.divider()
        st.subheader("📊 Document Info")
        st.markdown(f"**File:** `{info['filename']}`")
        c1, c2 = st.columns(2)
        c1.metric("Pages",  info["pages"])
        c2.metric("Chunks", info["chunks"])
        st.caption(f"File size: {info['size_kb']} KB")

    # ── 3. Phase 1: Document Health Report ──────────────────────────────
    if st.session_state.doc_health:
        h = st.session_state.doc_health
        st.divider()
        with st.expander(
            f"{health_emoji(h['health_score'])} Document Health — {health_label(h['health_score'])} "
            f"({h['health_score']}%)",
            expanded=False,
        ):
            hc1, hc2 = st.columns(2)
            hc1.metric("Words",        f"{h['total_words']:,}")
            hc2.metric("Reading Time", f"~{h['reading_time_mins']} min")

            hc3, hc4 = st.columns(2)
            hc3.metric("Text Coverage",   f"{h['text_coverage_pct']}%")
            hc4.metric("Chunk Quality",   f"{h['chunk_quality_score']}%")

            if h["image_only_risk"]:
                st.warning(
                    f"⚠️ {h['sparse_count']} page(s) appear image-only "
                    f"(pages: {', '.join(str(p) for p in h['sparse_pages'][:5])}). "
                    "Answers from these pages may be unavailable."
                )
            else:
                st.success("✅ All pages have extractable text.")

    # ── 4. Phase 1: Complexity Tuner ────────────────────────────────────
    if st.session_state.processing_done:
        st.divider()
        st.subheader("🎚️ Answer Complexity")
        st.session_state.complexity_level = st.select_slider(
            "Adjust the depth of answers:",
            options=COMPLEXITY_LEVELS,
            value=st.session_state.complexity_level,
            help=(
                "Simple — plain language for beginners\n"
                "Balanced — clear undergraduate level\n"
                "Expert — precise technical terminology"
            ),
        )

    # ── 5. Phase 1: Export Chat as PDF ──────────────────────────────────
    if st.session_state.processing_done and len(st.session_state.messages) > 1:
        st.divider()
        st.subheader("📤 Export Session")

        has_exchanges = any(m["role"] == "user" for m in st.session_state.messages)
        if has_exchanges:
            with st.spinner("Preparing export…"):
                try:
                    pdf_bytes = export_chat_to_pdf(
                        messages=st.session_state.messages,
                        document_name=st.session_state.document_name,
                    )
                    st.download_button(
                        label="⬇️ Download Q&A as PDF",
                        data=pdf_bytes,
                        file_name=f"DocuMind_Export_{st.session_state.document_name}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Export failed: {exc}")
                    logger.error(f"PDF export error: {exc}")
        else:
            st.caption("Ask at least one question to enable export.")

    # ── 6. Reset ─────────────────────────────────────────────────────────
    st.divider()
    if st.button("🗑️ Clear & Reset", use_container_width=True):
        for key in ["vectorstore", "doc_info", "doc_health", "document_name"]:
            st.session_state[key] = None
        st.session_state.chat_history    = []
        st.session_state.messages        = [
            {"role": "assistant", "content": "👋 Session cleared. Upload a PDF to get started!"}
        ]
        st.session_state.processing_done  = False
        st.session_state.complexity_level = "Balanced"
        st.rerun()

# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    if msg["role"] == "assistant" and "sources" in msg:
        with st.expander("📎 Source Chunks Used"):
            st.markdown(msg["sources"])

# Chat input
placeholder_text = (
    "Ask a question about your document…"
    if st.session_state.processing_done
    else "Upload and process a PDF first…"
)

if prompt := st.chat_input(placeholder_text, disabled=not st.session_state.processing_done):

    # Guards
    key_valid, key_err = validate_api_key(api_key)
    if not key_valid:
        st.error(key_err); st.stop()

    if st.session_state.vectorstore is None:
        st.error("Please upload and process a PDF document first."); st.stop()

    query_valid, query_err = validate_query(prompt)
    if not query_valid:
        st.warning(query_err); st.stop()

    # ── Phase 1: Language Detection ──────────────────────────────────────
    lang_code, lang_name = detect_language(prompt)
    flag = language_flag(lang_code)
    if lang_code != "en":
        st.toast(f"{flag} Detected {lang_name} — responding in {lang_name}", icon=flag)
        logger.info(f"Language detected: {lang_code} ({lang_name})")

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                retriever = get_retriever(st.session_state.vectorstore)

                # ── Phase 1: pass complexity + language into chain ────────
                rag_chain = build_rag_chain(
                    retriever=retriever,
                    api_key=api_key,
                    complexity_level=st.session_state.complexity_level,
                    language_code=lang_code,
                )

                result     = run_query(
                    rag_chain=rag_chain,
                    question=prompt,
                    chat_history=st.session_state.chat_history,
                )
                answer     = result["answer"]
                source_docs = result["sources"]

                st.markdown(answer)

                sources_formatted = format_retrieved_chunks(source_docs)
                with st.expander("📎 Source Chunks Used"):
                    st.markdown(f"```\n{sources_formatted}\n```")

                # Persist
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": f"```\n{sources_formatted}\n```",
                })
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=answer),
                ])

            except RuntimeError as exc:
                err_str = str(exc)

                if "rate_limit_error" in err_str:
                    msg = err_str.split("|", 1)[-1]
                    st.error(f"⏳ {msg}")
                    countdown = st.empty()
                    for i in range(60, 0, -1):
                        countdown.caption(f"Retry in {i}s…")
                        time.sleep(1)
                    countdown.empty()

                elif "timeout_error" in err_str:
                    st.error(f"⌛ {err_str.split('|', 1)[-1]}")

                elif "auth_error" in err_str:
                    st.error(f"🔑 {err_str.split('|', 1)[-1]}")

                else:
                    st.error("An unexpected error occurred. Please try again.")
                    logger.error(f"Unhandled RuntimeError: {exc}")

            except Exception as exc:
                st.error("An unexpected error occurred. Please try again.")
                logger.error(f"Unhandled exception: {exc}")