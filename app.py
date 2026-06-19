import os
import time
 
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
 
from utils.logger import logger, setup_logger
from utils.validators import validate_uploaded_file, validate_query, validate_api_key
from src.pdf_processor import process_pdf
from src.chunker import create_chunks
from src.embedder import get_embedding_model
from src.vector_store import build_vector_store
from src.retriever import get_retriever
from src.llm_chain import build_rag_chain, run_query, GROQ_MODEL, MAX_HISTORY_TURNS
from src.prompt_builder import format_retrieved_chunks
 
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
 
# --- API Key Resolution (local .env → Streamlit Cloud Secrets) ---
# --- API Key Resolution (local .env → Streamlit Cloud Secrets) ---
load_dotenv()
api_key: str | None = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception as e:
        st.error(f"Failed to read Streamlit secrets: {e}")

# If it's STILL completely blank after checking both, print a massive warning on the UI
if not api_key:
    st.error("🚨 DEBUG: os.getenv() and st.secrets both returned None. The app cannot find the key anywhere on your local machine.")
else:
    st.success(f"✅ DEBUG: Key found! It starts with: {api_key[:8]}...")
    os.environ["GROQ_API_KEY"] = api_key
 
# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------
 
def _init_session() -> None:
    """Initialise all required session state keys on first run."""
    defaults = {
        "chat_history": [],       # LangChain message objects (HumanMessage / AIMessage)
        "messages": [             # Streamlit UI message dicts
            {
                "role": "assistant",
                "content": "👋 Upload a PDF in the sidebar and ask me anything about it!",
            }
        ],
        "vectorstore": None,      # FAISS vector store
        "embedding_model": None,  # Cached embedding model (avoid reloading per query)
        "doc_info": None,         # Metadata dict for the sidebar info panel
        "processing_done": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
 
 
_init_session()
 
# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
 
st.title("📄 DocuMind AI")
st.markdown(
    f"<span style='color: grey; font-size: 0.9em;'>Powered by **Llama 3** via Groq API &nbsp;·&nbsp; "
    f"FAISS Vector Search &nbsp;·&nbsp; Model: `{GROQ_MODEL}`</span>",
    unsafe_allow_html=True,
)
st.divider()
 
# ---------------------------------------------------------------------------
# Sidebar — Document Upload & Info
# ---------------------------------------------------------------------------
 
with st.sidebar:
    st.header("📁 Document Upload")
 
    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type=["pdf"],
        help=f"Maximum file size: 50 MB. PDF format only.",
    )
 
    # --- Process button ---
    process_btn = st.button(
        "⚙️ Process Document",
        use_container_width=True,
        disabled=(uploaded_file is None),
    )
 
    if process_btn and uploaded_file is not None:
 
        # 1. Validate file
        is_valid, err_msg = validate_uploaded_file(uploaded_file)
        if not is_valid:
            st.error(err_msg)
        else:
            # Reset prior session on new upload
            st.session_state.vectorstore = None
            st.session_state.chat_history = []
            st.session_state.messages = [
                {"role": "assistant", "content": "👋 New document loaded! Ask me anything about it."}
            ]
            st.session_state.doc_info = None
            st.session_state.processing_done = False
 
            progress = st.progress(0, text="Starting…")
 
            try:
                # 2. Extract text
                progress.progress(10, text="📖 Extracting text from PDF…")
                file_bytes = uploaded_file.read()
                documents, status = process_pdf(file_bytes, uploaded_file.name)
 
                # Map status to user-friendly messages (SRS §8 table)
                if status == "empty_pdf":
                    st.error("This PDF appears to be empty (0 pages).")
                    progress.empty()
                    st.stop()
                elif status == "image_only_pdf":
                    st.error(
                        "This PDF appears to be image-only (scanned) with no extractable text. "
                        "Please use a text-based PDF."
                    )
                    progress.empty()
                    st.stop()
                elif status == "corrupt_pdf":
                    st.error("Could not read this PDF — the file may be corrupted. Please try another file.")
                    progress.empty()
                    st.stop()
                elif status == "unknown_error":
                    st.error("An unexpected error occurred while reading the PDF. Please try again.")
                    progress.empty()
                    st.stop()
 
                progress.progress(30, text="✂️ Chunking text…")
 
                # 3. Chunk
                chunks = create_chunks(documents)
 
                progress.progress(50, text="🧠 Loading embedding model…")
 
                # 4. Load embedding model (cache in session so it loads only once)
                if st.session_state.embedding_model is None:
                    st.session_state.embedding_model = get_embedding_model()
 
                progress.progress(70, text="📐 Building FAISS vector index…")
 
                # 5. Build vector store
                vector_store = build_vector_store(chunks, st.session_state.embedding_model)
                st.session_state.vectorstore = vector_store
 
                progress.progress(95, text="✅ Finalising…")
 
                # 6. Store document info for sidebar panel
                st.session_state.doc_info = {
                    "filename": uploaded_file.name,
                    "pages": documents[-1]["metadata"].get("total_pages", len(documents)),
                    "chunks": len(chunks),
                    "size_kb": round(len(file_bytes) / 1024, 1),
                }
                st.session_state.processing_done = True
 
                progress.progress(100, text="Done!")
                time.sleep(0.5)
                progress.empty()
 
                st.success(f"✅ '{uploaded_file.name}' processed successfully!")
                logger.info(
                    f"Document processed: {uploaded_file.name} | "
                    f"pages={st.session_state.doc_info['pages']} | "
                    f"chunks={len(chunks)}"
                )
 
            except ValueError as e:
                progress.empty()
                st.error(f"Processing error: {e}")
                logger.error(f"ValueError during processing: {e}")
 
            except RuntimeError as e:
                progress.empty()
                st.error(f"Processing error: {e}")
                logger.error(f"RuntimeError during processing: {e}")
 
    # --- Document Info Panel ---
    if st.session_state.doc_info:
        info = st.session_state.doc_info
        st.divider()
        st.subheader("📊 Document Info")
        st.markdown(f"**File:** {info['filename']}")
        col1, col2 = st.columns(2)
        col1.metric("Pages", info["pages"])
        col2.metric("Chunks", info["chunks"])
        st.caption(f"File size: {info['size_kb']} KB")
 
    # --- Reset Button ---
    st.divider()
    if st.button("🗑️ Clear & Reset", use_container_width=True):
        for key in ["vectorstore", "chat_history", "doc_info", "processing_done"]:
            st.session_state[key] = None if key in ("vectorstore", "doc_info") else []
        st.session_state.processing_done = False
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Session cleared. Upload a PDF to get started!"}
        ]
        st.rerun()
 
# ---------------------------------------------------------------------------
# Main Area — Chat Interface
# ---------------------------------------------------------------------------
 
# Display all messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
        # Re-render source expanders for assistant messages that have them
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📎 Source Chunks Used"):
                st.markdown(msg["sources"])
 
# Chat input (disabled until document is processed)
placeholder_text = (
    "Ask a question about your document…"
    if st.session_state.processing_done
    else "Upload and process a PDF first…"
)
 
if prompt := st.chat_input(placeholder_text, disabled=not st.session_state.processing_done):
 
    # --- Guard: API key ---
    key_valid, key_err = validate_api_key(api_key)
    if not key_valid:
        st.error(key_err)
        st.stop()
 
    # --- Guard: vector store ---
    if st.session_state.vectorstore is None:
        st.error("Please upload and process a PDF document first.")
        st.stop()
 
    # --- Guard: query ---
    query_valid, query_err = validate_query(prompt)
    if not query_valid:
        st.warning(query_err)
        st.stop()
 
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
 
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                retriever = get_retriever(st.session_state.vectorstore)
                rag_chain = build_rag_chain(retriever, api_key)
 
                result = run_query(
                    rag_chain=rag_chain,
                    question=prompt,
                    chat_history=st.session_state.chat_history,
                )
 
                answer: str = result["answer"]
                source_docs = result["sources"]
 
                # Display answer
                st.markdown(answer)
 
                # Display source chunks in collapsible expander (SRS §7.3)
                sources_formatted = format_retrieved_chunks(source_docs)
                with st.expander("📎 Source Chunks Used"):
                    st.markdown(f"```\n{sources_formatted}\n```")
 
                # Persist to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": f"```\n{sources_formatted}\n```",
                })
 
                # Update LangChain conversation history
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=answer),
                ])
 
            except RuntimeError as e:
                error_str = str(e)
 
                # Parse structured error codes from llm_chain.run_query
                if "rate_limit_error" in error_str:
                    msg = error_str.split("|", 1)[-1]
                    st.error(f"⏳ {msg}")
                    # Countdown timer for rate limit
                    countdown = st.empty()
                    for i in range(60, 0, -1):
                        countdown.caption(f"Retry in {i}s…")
                        time.sleep(1)
                    countdown.empty()
 
                elif "timeout_error" in error_str:
                    msg = error_str.split("|", 1)[-1]
                    st.error(f"⌛ {msg}")
 
                elif "auth_error" in error_str:
                    msg = error_str.split("|", 1)[-1]
                    st.error(f"🔑 {msg}")
 
                else:
                    st.error("An unexpected error occurred. Please try again.")
                    logger.error(f"Unhandled RuntimeError in chat: {e}")
 
            except Exception as e:
                st.error("An unexpected error occurred. Please try again.")
                logger.error(f"Unhandled exception in chat interface: {e}")
 