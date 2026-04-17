import streamlit as st
import os
import logging
from dotenv import load_dotenv

# Import our custom modules
from src.pdf_processor import process_pdf
from src.chunker import create_chunks
from src.embedder import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.llm_chain import get_llm
from src.prompt_builder import get_qa_prompt, format_retrieved_chunks

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

# --- Session State Initialization ---
# Senior Tip: We must initialize these keys so Streamlit remembers them across reruns.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False

# --- Core Functions ---
def reset_session():
    """Clears the session state for a fresh start."""
    st.session_state.chat_history = []
    st.session_state.retriever = None
    st.session_state.is_processed = False
    st.rerun()

def handle_document_upload(uploaded_file):
    """Orchestrates the entire Phase A pipeline (Ingestion to Indexing)."""
    with st.status("Processing Document...", expanded=True) as status:
        try:
            # 1. Extraction
            st.write("Extracting text from PDF...")
            file_bytes = uploaded_file.read()
            docs, msg = process_pdf(file_bytes, uploaded_file.name)
            
            if msg != "success":
                status.update(label=f"Error: {msg}", state="error")
                st.error(f"Failed to process PDF. Reason: {msg}")
                return

            # 2. Chunking
            st.write(f"Splitting {len(docs)} pages into chunks...")
            chunks = create_chunks(docs)
            
            # 3. Embedding & Indexing
            st.write("Generating embeddings and building FAISS index...")
            embedding_model = get_embedding_model()
            vector_store = create_vector_store(chunks, embedding_model)
            
            # 4. Save Retriever to Memory
            st.session_state.retriever = get_retriever(vector_store)
            st.session_state.is_processed = True
            
            status.update(label="Document processed successfully!", state="complete")
            
        except Exception as e:
                    st.error(f"SYSTEM CRASH REPORT: {str(e)}")
# --- UI Layout: Sidebar ---
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF to begin", type=["pdf"])
    
    if uploaded_file and not st.session_state.is_processed:
        # File size validation (Max 50MB as per SRS)
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("File too large. Max size is 50MB.")
        else:
            if st.button("Process Document", type="primary"):
                handle_document_upload(uploaded_file)
                
    if st.session_state.is_processed:
        st.success(f"Active Document: {uploaded_file.name}")
        if st.button("Clear & Upload New File"):
            reset_session()

# --- UI Layout: Main Area ---
st.title("🧠 DocuMind AI")
st.caption("An Intelligent RAG-Powered Document Q&A System powered by Llama 3")

# Display state when no document is ready
if not st.session_state.is_processed:
    st.info("👈 Please upload and process a PDF document in the sidebar to start asking questions.")

# Display Chat Interface when ready
else:
    # 1. Render Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # If the assistant message has citations, show them in an expander
            if "sources" in message and message["sources"]:
                with st.expander("View Source Chunks"):
                    st.text(message["sources"])

    # 2. Chat Input Box
    if prompt := st.chat_input("Ask a question about your document..."):
        
        # Display user question immediately
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Append to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                try:
                    # Phase B: Query Pipeline
                    
                    # 1. Retrieve relevant chunks
                    raw_docs = st.session_state.retriever.invoke(prompt)
                    formatted_context = format_retrieved_chunks(raw_docs)
                    
                    # 2. Limit Chat History (Context Bleeding mitigation)
                    # Grab only the last 3 user/assistant pairs (6 messages total)
                    recent_history = st.session_state.chat_history[-6:-1] 
                    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
                    
                    # 3. Build Prompt & Call LLM
                    llm = get_llm()
                    qa_prompt = get_qa_prompt()
                    
                    # We format the prompt with our variables
                    messages = qa_prompt.format_messages(
                        context=formatted_context,
                        chat_history=history_str,
                        input=prompt
                    )
                    
                    # 4. Get Answer
                    response = llm.invoke(messages)
                    answer_text = response.content
                    
                    # Display Answer
                    st.markdown(answer_text)
                    with st.expander("View Source Chunks"):
                        st.text(formatted_context)
                        
                    # 5. Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": answer_text,
                        "sources": formatted_context
                    })
                    
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    st.error("Failed to generate a response. Please try again or check your API key limit.")