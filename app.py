import os
import logging
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Custom Imports (Your existing files)
from src.pdf_processor import process_pdf
from src.chunker import create_chunks

# --- Setup & Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="centered")

# --- Secure API Key Handling ---
# 1. Try to load from local .env file first
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 2. If no local key is found, safely fallback to Streamlit Cloud Secrets
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

# Ensure the key is set in the environment for LangChain to detect
if api_key:
    os.environ["GROQ_API_KEY"] = api_key

# --- Initialize Session States ---
# Changed to [] to fix the "class 'str'" error
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload a PDF, and ask me anything about it!"}]

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Application Header ---
st.title("📄 DocuMind AI")
st.markdown("Powered by Llama 3 (Groq) & FAISS Vector Search")

# --- Sidebar: File Processing ---
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Extracting and processing text..."):
            try:
                # 1. Extract text using your custom module
                raw_text = process_pdf(uploaded_file)
                
                # 2. Chunk text using your custom module
                chunks = create_chunks(raw_text)
                
                # 3. Create Embeddings & Vectorstore
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
                
                # Save to session state
                st.session_state.vectorstore = vectorstore
                st.success("Document processed successfully! You can now ask questions.")
            
            except Exception as e:
                logger.error(f"Processing Error: {e}")
                st.error(f"Error processing PDF: {str(e)}")

# --- Main Chat Interface ---
# Display all previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your document..."):
    
    # 1. Block query if API key is missing
    if not api_key:
        st.error("GROQ API Key is missing. Please add it to your secrets.")
        st.stop()
        
    # 2. Block query if no document is processed
    if st.session_state.vectorstore is None:
        st.error("Please upload and process a PDF document first.")
        st.stop()

    # 3. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 4. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Setup LLM
                llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
                
                # Setup Retriever
                retriever = st.session_state.vectorstore.as_retriever()
                
                # Formulate contextualized question prompt
                contextualize_q_system_prompt = (
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return it as is."
                )
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )
                
                # Formulate Q&A prompt
                qa_system_prompt = (
                    "You are a helpful AI assistant. Use the following pieces of retrieved "
                    "context to answer the question. If you don't know the answer, just say "
                    "that you don't know. Be comprehensive but clear."
                    "\n\n"
                    "{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                
                # Build the final chains
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                
                # Execute the chain using LangChain's native history format
                response = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                
                answer = response["answer"]
                st.write(answer)
                
                # Append to UI messages
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Append strictly typed LangChain message objects to internal memory
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=answer)
                ])
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                st.error(f"SYSTEM CRASH REPORT: {str(e)}")