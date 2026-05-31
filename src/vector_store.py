from typing import List, Dict
 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
 
from utils.logger import logger
 
 
def build_vector_store(
    chunks: List[Dict],
    embedding_model: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Build a FAISS vector store from a list of chunk dicts.
 
    Converts each chunk dict into a LangChain Document (preserving metadata),
    then embeds all documents and indexes them in FAISS.
 
    Args:
        chunks:          Output of chunker.create_chunks — list of dicts with
                         "text" and "metadata" keys.
        embedding_model: Initialised HuggingFaceEmbeddings instance from embedder.py.
 
    Returns:
        FAISS vector store, ready for similarity search via .as_retriever().
 
    Raises:
        ValueError: If chunks list is empty.
        RuntimeError: If FAISS index construction fails.
    """
    if not chunks:
        raise ValueError("Cannot build vector store from empty chunks list.")
 
    # Convert chunk dicts → LangChain Document objects
    documents: List[Document] = [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        for chunk in chunks
    ]
 
    logger.info(f"Building FAISS index from {len(documents)} chunk(s)...")
 
    try:
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model,
        )
        logger.info("FAISS index built successfully.")
        return vector_store
 
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        raise RuntimeError(f"Vector store construction failed: {e}") from e
 
 
__all__ = ["build_vector_store"]