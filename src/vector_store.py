from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# COMMENT THIS OUT TEMPORARILY
# from utils.logger import logger 

def build_vector_store(
    chunks: List[Dict],
    embedding_model: HuggingFaceEmbeddings,
) -> FAISS:
    # Import it locally inside the function scope instead
    from utils.logger import logger 
    
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