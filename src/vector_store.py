from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def create_vector_store(chunks: List[Dict], embedding_model) -> FAISS:
    """
    Ingests text chunks, embeds them, and creates a FAISS vector index.
    """
    try:
        # LangChain's FAISS wrapper expects standard 'Document' objects, 
        # so we quickly map our dictionaries to the expected format.
        docs = [
            Document(page_content=chunk["text"], metadata=chunk["metadata"]) 
            for chunk in chunks
        ]
        
        logger.info(f"Creating FAISS index with {len(docs)} chunks...")
        
        # This single line handles passing all text through the embedding model 
        # and building the mathematical index.
        vector_store = FAISS.from_documents(docs, embedding_model)
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to create FAISS vector store: {e}")
        raise e