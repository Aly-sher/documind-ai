from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
 
from utils.logger import logger
 
# SRS §4.1 Phase B: top-5 chunks retrieved per query
DEFAULT_K: int = 5
 
 
def get_retriever(
    vector_store: FAISS,
    k: int = DEFAULT_K,
) -> VectorStoreRetriever:
    """
    Configure the FAISS vector store as a LangChain retriever.
 
    Args:
        vector_store: An initialised FAISS vector store (from vector_store.py).
                      Must not be None — call this only after a document is processed.
        k:            Number of top-similar chunks to retrieve per query (default 5).
 
    Returns:
        VectorStoreRetriever configured for similarity search.
 
    Raises:
        ValueError: If vector_store is None (document not yet processed).
    """
    if vector_store is None:
        raise ValueError(
            "Vector store is not initialised. "
            "Please upload and process a document before querying."
        )
 
    logger.info(f"Configuring retriever — search_type='similarity', k={k}.")
 
    return vector_store.as_retriever(
        search_type="similarity",   # Explicit — uses L2/cosine distance
        search_kwargs={"k": k},
    )
 
 
__all__ = ["get_retriever", "DEFAULT_K"]
 