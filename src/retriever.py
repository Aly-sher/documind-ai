import logging

logger = logging.getLogger(__name__)

def get_retriever(vector_store, k: int = 5):
    """
    Configures the FAISS index to act as a retriever.
    Returns the top 'k' most similar chunks based on Euclidean distance.
    """
    logger.info(f"Configuring retriever to fetch top {k} results.")
    
    # The 'search_kwargs' parameter is where we tell FAISS how many chunks 
    # to bring back. 5 chunks of 500 tokens = ~2500 tokens of context, 
    # which fits very comfortably inside Llama 3's context window.
    return vector_store.as_retriever(search_kwargs={"k": k})