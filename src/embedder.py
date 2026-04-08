from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Initializes the local sentence-transformer model.
    Runs entirely on CPU to ensure compatibility across all deployment environments.
    """
    try:
        # For a portfolio project deployed on Streamlit Cloud, forcing CPU is safer
        # as cloud instances usually don't have GPUs unless specifically provisioned.
        model_kwargs = {'device': 'cpu'}
        
        # Senior Tip: Normalizing embeddings maps all vectors to a unit sphere.
        # This makes the L2 distance calculation in FAISS mathematically equivalent 
        # to Cosine Similarity, which generally yields much better search results!
        encode_kwargs = {'normalize_embeddings': True} 
        
        logger.info(f"Loading local embedding model: {model_name}")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise e