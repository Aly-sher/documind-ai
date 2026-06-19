from langchain_community.embeddings import HuggingFaceEmbeddings
 
from utils.logger import logger
 
# The SRS-specified embedding model (sentence-transformers prefix required for HuggingFace hub)
_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
 
# Known output dimensionality — used for downstream validation
EMBEDDING_DIMENSION: int = 384
 
 
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource
def get_embedding_model(model_name: str = _MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Initialise and return the local sentence-transformer embedding model.
    
    Optimised via Streamlit's cache_resource to load exactly once into RAM 
    and prevent context re-initialisation bottlenecks during app re-runs.

    The model is downloaded on first run (~90 MB) and cached by HuggingFace
    in ~/.cache/huggingface/. Subsequent loads are instant.

    Args:
        model_name: HuggingFace model identifier. Defaults to all-MiniLM-L6-v2.

    Returns:
        HuggingFaceEmbeddings instance ready for use with FAISS.

    Raises:
        RuntimeError: If the model fails to load for any reason.
    """
    try:
        # Standard configuration for local CPU processing
        model_kwargs = {'device': 'cpu'} 
        encode_kwargs = {'normalize_embeddings': True}
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{model_name}': {str(e)}") from e

__all__ = ["get_embedding_model", "EMBEDDING_DIMENSION"]