from langchain_community.embeddings import HuggingFaceEmbeddings
 
from utils.logger import logger
 
# The SRS-specified embedding model (sentence-transformers prefix required for HuggingFace hub)
_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
 
# Known output dimensionality — used for downstream validation
EMBEDDING_DIMENSION: int = 384
 
 
def get_embedding_model(model_name: str = _MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Initialise and return the local sentence-transformer embedding model.
 
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
        logger.info(f"Loading local embedding model: '{model_name}' (CPU)")
 
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # L2-normalised → cosine similarity via dot product
        )
 
        # Smoke-test: embed a single string and verify output dimensionality
        test_vector = embeddings.embed_query("test")
        assert len(test_vector) == EMBEDDING_DIMENSION, (
            f"Expected {EMBEDDING_DIMENSION}-dim vector, got {len(test_vector)}-dim. "
            f"Check model_name: '{model_name}'"
        )
 
        logger.info(
            f"Embedding model loaded successfully. "
            f"Output dimension: {EMBEDDING_DIMENSION}."
        )
        return embeddings
 
    except AssertionError as e:
        logger.error(f"Embedding model dimension mismatch: {e}")
        raise RuntimeError(str(e)) from e
 
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}")
        raise RuntimeError(f"Could not initialise embedding model: {e}") from e
 
 
__all__ = ["get_embedding_model", "EMBEDDING_DIMENSION"]
 