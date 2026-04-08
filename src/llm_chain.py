import os
import logging
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

def get_llm(model_name: str = "llama3-8b-8192", temperature: float = 0.0):
    """
    Initializes the Groq LLM client.
    """
    # Defensive check: ensure the API key is loaded before trying to connect
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not found.")
        raise ValueError("GROQ_API_KEY is required in the environment or .env file.")

    try:
        logger.info(f"Initializing Groq LLM with model: {model_name}")
        
        # Senior Tip: Setting temperature to 0.0 is critical for RAG. 
        # We want factual extraction, not creative improvisation.
        llm = ChatGroq(
            temperature=temperature,
            model_name=model_name,
            groq_api_key=api_key,
            max_tokens=1024  # Cap the response length to keep it concise
        )
        return llm
        
    except Exception as e:
        logger.error(f"Failed to initialize Groq LLM: {e}")
        raise e