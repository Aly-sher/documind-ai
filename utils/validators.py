"""
utils/validators.py
-------------------
Input validation functions for DocuMind AI.
Every user-facing input is validated here before entering the processing pipeline.
Covers all error cases defined in SRS Section 8.
"""

from typing import Tuple
from utils.logger import logger

# Constants matching SRS Section 8
MAX_FILE_SIZE_MB: int = 50
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS: set = {".pdf"}


def validate_uploaded_file(file) -> Tuple[bool, str]:
    """
    Validates a Streamlit UploadedFile object before processing.

    Checks:
    - File is not None
    - File extension is .pdf
    - File size is within the 50MB limit

    Returns:
        (is_valid: bool, error_message: str)
        error_message is empty string when is_valid is True.
    """
    if file is None:
        return False, "No file uploaded."

    filename: str = file.name.lower()
    if not filename.endswith(".pdf"):
        logger.warning(f"Invalid file type uploaded: {file.name}")
        return False, "Please upload a PDF file only."

    # Streamlit exposes size via .size attribute
    file_size: int = file.size
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        logger.warning(f"File too large: {size_mb:.1f}MB — limit is {MAX_FILE_SIZE_MB}MB")
        return False, f"File too large ({size_mb:.1f} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB."

    if file_size == 0:
        return False, "The uploaded file is empty."

    logger.info(f"File validation passed: {file.name} ({file_size / 1024:.1f} KB)")
    return True, ""


def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validates a user's chat query before it enters the RAG pipeline.

    Checks:
    - Query is not None or empty
    - Query is not purely whitespace
    - Query length is reasonable (under 2000 characters)

    Returns:
        (is_valid: bool, error_message: str)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty."

    if len(query.strip()) > 2000:
        return False, "Query is too long. Please keep questions under 2000 characters."

    return True, ""


def validate_api_key(api_key: str | None) -> Tuple[bool, str]:
    """
    Validates that the Groq API key is present and plausibly formatted.

    Returns:
        (is_valid: bool, error_message: str)
    """
    if not api_key or not api_key.strip():
        logger.error("GROQ_API_KEY is missing from environment.")
        return False, "API key not configured. Add GROQ_API_KEY to your .env file or Streamlit Secrets."

    # Groq keys always start with "gsk_"
    if not api_key.strip().startswith("gsk_"):
        logger.warning("API key present but does not match expected Groq format.")
        return False, "API key appears invalid. Groq API keys begin with 'gsk_'."

    return True, ""


__all__ = [
    "validate_uploaded_file",
    "validate_query",
    "validate_api_key",
    "MAX_FILE_SIZE_MB",
]