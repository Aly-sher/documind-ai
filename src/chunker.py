from typing import List, Dict
 
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
from utils.logger import logger
 
# ---------------------------------------------------------------------------
# Token-counting length function
# ---------------------------------------------------------------------------
 
_TOKENIZER = tiktoken.get_encoding("cl100k_base")  # Same tokenizer as GPT-4 / embeddings
 
 
def _token_length(text: str) -> int:
    """Return the number of tokens in `text` using the cl100k_base tokenizer."""
    return len(_TOKENIZER.encode(text))
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def create_chunks(
    documents: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """
    Split a list of page-level document dicts into smaller, overlapping chunks.
 
    Args:
        documents:     Output of pdf_processor.process_pdf — list of dicts with
                       keys "text" (str) and "metadata" (dict).
        chunk_size:    Maximum chunk size in tokens (default 500 per SRS §5.4).
        chunk_overlap: Token overlap between adjacent chunks (default 50 per SRS §5.4).
 
    Returns:
        List of chunk dicts, each containing:
          "text"     → chunk text (str)
          "metadata" → inherited page metadata + "chunk_index" (int, 0-based)
 
    Raises:
        ValueError: If documents list is empty or malformed.
    """
    if not documents:
        raise ValueError("documents list is empty — nothing to chunk.")
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],  # Respect paragraph → sentence → word hierarchy
        length_function=_token_length,            # Token-based, not character-based
    )
 
    chunks: List[Dict] = []
    chunk_index: int = 0
 
    for doc in documents:
        # Guard against malformed input dicts
        if "text" not in doc or "metadata" not in doc:
            logger.warning(f"Skipping malformed document dict (missing 'text' or 'metadata'): {doc}")
            continue
 
        page_text: str = doc["text"]
        page_metadata: dict = doc["metadata"]
 
        if not page_text.strip():
            logger.debug(f"Skipping blank page — source: {page_metadata.get('source')}, "
                         f"page: {page_metadata.get('page')}")
            continue
 
        raw_chunks: List[str] = splitter.split_text(page_text)
 
        for chunk_text in raw_chunks:
            # Drop empty or whitespace-only chunks
            if not chunk_text.strip():
                continue
 
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **page_metadata,           # Inherit all page metadata (source, page, total_pages)
                    "chunk_index": chunk_index, # Global chunk index for ordering / debugging
                },
            })
            chunk_index += 1
 
    if not chunks:
        raise ValueError(
            "Chunking produced zero chunks. The document may contain no processable text."
        )
 
    logger.info(
        f"Chunking complete: {len(documents)} page(s) → {len(chunks)} chunk(s) "
        f"(size={chunk_size} tokens, overlap={chunk_overlap} tokens)."
    )
    return chunks
 
 
__all__ = ["create_chunks"]
 