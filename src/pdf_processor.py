import fitz  # PyMuPDF
import re
import logging
from typing import List, Dict, Tuple

# We'll use standard logging for now, upgrade to loguru later as per SRS
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Removes null bytes and normalizes whitespace."""
    text = text.replace('\x00', '')
    # Condense multiple whitespaces/newlines into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_pdf(file_bytes: bytes, filename: str) -> Tuple[List[Dict], str]:
    """
    Extracts text from a PDF stream and attaches page metadata.
    Returns a tuple of (documents_list, status_message).
    """
    try:
        # Senior Tip: Opening from a memory stream means we don't have to save 
        # Streamlit uploads to the disk first. It's faster and safer.
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        if doc.page_count == 0:
            return [], "empty_pdf"
            
        documents = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            
            # Only process pages that actually contain text
            if text.strip(): 
                cleaned_text = clean_text(text)
                
                # Crucial Step: Tagging the metadata for our citation feature
                metadata = {
                    "source": filename,
                    "page": page_num + 1  # 1-indexed so it makes sense to users
                }
                documents.append({"text": cleaned_text, "metadata": metadata})
                
        if not documents:
            return [], "image_only_pdf"
            
        return documents, "success"
        
    except fitz.FileDataError:
        logger.error(f"Failed to read file: {filename}. Corrupt PDF.")
        return [], "corrupt_pdf"
    except Exception as e:
        logger.error(f"Unexpected error processing {filename}: {e}")
        return [], "unknown_error"