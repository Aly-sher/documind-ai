import re
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple
 
import fitz  # PyMuPDF
 
from utils.logger import logger
 
 
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
 
def _normalize_unicode(text: str) -> str:
    """Normalise unicode characters to their closest ASCII equivalents where possible."""
    return unicodedata.normalize("NFKC", text)
 
 
def _remove_null_bytes(text: str) -> str:
    """Strip null bytes that occasionally appear in PDF extractions."""
    return text.replace("\x00", "").replace("\ufffd", "")
 
 
def _remove_standalone_page_numbers(text: str) -> str:
    """
    Remove lines that consist solely of a number (standalone page numbers
    that appear as artefacts from PDF extraction).
    Example: a line containing only '42' or ' 42 ' is removed.
    """
    return re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
 
 
def _normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace and blank lines into single newlines."""
    # Collapse runs of spaces/tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines into 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
 
 
def _detect_repeated_lines(pages_text: List[str], threshold: float = 0.6) -> set:
    """
    Detect header/footer lines that appear on more than `threshold` fraction
    of pages. These are stripped from all pages.
 
    Args:
        pages_text: List of raw page texts.
        threshold:  Fraction of pages a line must appear on to be considered
                    a header/footer (default 60%).
 
    Returns:
        Set of line strings identified as repeated headers/footers.
    """
    if len(pages_text) < 3:
        # Not enough pages for meaningful detection
        return set()
 
    line_counter: Counter = Counter()
    for page_text in pages_text:
        # Only consider short lines (headers/footers are rarely long)
        for line in page_text.splitlines():
            stripped = line.strip()
            if 1 < len(stripped) < 120:
                line_counter[stripped] += 1
 
    repeated: set = set()
    total_pages = len(pages_text)
    for line, count in line_counter.items():
        if count / total_pages >= threshold:
            repeated.add(line)
 
    if repeated:
        logger.debug(f"Detected {len(repeated)} repeated header/footer lines to strip.")
 
    return repeated
 
 
def _clean_page_text(text: str, repeated_lines: set) -> str:
    """
    Apply all cleaning steps to a single page's text in the correct order.
 
    Steps (per SRS Section 5.3):
    1. Remove null bytes and replacement characters
    2. Normalise unicode
    3. Remove repeated header/footer lines
    4. Remove standalone page numbers
    5. Normalise whitespace
    """
    text = _remove_null_bytes(text)
    text = _normalize_unicode(text)
 
    if repeated_lines:
        lines = text.splitlines()
        lines = [ln for ln in lines if ln.strip() not in repeated_lines]
        text = "\n".join(lines)
 
    text = _remove_standalone_page_numbers(text)
    text = _normalize_whitespace(text)
    return text
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def process_pdf(file_bytes: bytes, filename: str) -> Tuple[List[Dict], str]:
    """
    Extract, clean, and structure text from a PDF document.
 
    Args:
        file_bytes: Raw bytes of the uploaded PDF file.
        filename:   Original filename — stored in chunk metadata for citations.
 
    Returns:
        Tuple of:
          - documents (List[Dict]): Each dict has keys:
              "text"     → cleaned page text (str)
              "metadata" → {"source": filename, "page": int (1-indexed)}
          - status (str): One of:
              "success" | "empty_pdf" | "image_only_pdf" |
              "corrupt_pdf" | "unknown_error"
 
    All status strings map directly to the SRS Section 8 error handling table.
    """
    doc = None
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
 
        if doc.page_count == 0:
            logger.warning(f"PDF has 0 pages: {filename}")
            return [], "empty_pdf"
 
        logger.info(f"Processing PDF: '{filename}' — {doc.page_count} page(s)")
 
        # --- Pass 1: extract raw text from every page ---
        raw_pages: List[str] = []
        for page_num in range(doc.page_count):
            raw_pages.append(doc[page_num].get_text())
 
        # --- Detect repeated headers/footers across all pages ---
        repeated_lines = _detect_repeated_lines(raw_pages)
 
        # --- Pass 2: clean and structure ---
        documents: List[Dict] = []
        for page_num, raw_text in enumerate(raw_pages):
            cleaned = _clean_page_text(raw_text, repeated_lines)
            if not cleaned:
                logger.debug(f"Page {page_num + 1} produced no text after cleaning — skipping.")
                continue
 
            documents.append({
                "text": cleaned,
                "metadata": {
                    "source": filename,
                    "page": page_num + 1,  # 1-indexed for human-readable citations
                    "total_pages": doc.page_count,
                },
            })
 
        if not documents:
            logger.warning(f"No extractable text found in '{filename}'. Likely image-only PDF.")
            return [], "image_only_pdf"
 
        total_chars = sum(len(d["text"]) for d in documents)
        logger.info(
            f"Extraction complete: {len(documents)} pages with text, "
            f"{total_chars:,} total characters."
        )
        return documents, "success"
 
    except fitz.FileDataError:
        logger.error(f"Corrupt PDF — could not open '{filename}'.")
        return [], "corrupt_pdf"
 
    except Exception as e:
        logger.error(f"Unexpected error processing '{filename}': {e}")
        return [], "unknown_error"
 
    finally:
        if doc is not None:
            doc.close()
 
 
__all__ = ["process_pdf"]