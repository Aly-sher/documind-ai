"""
Document health analysis for DocuMind AI.
Runs after PDF extraction to surface quality metrics in the sidebar.
"""

from __future__ import annotations
from typing import Any, Dict, List
import re

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_document_health(
    documents: List[Any],
    chunks: List[Any],
) -> Dict[str, Any]:
    """
    Analyse extracted document content and return modern quality metrics.
    """
    if not documents:
        return {}

    total_pages = len(documents)

    # ── Extract raw text per page ────────────────────────────────────────
    page_texts: List[str] = []
    for doc in documents:
        if hasattr(doc, "page_content"):
            page_texts.append(doc.page_content)
        elif isinstance(doc, dict):
            page_texts.append(doc.get("page_content", doc.get("content", "")))
        else:
            page_texts.append(str(doc))

    total_chars = sum(len(t) for t in page_texts)
    total_words = sum(len(t.split()) for t in page_texts)

    # ── Reading time  (avg 200 words / min) ─────────────────────────────
    reading_time_mins = max(1, round(total_words / 200))

    if total_chars == 0:
        # Failsafe for completely empty image-based PDFs
        return _build_empty_response(total_pages, len(chunks))

    # ── 1. Page Coverage Score ──────────────────────────────────────────
    # Check for empty/sparse pages (lowered threshold to respect CV formatting)
    SPARSE_THRESHOLD = 30
    sparse_pages = [i + 1 for i, t in enumerate(page_texts) if len(t.strip()) < SPARSE_THRESHOLD]
    sparse_count = len(sparse_pages)
    valid_pages = total_pages - sparse_count
    page_score = (valid_pages / total_pages) * 100

    # ── 2. Text Quality (Gibberish & Encoding Check) ────────────────────
    # Good PDFs have actual letters/numbers. Broken PDFs have raw symbols or CID tags.
    combined_text = " ".join(page_texts)
    alpha_num_count = len(re.findall(r'\w', combined_text))
    cid_count = len(re.findall(r'\(cid:\d+\)', combined_text))

    if cid_count > 10:
        # PyMuPDF throws massive CID tags when font encoding is completely scrambled
        quality_score = 0
    else:
        # Standard English/Urdu text is roughly 65-80% alphanumeric (rest is spaces/punctuation)
        ratio = alpha_num_count / total_chars
        if ratio > 0.60:
            quality_score = 100
        elif ratio > 0.40:
            quality_score = 70
        else:
            quality_score = max(0, int(ratio * 100))

    # ── 3. Chunk Health Score ───────────────────────────────────────────
    # Modern LLMs like Llama 3.1 love large chunks. We only penalize if chunks are tiny fragments.
    chunk_score = 0
    if chunks:
        avg_chunk_size = sum(
            len(c.page_content if hasattr(c, "page_content") else c.get("page_content", "")) 
            for c in chunks
        ) / len(chunks)
        
        if avg_chunk_size > 100:
            chunk_score = 100
        elif avg_chunk_size > 30:
            chunk_score = 50

    # ── Final Document Health Score ──────────────────────────────────────
    # 50% Text Quality, 30% Page Coverage, 20% Chunk Structure
    health_score = round((quality_score * 0.5) + (page_score * 0.3) + (chunk_score * 0.2), 1)

    return {
        "total_pages":        total_pages,
        "total_words":        total_words,
        "total_chars":        total_chars,
        "reading_time_mins":  reading_time_mins,
        "sparse_pages":       sparse_pages,
        "sparse_count":       sparse_count,
        "image_only_risk":    cid_count > 10 or (sparse_count == total_pages),
        "text_coverage_pct":  round(page_score, 1),
        "avg_chars_per_page": round(total_chars / total_pages) if total_pages else 0,
        "total_chunks":       len(chunks),
        "chunk_quality_score": chunk_score,
        "health_score":       health_score,
    }


def _build_empty_response(total_pages: int, chunk_count: int) -> Dict[str, Any]:
    """Helper to return a 0 health score if the document contains absolutely no text."""
    return {
        "total_pages": total_pages, "total_words": 0, "total_chars": 0,
        "reading_time_mins": 0, "sparse_pages": list(range(1, total_pages + 1)),
        "sparse_count": total_pages, "image_only_risk": True,
        "text_coverage_pct": 0.0, "avg_chars_per_page": 0,
        "total_chunks": chunk_count, "chunk_quality_score": 0.0, "health_score": 0.0,
    }


def health_emoji(score: float) -> str:
    """Return a colour-coded emoji for the health score."""
    if score >= 80:
        return "🟢"
    if score >= 55:
        return "🟡"
    return "🔴"


def health_label(score: float) -> str:
    """Return a plain-English label for the health score."""
    if score >= 80:
        return "Good"
    if score >= 55:
        return "Fair"
    return "Poor"
