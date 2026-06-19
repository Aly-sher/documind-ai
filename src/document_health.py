"""
Document health analysis for DocuMind AI.
Runs after PDF extraction to surface quality metrics in the sidebar.
"""

from __future__ import annotations
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_document_health(
    documents: List[Any],
    chunks: List[Any],
) -> Dict[str, Any]:
    """
    Analyse extracted document content and return quality metrics.

    Args:
        documents: Page-level Document objects from process_pdf().
        chunks:    Chunk-level Document objects from create_chunks().

    Returns:
        Dict of health metrics consumed by the sidebar renderer in app.py.
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

    # ── Sparse / image-only page detection ──────────────────────────────
    # Pages with fewer than 60 chars are almost certainly image-only
    SPARSE_THRESHOLD = 60
    sparse_pages = [i + 1 for i, t in enumerate(page_texts)
                    if len(t.strip()) < SPARSE_THRESHOLD]
    sparse_count = len(sparse_pages)
    text_coverage_pct = round((1 - sparse_count / total_pages) * 100, 1)

    # ── Chunk quality score ──────────────────────────────────────────────
    # Measures how close average chunk length is to the ideal 400-600 char range.
    chunk_quality_score = 0.0
    if chunks:
        lengths: List[int] = []
        for c in chunks:
            if hasattr(c, "page_content"):
                lengths.append(len(c.page_content))
            elif isinstance(c, dict):
                lengths.append(len(c.get("page_content", "")))

        if lengths:
            avg_len = sum(lengths) / len(lengths)
            ideal   = 500          # midpoint of ideal 400-600 range
            deviation = abs(avg_len - ideal) / ideal
            chunk_quality_score = max(0.0, round((1 - deviation) * 100, 1))

    # ── Overall document health score ────────────────────────────────────
    health_score = round((text_coverage_pct + min(chunk_quality_score, 100)) / 2, 1)

    return {
        "total_pages":        total_pages,
        "total_words":        total_words,
        "total_chars":        total_chars,
        "reading_time_mins":  reading_time_mins,
        "sparse_pages":       sparse_pages,        # list of page numbers
        "sparse_count":       sparse_count,
        "image_only_risk":    sparse_count > 0,
        "text_coverage_pct":  text_coverage_pct,
        "avg_chars_per_page": round(total_chars / total_pages) if total_pages else 0,
        "total_chunks":       len(chunks),
        "chunk_quality_score": chunk_quality_score,
        "health_score":       health_score,
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
