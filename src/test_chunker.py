"""
tests/test_chunker.py
----------------------
Unit tests for src/chunker.py
"""

import pytest
from src.chunker import create_chunks


def _make_docs(texts):
    """Helper to create minimal document dicts for testing."""
    return [
        {"text": t, "metadata": {"source": "test.pdf", "page": i + 1, "total_pages": len(texts)}}
        for i, t in enumerate(texts)
    ]


class TestCreateChunks:
    def test_returns_chunks_for_valid_input(self):
        docs = _make_docs(["This is a sentence. " * 50])
        chunks = create_chunks(docs)
        assert len(chunks) >= 1

    def test_each_chunk_has_text_and_metadata(self):
        docs = _make_docs(["Sample content for testing. " * 30])
        chunks = create_chunks(docs)
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["text"].strip() != ""

    def test_metadata_inherits_page_number(self):
        docs = _make_docs(["Page one text. " * 30, "Page two text. " * 30])
        chunks = create_chunks(docs)
        pages_seen = {c["metadata"]["page"] for c in chunks}
        assert 1 in pages_seen
        assert 2 in pages_seen

    def test_chunk_index_is_sequential(self):
        docs = _make_docs(["Word. " * 100])
        chunks = create_chunks(docs)
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_raises_on_empty_documents(self):
        with pytest.raises(ValueError):
            create_chunks([])

    def test_raises_when_all_pages_blank(self):
        docs = _make_docs(["   ", "\n\n"])
        with pytest.raises(ValueError):
            create_chunks(docs)

    def test_no_empty_chunks_produced(self):
        docs = _make_docs(["Real content here. " * 20])
        chunks = create_chunks(docs)
        for chunk in chunks:
            assert chunk["text"].strip() != ""