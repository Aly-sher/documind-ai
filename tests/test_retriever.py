"""
tests/test_retriever.py
------------------------
Unit tests for src/retriever.py
"""

import pytest
from src.retriever import get_retriever


class TestGetRetriever:
    def test_raises_on_none_vector_store(self):
        with pytest.raises(ValueError, match="not initialised"):
            get_retriever(None)

    def test_returns_retriever_for_valid_store(self):
        """Integration test — requires embedding model; skipped in CI without it."""
        pytest.importorskip("sentence_transformers")

        from src.embedder import get_embedding_model
        from src.vector_store import build_vector_store

        chunks = [
            {"text": "The capital of France is Paris.", "metadata": {"source": "test.pdf", "page": 1, "total_pages": 1, "chunk_index": 0}},
            {"text": "The Eiffel Tower is in Paris.", "metadata": {"source": "test.pdf", "page": 1, "total_pages": 1, "chunk_index": 1}},
        ]
        model = get_embedding_model()
        store = build_vector_store(chunks, model)
        retriever = get_retriever(store, k=2)
        assert retriever is not None