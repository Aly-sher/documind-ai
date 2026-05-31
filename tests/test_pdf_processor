"""
tests/test_pdf_processor.py
----------------------------
Unit tests for src/pdf_processor.py
"""

import pytest
from src.pdf_processor import process_pdf, _clean_page_text, _detect_repeated_lines


class TestCleanPageText:
    def test_removes_null_bytes(self):
        dirty = "Hello\x00World"
        result = _clean_page_text(dirty, set())
        assert "\x00" not in result

    def test_normalizes_whitespace(self):
        dirty = "Hello    \n\n\n\n   World"
        result = _clean_page_text(dirty, set())
        assert "\n\n\n" not in result

    def test_removes_standalone_page_numbers(self):
        text = "Some text\n42\nMore text"
        result = _clean_page_text(text, set())
        assert "42" not in result.split()  # 42 should be gone as standalone line

    def test_removes_repeated_lines(self):
        repeated = {"DocuMind AI — Confidential"}
        text = "DocuMind AI — Confidential\nActual content here."
        result = _clean_page_text(text, repeated)
        assert "DocuMind AI — Confidential" not in result
        assert "Actual content" in result


class TestDetectRepeatedLines:
    def test_detects_header_on_majority_of_pages(self):
        pages = [
            "Header Text\nPage content one.",
            "Header Text\nPage content two.",
            "Header Text\nPage content three.",
            "Header Text\nPage content four.",
        ]
        repeated = _detect_repeated_lines(pages, threshold=0.6)
        assert "Header Text" in repeated

    def test_ignores_unique_lines(self):
        pages = [
            "Unique line one\nShared content.",
            "Unique line two\nShared content.",
            "Unique line three\nShared content.",
        ]
        repeated = _detect_repeated_lines(pages, threshold=0.6)
        assert "Unique line one" not in repeated

    def test_returns_empty_for_few_pages(self):
        pages = ["Page one content.", "Page two content."]
        repeated = _detect_repeated_lines(pages)
        assert repeated == set()


class TestProcessPdf:
    def test_returns_corrupt_pdf_status_for_bad_bytes(self):
        bad_bytes = b"this is not a pdf"
        docs, status = process_pdf(bad_bytes, "fake.pdf")
        assert status == "corrupt_pdf"
        assert docs == []

    def test_returns_list_and_success_for_valid_pdf(self, tmp_path):
        """Integration-style test using a minimal real PDF created by fitz."""
        import fitz
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "This is a test document page one.")
        doc.save(str(pdf_path))
        doc.close()

        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        docs, status = process_pdf(file_bytes, "test.pdf")
        assert status == "success"
        assert len(docs) >= 1
        assert "text" in docs[0]
        assert docs[0]["metadata"]["page"] == 1
        assert docs[0]["metadata"]["source"] == "test.pdf"