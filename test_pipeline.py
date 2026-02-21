"""
test_pipeline.py
----------------
Unit tests for the FinanceRAG pipeline components.
Run with: pytest test_pipeline.py -v
"""

import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
from langchain.schema import Document

from rag_pipeline import DocumentProcessor, FinanceRAGPipeline


# ──────────────────────────────────────────────
# DOCUMENT PROCESSOR TESTS
# ──────────────────────────────────────────────
class TestDocumentProcessor:

    def setup_method(self):
        self.processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

    def test_load_raw_text_returns_documents(self):
        text = "Apple Inc. reported revenue of $383 billion for fiscal year 2023. " * 20
        docs = self.processor.load_raw_text(text, source_name="test_source")
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_load_raw_text_source_metadata(self):
        docs = self.processor.load_raw_text("Revenue: $10B", source_name="apple_10k")
        assert docs[0].metadata["source"] == "apple_10k"

    def test_chunk_size_respected(self):
        long_text = "Tesla earnings increased by 20% year over year. " * 100
        docs = self.processor.load_raw_text(long_text)
        for doc in docs:
            assert len(doc.page_content) <= 600  # chunk_size + some buffer

    def test_load_text_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Microsoft Q4 2023 revenue was $56.2 billion.\n" * 30)
            tmp_path = f.name
        try:
            docs = self.processor.load_file(tmp_path)
            assert len(docs) > 0
        finally:
            os.unlink(tmp_path)

    def test_unsupported_file_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            self.processor.load_file("report.xlsx")


# ──────────────────────────────────────────────
# PIPELINE TESTS (mocked OpenAI)
# ──────────────────────────────────────────────
class TestFinanceRAGPipeline:

    @patch("rag_pipeline.OpenAIEmbeddings")
    @patch("rag_pipeline.ChatOpenAI")
    def test_pipeline_initializes(self, mock_llm, mock_emb):
        pipeline = FinanceRAGPipeline(api_key="sk-test-key")
        assert pipeline.document_count == 0
        assert pipeline.qa_chain is None

    @patch("rag_pipeline.OpenAIEmbeddings")
    @patch("rag_pipeline.ChatOpenAI")
    @patch("rag_pipeline.FAISS")
    def test_ingest_text_updates_count(self, mock_faiss, mock_llm, mock_emb):
        mock_faiss.from_documents.return_value = MagicMock()
        pipeline = FinanceRAGPipeline(api_key="sk-test-key")

        long_text = "Revenue grew 15% in Q4 2023. Net income was $2.3B. " * 50
        chunks = pipeline.ingest_text(long_text, "test_report")
        assert chunks > 0
        assert pipeline.document_count == chunks

    @patch("rag_pipeline.OpenAIEmbeddings")
    @patch("rag_pipeline.ChatOpenAI")
    def test_query_without_documents_raises(self, mock_llm, mock_emb):
        pipeline = FinanceRAGPipeline(api_key="sk-test-key")
        with pytest.raises(ValueError, match="No documents ingested"):
            pipeline.query("What is the revenue?")

    @patch("rag_pipeline.OpenAIEmbeddings")
    @patch("rag_pipeline.ChatOpenAI")
    def test_reset_clears_state(self, mock_llm, mock_emb):
        pipeline = FinanceRAGPipeline(api_key="sk-test-key")
        pipeline.document_count = 100
        pipeline.reset()
        assert pipeline.document_count == 0
        assert pipeline.qa_chain is None


# ──────────────────────────────────────────────
# INTEGRATION TEST (requires real API key)
# ──────────────────────────────────────────────
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Skipping integration test — set OPENAI_API_KEY to run."
)
def test_full_pipeline_integration():
    pipeline = FinanceRAGPipeline(api_key=os.environ["OPENAI_API_KEY"])
    sample_text = """
    Apple Inc. Financial Results Q3 2023:
    Net revenue: $81.8 billion (down 1% year over year)
    iPhone revenue: $39.7 billion
    Services revenue: $21.2 billion (all-time record)
    Net income: $19.9 billion
    Earnings per share (diluted): $1.26
    CEO Tim Cook stated: 'We are pleased to report a June quarter revenue record 
    in Services and quarterly records in several countries.'
    """
    pipeline.ingest_text(sample_text, "apple_q3_2023")
    result = pipeline.query("What was Apple's Services revenue in Q3 2023?")
    assert "21.2" in result["answer"] or "record" in result["answer"].lower()
    assert len(result["sources"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
