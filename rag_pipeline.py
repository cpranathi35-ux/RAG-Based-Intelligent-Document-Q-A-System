"""
rag_pipeline.py
---------------
Core RAG (Retrieval-Augmented Generation) pipeline for Finance Q&A.
Handles document ingestion, embedding, vector storage, and LLM querying.
"""

import os
import re
from typing import List, Tuple
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


# ──────────────────────────────────────────────
# FINANCE-SPECIFIC PROMPT TEMPLATE
# ──────────────────────────────────────────────
FINANCE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a financial analyst assistant specialized in analyzing stock reports, 
earnings statements, SEC filings, and financial documents.

Use ONLY the context below to answer the question. If the answer is not in 
the context, say "I could not find this information in the provided documents."
Always cite specific figures, dates, or sections when available.

Context:
{context}

Question: {question}

Answer (be precise and cite data where possible):
"""
)


# ──────────────────────────────────────────────
# DOCUMENT PROCESSOR
# ──────────────────────────────────────────────
class DocumentProcessor:
    """Loads and chunks financial documents for embedding."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return self.splitter.split_documents(pages)

    def load_text(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        return self.splitter.split_documents(docs)

    def load_raw_text(self, text: str, source_name: str = "uploaded_text") -> List[Document]:
        doc = Document(page_content=text, metadata={"source": source_name})
        return self.splitter.split_documents([doc])

    def load_file(self, file_path: str) -> List[Document]:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self.load_pdf(file_path)
        elif ext in [".txt", ".md"]:
            return self.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use PDF or TXT.")


# ──────────────────────────────────────────────
# VECTOR STORE MANAGER
# ──────────────────────────────────────────────
class VectorStoreManager:
    """Manages FAISS vector store creation and persistence."""

    def __init__(self, api_key: str):
        # Set as environment variable — avoids Pydantic version conflicts
        os.environ["OPENAI_API_KEY"] = api_key
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def build(self, documents: List[Document]) -> None:
        """Build FAISS index from document chunks."""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def add_documents(self, documents: List[Document]) -> None:
        """Add more documents to an existing index."""
        if self.vector_store is None:
            self.build(documents)
        else:
            self.vector_store.add_documents(documents)

    def save(self, path: str = "faiss_index") -> None:
        if self.vector_store:
            self.vector_store.save_local(path)

    def load(self, path: str = "faiss_index") -> None:
        self.vector_store = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )

    def get_retriever(self, k: int = 5):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Please load documents first.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})


# ──────────────────────────────────────────────
# RAG PIPELINE
# ──────────────────────────────────────────────
class FinanceRAGPipeline:
    """
    End-to-end RAG pipeline for financial document Q&A.
    
    Usage:
        pipeline = FinanceRAGPipeline(api_key="sk-...")
        pipeline.ingest_file("apple_10k.pdf")
        result = pipeline.query("What was Apple's net revenue in Q4?")
        print(result["answer"])
    """

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.0):
        # Set as environment variable — avoids Pydantic version conflicts
        os.environ["OPENAI_API_KEY"] = api_key
        self.api_key = api_key
        self.processor = DocumentProcessor()
        self.vector_manager = VectorStoreManager(api_key)
        self.llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
        )
        self.qa_chain = None
        self.document_count = 0

    def _build_chain(self) -> None:
        retriever = self.vector_manager.get_retriever(k=5)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": FINANCE_PROMPT},
        )

    def ingest_file(self, file_path: str) -> int:
        """Load a file into the vector store. Returns number of chunks created."""
        docs = self.processor.load_file(file_path)
        self.vector_manager.add_documents(docs)
        self.document_count += len(docs)
        self._build_chain()
        return len(docs)

    def ingest_text(self, text: str, source_name: str = "pasted_text") -> int:
        """Load raw text into the vector store."""
        docs = self.processor.load_raw_text(text, source_name)
        self.vector_manager.add_documents(docs)
        self.document_count += len(docs)
        self._build_chain()
        return len(docs)

    def query(self, question: str) -> dict:
        """
        Run a query against the ingested documents.
        
        Returns:
            {
                "answer": str,
                "sources": List[str],
                "chunks": List[str]
            }
        """
        if self.qa_chain is None:
            raise ValueError("No documents ingested. Please upload a financial document first.")

        result = self.qa_chain.invoke({"query": question})

        sources = list({
            doc.metadata.get("source", "Unknown")
            for doc in result.get("source_documents", [])
        })
        chunks = [doc.page_content[:300] for doc in result.get("source_documents", [])]

        return {
            "answer": result["result"],
            "sources": sources,
            "chunks": chunks,
        }

    def save_index(self, path: str = "faiss_index") -> None:
        self.vector_manager.save(path)

    def load_index(self, path: str = "faiss_index") -> None:
        self.vector_manager.load(path)
        self._build_chain()

    def reset(self) -> None:
        """Clear all ingested documents."""
        self.vector_manager.vector_store = None
        self.qa_chain = None
        self.document_count = 0
        