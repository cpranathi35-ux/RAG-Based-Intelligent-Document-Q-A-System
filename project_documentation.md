# FinanceRAG — Project Documentation

**Project Title:** FinanceRAG: A Retrieval-Augmented Generation System for Financial Document Analysis  
**Institution:** New Jersey Institute of Technology (NJIT)  
**Program:** Master of Science in Data Science  
**Duration:** January 2023 – December 2024  
**Domain:** FinTech / NLP / LLM Applications  

---

## 1. Executive Summary

FinanceRAG is an AI-powered question-answering system designed to extract precise, cited insights from dense financial documents — including SEC 10-K filings, earnings call transcripts, quarterly reports, and analyst notes. The system combines large language model (LLM) capabilities with vector-based semantic retrieval, enabling analysts and investors to interact with financial documents in natural language without reading every page.

The project addresses a key limitation of standard LLMs: hallucination of financial figures. By grounding every response in retrieved document context, FinanceRAG produces verifiable, source-cited answers.

---

## 2. Problem Statement

Financial documents are dense, lengthy, and use specialized terminology. A typical SEC 10-K filing can exceed 100 pages. Analysts spend significant time locating specific figures, risk factors, and management commentary. Existing solutions include:

- **Keyword search:** Fast but misses semantic relationships (e.g., searching "profit" misses "net income")
- **General LLMs (e.g., ChatGPT):** May hallucinate financial figures not in training data
- **Manual reading:** Accurate but extremely time-consuming

**FinanceRAG bridges this gap** by combining the precision of document retrieval with the language understanding of GPT models.

---

## 3. System Architecture

### 3.1 High-Level Components

The system consists of four main components:

**3.1.1 Document Ingestion Layer (`DocumentProcessor`)**  
Responsible for loading financial documents (PDF, TXT), splitting them into semantically meaningful chunks, and adding source metadata. Uses LangChain's `RecursiveCharacterTextSplitter` with a chunk size of 1,000 tokens and an overlap of 150 tokens to preserve sentence continuity across chunk boundaries.

**3.1.2 Embedding & Vector Store (`VectorStoreManager`)**  
Converts document chunks into 1,536-dimensional dense vector embeddings using OpenAI's `text-embedding-ada-002` model. Embeddings are indexed in a FAISS (Facebook AI Similarity Search) in-memory vector store for millisecond-level similarity retrieval.

**3.1.3 RAG Chain (`FinanceRAGPipeline`)**  
Orchestrates the end-to-end retrieval and generation flow using LangChain's `RetrievalQA` chain. On each query, the top-5 most semantically similar chunks are retrieved and injected into a custom finance-specific prompt template before being passed to the OpenAI LLM.

**3.1.4 Frontend (`app.py`)**  
A Streamlit web application providing a user-facing interface for document upload, query input, answer display, source citation, and context chunk inspection.

### 3.2 Data Flow

```
1. User uploads PDF/TXT financial document
        ↓
2. DocumentProcessor splits into ~1,000-token chunks
        ↓
3. OpenAI Embeddings converts each chunk → 1536-dim vector
        ↓
4. FAISS indexes all vectors in memory
        ↓
5. User submits natural language question
        ↓
6. Question is embedded → top-5 similar chunks retrieved via cosine similarity
        ↓
7. Retrieved chunks + question injected into finance prompt template
        ↓
8. GPT-3.5-turbo / GPT-4 generates grounded answer
        ↓
9. Answer + source citations displayed in UI
```

---

## 4. Technical Implementation

### 4.1 Chunking Strategy

Financial documents present unique chunking challenges:

- **Tables and figures** can span multiple lines and lose meaning when split mid-table
- **Section headers** (e.g., "Risk Factors", "MD&A") need to remain with their content
- **Numerical data** often has context that spans several sentences

To address this, `RecursiveCharacterTextSplitter` uses a hierarchy of separators (`\n\n`, `\n`, `.`, ` `) to prefer splitting at paragraph and sentence boundaries rather than mid-sentence.

**Chunk Parameters:**
- Chunk size: 1,000 characters (~250 tokens)
- Overlap: 150 characters to preserve cross-chunk context
- Separators: `["\n\n", "\n", ".", " "]`

### 4.2 Embedding Model

- **Model:** `text-embedding-ada-002` (OpenAI)
- **Dimensions:** 1,536
- **Cost:** ~$0.0001 per 1,000 tokens (very cheap)
- **Rationale:** Strong semantic understanding; widely used for RAG applications as of 2023

### 4.3 Vector Similarity Search

FAISS uses **cosine similarity** to find the top-k most relevant chunks for a given question embedding. With `k=5`, the 5 most relevant text sections are returned per query.

**Why FAISS over alternatives (Pinecone, Chroma)?**
- FAISS is open-source and requires no external API or account
- Runs entirely in-memory — suitable for academic/prototyping use
- Industry-standard at scale (used internally at Meta)

### 4.4 Prompt Engineering

A custom prompt template was designed specifically for financial Q&A:

- **Explicit grounding instruction:** Forces the model to only use provided context
- **Graceful failure mode:** If answer not in context, model is instructed to say so (rather than hallucinate)
- **Citation instruction:** Model is prompted to reference specific figures and sections
- **Zero temperature:** `temperature=0.0` ensures deterministic, consistent answers for financial data

### 4.5 LLM Configuration

| Parameter | Value | Reason |
|---|---|---|
| Model | gpt-3.5-turbo | Cost-effective; strong comprehension |
| Temperature | 0.0 | Deterministic responses for factual data |
| Max tokens | Default (~4,096) | Allow full answer generation |
| Chain type | stuff | Injects all retrieved chunks into one prompt |

---

## 5. Evaluation

### 5.1 Qualitative Evaluation

The system was tested on publicly available financial documents:

- Apple Inc. 10-K (2023)
- Tesla Q4 2023 Earnings Release
- Microsoft FY2024 Annual Report

**Test question categories:**
1. Specific financial figures ("What was net income in Q3?")
2. Risk factor extraction ("What cybersecurity risks are disclosed?")
3. Trend analysis ("How did revenue change compared to last year?")
4. Management commentary ("What did leadership say about AI investments?")
5. Guidance and outlook ("What is the revenue guidance for next quarter?")

**Findings:**
- System correctly retrieved relevant chunks in ~90% of specific figure queries
- Answers were accurate and cited when relevant text was present in documents
- System correctly declined to answer (~"I could not find this information") when the relevant section was missing — confirming anti-hallucination behavior

### 5.2 Limitations

- **Table extraction:** Complex financial tables in PDFs are sometimes incorrectly parsed by PyPDF, leading to incomplete retrieval of tabular data
- **Multi-document reasoning:** The system retrieves from all documents in the index simultaneously, which can occasionally lead to mixing context from different companies
- **Context window:** The "stuff" chain has an effective limit on how much context can be passed to the LLM at once (~4K tokens with GPT-3.5)
- **Cost at scale:** For organizations with hundreds of large documents, the embedding cost and storage requirements increase proportionally

---

## 6. Future Improvements

| Enhancement | Description | Feasibility |
|---|---|---|
| Table-aware parsing | Use `camelot` or `pdfplumber` for structured table extraction | High |
| Multi-query retrieval | Generate multiple sub-questions per query to improve recall | Medium |
| MapReduce chain | Handle larger documents with summarize-then-answer approach | High |
| Reranking | Add a cross-encoder reranker (e.g., Cohere Rerank) on retrieved chunks | Medium |
| Persistent storage | Replace in-memory FAISS with persistent Chroma or Pinecone | High |
| Multi-modal support | Extract and analyze charts/graphs from PDFs | Low |
| Fine-tuning | Fine-tune a small model on financial QA datasets (FinQA, TAT-QA) | Low |

---

## 7. Ethical Considerations

- **Investment decisions:** This system is intended as an analysis aid only and should not be used as the sole basis for investment decisions
- **Data privacy:** Financial documents should only be uploaded if the user has the right to process them
- **API key security:** Users are responsible for keeping their OpenAI API keys secure
- **Hallucination risk:** While RAG significantly reduces hallucination, users should verify critical financial figures independently

---

## 8. References

1. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
2. OpenAI (2023). *GPT-4 Technical Report*. arXiv:2303.08774.
3. Johnson, J. et al. (2019). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data.
4. LangChain Documentation (2023). https://docs.langchain.com
5. Maia, M. et al. (2018). *WWW'18 Open Challenge: Financial Opinion Mining and Question Answering*. WWW'18 Companion.

---

*Document Version: 1.0 | Last Updated: December 2024*
