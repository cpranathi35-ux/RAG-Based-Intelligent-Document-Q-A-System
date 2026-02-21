# 📈 FinanceRAG — AI-Powered Financial Document Q&A System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5%2F4-orange)
![FAISS](https://img.shields.io/badge/VectorStore-FAISS-red)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions about financial documents — stock reports, 10-K filings, earnings statements, and analyst notes — powered by OpenAI and LangChain.

**Developed as part of the NJIT Data Science Master's Program (2023–2024)**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Running Tests](#running-tests)
- [Sample Use Cases](#sample-use-cases)
- [Technologies Used](#technologies-used)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

**FinanceRAG** solves a real problem in financial analysis: extracting specific, accurate insights from large, dense financial documents without reading them page by page.

Instead of asking a general-purpose LLM (which may hallucinate financial figures), FinanceRAG **grounds every answer in your actual documents** using a Retrieval-Augmented Generation pipeline:

1. **Ingest** — Upload PDF or TXT financial documents
2. **Chunk & Embed** — Documents are split and converted to vector embeddings
3. **Retrieve** — On each query, the most relevant document sections are fetched
4. **Generate** — GPT synthesizes a precise answer using only those sections

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────┐    ┌──────────────────────┐
│  Streamlit UI   │    │  Document Ingestion   │
│   (app.py)      │    │  (PDF/TXT → Chunks)   │
└────────┬────────┘    └──────────┬───────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────────┐
│  RAG Pipeline   │◄───│  FAISS Vector Store  │
│ (rag_pipeline)  │    │  (OpenAI Embeddings) │
└────────┬────────┘    └──────────────────────┘
         │
         ▼
┌─────────────────┐
│   OpenAI LLM   │
│ (GPT-3.5/GPT-4) │
└────────┬────────┘
         │
         ▼
    Answer + Sources
```

---

## ✨ Features

- **Multi-document support** — Upload and query across multiple reports simultaneously
- **Source citations** — Every answer cites which document it came from
- **Context transparency** — View the exact text chunks retrieved for any answer
- **Finance-tuned prompting** — Custom prompt template designed for financial analysis
- **Model flexibility** — Switch between GPT-3.5-turbo and GPT-4
- **Paste text support** — Paste earnings call transcripts or analyst notes directly
- **Persistent indexing** — Save and reload your FAISS index between sessions
- **Suggested questions** — Built-in finance-specific starter questions

---

## 📁 Project Structure

```
finance-rag-qa/
│
├── app.py                  # Streamlit web application (frontend)
├── rag_pipeline.py         # Core RAG logic (backend)
├── test_pipeline.py        # Unit & integration tests
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
│
├── docs/
│   ├── project_documentation.md   # Technical project documentation
│   └── user_manual.md             # End-user guide
│
├── sample_data/            # Place sample PDFs here for testing
│   └── README.md
│
└── faiss_index/            # Auto-generated; stores your vector index
    └── (created at runtime)
```

---

## ✅ Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | [Download](https://www.python.org/downloads/) |
| pip | Latest | Comes with Python |
| OpenAI API Key | Active | [Get one here](https://platform.openai.com/api-keys) |
| Git | Any | [Download](https://git-scm.com/) |
| RAM | 4GB+ | 8GB recommended for large documents |

> **Cost note:** Running queries uses OpenAI API tokens. Approximate cost: ~$0.002 per query with GPT-3.5-turbo. GPT-4 is ~10–20× more expensive.

---

## 🚀 Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/finance-rag-qa.git
cd finance-rag-qa
```

### Step 2 — Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> This installs LangChain, OpenAI SDK, FAISS, Streamlit, and all other dependencies.

### Step 4 — Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Open .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

This will open the app in your browser at `http://localhost:8501`.

**First time setup in the app:**
1. Enter your OpenAI API key in the sidebar
2. Click **Initialize Pipeline**
3. Upload a financial PDF (e.g., Apple's 10-K)
4. Ask questions in the main panel

---

## 🧪 Running Tests

```bash
# Run unit tests (no API key required)
pytest test_pipeline.py -v

# Run all tests including integration test (requires real API key)
OPENAI_API_KEY=sk-your-key pytest test_pipeline.py -v
```

Expected output:
```
test_pipeline.py::TestDocumentProcessor::test_load_raw_text_returns_documents PASSED
test_pipeline.py::TestDocumentProcessor::test_load_raw_text_source_metadata   PASSED
test_pipeline.py::TestDocumentProcessor::test_chunk_size_respected             PASSED
...
```

---

## 💼 Sample Use Cases

| Document Type | Example Questions |
|---|---|
| Apple 10-K | "What were the main revenue segments?" |
| Tesla Earnings | "How did automotive revenue change YoY?" |
| Fed Reserve Report | "What interest rate decisions were made?" |
| Analyst Report | "What is the 12-month price target?" |
| Earnings Call Transcript | "What did the CEO say about AI investments?" |

---

## 🛠 Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| LangChain | 0.2.16 | RAG orchestration framework |
| OpenAI API | 1.43 | Embeddings + LLM (GPT-3.5/4) |
| FAISS | 1.8.0 | Vector similarity search |
| Streamlit | 1.38 | Web UI |
| PyPDF | 4.3 | PDF parsing |
| tiktoken | 0.7 | Token counting |

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: langchain` | Run `pip install -r requirements.txt` |
| `AuthenticationError: Invalid API key` | Check your key at platform.openai.com |
| App doesn't open in browser | Try `http://localhost:8501` manually |
| PDF not loading | Ensure it's not password-protected |
| Slow responses | Switch to `gpt-3.5-turbo` for faster results |
| Out of memory on large PDFs | Reduce chunk size in `rag_pipeline.py` |

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add some feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Built with ❤️ | NJIT Data Science M.S. | 2023–2024*
