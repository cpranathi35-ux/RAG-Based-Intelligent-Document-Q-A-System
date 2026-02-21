# FinanceRAG — User Manual

**Version:** 1.0  
**Last Updated:** December 2024  
**Application:** FinanceRAG — AI-Powered Financial Document Q&A

---

## Welcome to FinanceRAG 👋

FinanceRAG lets you **ask questions in plain English** about your financial documents — stock reports, earnings statements, 10-K filings, analyst notes — and get accurate, cited answers instantly.

No more scrolling through 100-page PDFs to find one number.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Launching the Application](#2-launching-the-application)
3. [Application Interface Overview](#3-application-interface-overview)
4. [Step-by-Step: Your First Query](#4-step-by-step-your-first-query)
5. [Uploading Documents](#5-uploading-documents)
6. [Asking Questions](#6-asking-questions)
7. [Understanding the Answer](#7-understanding-the-answer)
8. [Advanced Features](#8-advanced-features)
9. [Tips for Better Results](#9-tips-for-better-results)
10. [Common Errors and Solutions](#10-common-errors-and-solutions)
11. [Frequently Asked Questions](#11-frequently-asked-questions)

---

## 1. System Requirements

Before launching the app, ensure you have:

| Requirement | Details |
|---|---|
| Operating System | Windows 10+, macOS 12+, or Ubuntu 20+ |
| Python | Version 3.10 or higher |
| Internet Connection | Required (to call OpenAI API) |
| OpenAI API Key | Required (free to create at platform.openai.com) |
| Browser | Chrome, Firefox, or Safari (modern version) |
| RAM | Minimum 4GB free |

---

## 2. Launching the Application

**Step 1:** Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux)

**Step 2:** Navigate to the project folder:
```
cd finance-rag-qa
```

**Step 3:** Activate your virtual environment:
```
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

**Step 4:** Launch the app:
```
streamlit run app.py
```

**Step 5:** Your browser will automatically open to:
```
http://localhost:8501
```

> ✅ If your browser doesn't open automatically, copy and paste the URL above into your browser.

---

## 3. Application Interface Overview

The application has two main areas:

### Left Sidebar (Configuration Panel)
This is where you:
- Enter your OpenAI API Key
- Choose the AI model
- Upload your financial documents
- View index statistics
- Reset the application

### Main Panel (Q&A Interface)
This is where you:
- See suggested starter questions
- Type your own questions
- View AI-generated answers
- See source citations
- Browse retrieved document chunks

---

## 4. Step-by-Step: Your First Query

Follow these steps to get your first answer in under 2 minutes:

**Step 1 — Enter your API Key**  
In the left sidebar, find the **"🔑 OpenAI API Key"** field. Type or paste your API key (starts with `sk-`). Your key is hidden for security.

**Step 2 — Click Initialize Pipeline**  
Press the **"✅ Initialize Pipeline"** button. You will see "Pipeline ready!" when successful.

**Step 3 — Upload a Document**  
Under **"📂 Upload Documents"**, click **"Browse files"** and select a PDF or TXT financial document from your computer. Wait for the green checkmark confirming it was loaded.

**Step 4 — Ask a Question**  
In the main panel, type your question in the text box. For example:  
`What was the total revenue for the year?`

**Step 5 — Click "Ask →"**  
Press the blue **"Ask →"** button. In a few seconds, the answer will appear below with source citations.

---

## 5. Uploading Documents

### Supported File Types

| Type | Extension | Notes |
|---|---|---|
| PDF | `.pdf` | Best for annual reports, 10-Ks, earnings releases |
| Text | `.txt` | Good for earnings transcripts, pasted content |

### How to Upload

1. Click **"Browse files"** under the Upload section in the sidebar
2. Select one or more files from your computer
3. Wait for the green success message showing how many chunks were created
4. You can upload multiple documents — they all become searchable together

### What Documents Work Best

✅ **Excellent results:**
- SEC 10-K and 10-Q filings
- Quarterly earnings press releases
- Analyst research reports (text-based)
- Earnings call transcripts

⚠️ **May have reduced accuracy:**
- Scanned PDFs (non-text/image-based scans)
- Documents with complex multi-column layouts
- Password-protected PDFs

### Pasting Text Directly

If you have text copied from a website or document:
1. Scroll to **"📋 Or Paste Text"** in the sidebar
2. Paste your text into the text area
3. Give it a source label (e.g., "apple_q3_transcript")
4. Click **"📥 Load Text"**

---

## 6. Asking Questions

### Using Suggested Questions

Five finance-specific starter questions appear at the top of the main panel. Click any of them to instantly populate the question box. These are great starting points.

### Writing Effective Questions

Be specific and include context for best results. See the comparison table below:

| Less Effective | More Effective |
|---|---|
| "Revenue?" | "What was the total net revenue for fiscal year 2023?" |
| "Risks?" | "What cybersecurity and data privacy risks are disclosed?" |
| "Did they do well?" | "How did earnings per share change compared to the prior year?" |
| "Future plans?" | "What is the company's guidance for Q1 2025 revenue?" |

### Multiple Questions

You can ask as many questions as you like in sequence. All questions and answers are saved in your chat history below the input box.

---

## 7. Understanding the Answer

Each answer consists of three parts:

### The Answer Box (Blue)
The AI's direct answer to your question, grounded in your documents. If the system cannot find the information, it will say: *"I could not find this information in the provided documents"* — this is correct behavior, not a bug.

### Sources
Small blue badges showing which documents the answer was retrieved from. If you uploaded `apple_10k_2023.pdf` and the answer came from it, you'll see `📄 apple_10k_2023.pdf`.

### Retrieved Context Chunks (Expandable)
Click **"🔎 View retrieved context chunks"** to see the exact text passages from your document that were used to generate the answer. This allows you to verify accuracy yourself.

---

## 8. Advanced Features

### Switching Models

In the sidebar, use the **"🤖 Model"** dropdown to switch between:

| Model | Speed | Cost | Best For |
|---|---|---|---|
| gpt-3.5-turbo | Fast | Low (~$0.002/query) | General queries, quick analysis |
| gpt-4 | Medium | Higher (~$0.03/query) | Complex multi-step analysis |
| gpt-4-turbo | Fast | High | Large document contexts |

### Saving Your Index

After loading documents, you can save the FAISS index programmatically (via code) to avoid re-embedding the same documents next session. See `rag_pipeline.py` for `save_index()` and `load_index()` methods.

### Clearing Chat History

Click **"Clear History"** to remove previous Q&A pairs from the screen without resetting your uploaded documents.

### Reset Everything

Click **"🗑️ Reset Everything"** to clear all uploaded documents, chat history, and the vector index. You will need to re-upload documents and re-initialize the pipeline after this.

---

## 9. Tips for Better Results

**Tip 1: Upload the most relevant document**  
If you want to ask about Apple's 2023 revenue, upload Apple's 2023 10-K — not a general market report.

**Tip 2: Ask one question at a time**  
The system performs best with focused, single-topic questions. Instead of "What's the revenue, profit, and future guidance?", ask three separate questions.

**Tip 3: Use financial terminology**  
The system understands financial language. Use terms like "EBITDA", "operating income", "diluted EPS", "capital expenditures" for more precise retrieval.

**Tip 4: Check the source chunks**  
If an answer seems off, expand the context chunks to see what the model was working from. You can verify against the original document.

**Tip 5: Use GPT-4 for complex analysis**  
For questions requiring reasoning across multiple sections (e.g., "How do the disclosed risks compare to the company's mitigation strategies?"), GPT-4 performs meaningfully better than GPT-3.5.

**Tip 6: Keep an eye on costs**  
If you're uploading many documents or running many queries, monitor your OpenAI API usage at platform.openai.com/usage.

---

## 10. Common Errors and Solutions

| Error Message | Cause | Solution |
|---|---|---|
| "Please enter your API key and initialize the pipeline" | Pipeline not initialized | Enter API key and click "Initialize Pipeline" |
| "Please upload at least one financial document first" | No documents loaded | Upload a PDF or TXT file |
| "AuthenticationError: Incorrect API key" | Invalid or expired API key | Go to platform.openai.com and regenerate your key |
| "RateLimitError" | Too many requests or quota exceeded | Wait 60 seconds and try again; check your OpenAI billing |
| Document uploads but shows 0 chunks | File may be image-based PDF | Convert to text-based PDF using Adobe Acrobat or similar |
| Answer says "I could not find this" | The info is not in your document | Upload the correct/more complete document |
| App doesn't open in browser | Browser didn't auto-launch | Go to `http://localhost:8501` manually |
| ModuleNotFoundError | Dependencies not installed | Run `pip install -r requirements.txt` |

---

## 11. Frequently Asked Questions

**Q: Is my financial document sent anywhere besides OpenAI?**  
A: Your document text is sent to OpenAI's API for embedding generation and answer generation. Refer to OpenAI's data usage policy at openai.com/policies. No data is stored on any third-party server by this application.

**Q: How many documents can I upload at once?**  
A: There is no hard limit, but performance and memory usage increase with each document. For best results, keep the total document size under 50MB per session.

**Q: Can I use this for investment decisions?**  
A: FinanceRAG is a research and analysis tool. It should support — not replace — professional financial analysis. Always verify critical figures with original documents before making financial decisions.

**Q: Will it work with non-English documents?**  
A: The system works best with English documents. OpenAI models support some multilingual capability, but accuracy for financial figures may be lower in non-English reports.

**Q: Does my data persist between sessions?**  
A: No. When you close the app, the document index is cleared. You will need to re-upload documents when you restart. Use `save_index()` in the code to persist the index between sessions.

**Q: Why is the answer sometimes incomplete or vague?**  
A: The answer quality depends on how clearly the relevant information appears in the source document. If the information is in a complex table or chart, it may not be extracted accurately by the PDF parser.

---

*FinanceRAG User Manual v1.0 | NJIT Data Science M.S. Project 2023–2024*  
*For technical issues, refer to the README.md or project documentation.*
