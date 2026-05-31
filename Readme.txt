# 📄 DocuMind AI

> An intelligent, production-grade RAG-powered document Q&A system built with LangChain, FAISS, Groq (Llama 3), and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Llama%203%208B-orange)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple)](https://faiss.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🧠 What Is This?

DocuMind AI lets you upload **any PDF** and have a natural conversation with it. Ask factual questions, request summaries, explore specific sections — and get accurate, cited answers grounded strictly in your document.

Built on **Retrieval-Augmented Generation (RAG)**, it solves three core limitations of standard LLMs:

| Problem | How DocuMind Solves It |
|---|---|
| Fixed context window | Chunks document → retrieves only relevant sections |
| Hallucination | Answers grounded strictly in retrieved context |
| Training cutoff | Works on any private document the LLM has never seen |

---

## ✨ Features

- 📤 **PDF Upload** — drag-and-drop upload with size and format validation
- 🔍 **Semantic Search** — FAISS vector search across all document chunks
- 🤖 **Llama 3 via Groq** — near-instant responses (~750 tokens/sec)
- 📎 **Source Citations** — every answer shows exactly which page it came from
- 🧵 **Conversation History** — multi-turn Q&A with context awareness
- 🛡️ **Robust Error Handling** — every failure point handled gracefully
- 🆓 **100% Free to Run** — local embeddings + Groq free tier

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  PHASE A — INDEXING                  │
│                                                     │
│  PyMuPDF          LangChain          HuggingFace    │
│  Extraction  ──▶  Chunking     ──▶  Embedding       │
│  (page text)      (500 tokens)       (384-dim vec)  │
│                                          │           │
│                                          ▼           │
│                                     FAISS Index      │
└─────────────────────────────────────────────────────┘
                                          │
                                          │  (stored in session)
                                          │
User Question                             │
    │                                     │
    ▼                                     ▼
┌─────────────────────────────────────────────────────┐
│                  PHASE B — QUERY                     │
│                                                     │
│  Query          FAISS            Prompt             │
│  Embedding ──▶  Top-5      ──▶  Construction  ──▶  │
│                 Chunks           (chunks +           │
│                                   history)           │
│                                          │           │
│                                          ▼           │
│                                   Groq API           │
│                                   (Llama 3 8B)       │
│                                          │           │
│                                          ▼           │
│                              Answer + Page Citations │
└─────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
documind-ai/
├── app.py                    # Main Streamlit application (entry point)
├── requirements.txt          # Pinned dependencies
├── .env.example              # API key template (copy → .env)
├── .gitignore
├── README.md
│
├── src/                      # Core pipeline modules
│   ├── __init__.py
│   ├── pdf_processor.py      # PDF extraction & cleaning (PyMuPDF)
│   ├── chunker.py            # Token-aware text chunking (LangChain)
│   ├── embedder.py           # Local embedding model (sentence-transformers)
│   ├── vector_store.py       # FAISS index construction
│   ├── retriever.py          # Similarity search configuration
│   ├── prompt_builder.py     # Prompt construction & context formatting
│   └── llm_chain.py          # LangChain RAG chain + Groq integration
│
├── utils/                    # Shared utilities
│   ├── __init__.py
│   ├── logger.py             # Loguru logging configuration
│   └── validators.py         # Input validation (file, query, API key)
│
└── tests/                    # Unit tests
    ├── test_pdf_processor.py
    ├── test_chunker.py
    └── test_retriever.py
```

---

## 🚀 Local Setup

### 1. Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/documind-ai.git
cd documind-ai
```

### 3. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First install downloads the PyTorch CPU wheel (~200 MB) and the embedding model (~90 MB). Subsequent runs are instant.

### 5. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 6. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Streamlit Cloud Deployment

1. Push your code to a **public GitHub repository** (`.env` is in `.gitignore` — it will not be pushed)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set `app.py` as the entry point
4. Under **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key_here"
   ```
5. Click **Deploy** — Streamlit Cloud auto-installs `requirements.txt`

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- PDF text extraction and cleaning
- Chunk size and metadata correctness
- Retriever null-guard behaviour

---

## 🔧 Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | Streamlit | Rapid, Python-native web UI |
| PDF Processing | PyMuPDF (fitz) | Handles multi-column, tables, embedded text |
| Text Chunking | LangChain RecursiveCharacterTextSplitter | Respects sentence boundaries |
| Token Counting | tiktoken | Accurate token-based chunk sizing |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 | Free, local, 384-dim, production-quality |
| Vector Database | FAISS (IndexFlatL2) | Free, local, no rate limits, industry standard |
| LLM | Llama 3 8B via Groq API | Free tier, ~750 tok/s, open-source ecosystem |
| LLM Orchestration | LangChain | History-aware retrieval chain |
| Logging | Loguru | Structured, rotating, production-grade |
| Environment | python-dotenv | Secure key management |

---

## ⚙️ Configuration

All tunable parameters are documented in their respective modules:

| Parameter | Default | Location | Description |
|---|---|---|---|
| `chunk_size` | 500 tokens | `src/chunker.py` | Max tokens per chunk |
| `chunk_overlap` | 50 tokens | `src/chunker.py` | Overlap between adjacent chunks |
| `k` | 5 | `src/retriever.py` | Number of chunks retrieved per query |
| `MAX_HISTORY_TURNS` | 3 | `src/llm_chain.py` | Conversation turns kept in context |
| `MAX_FILE_SIZE_MB` | 50 | `utils/validators.py` | Max PDF upload size |
| `GROQ_MODEL` | llama3-8b-8192 | `src/llm_chain.py` | Groq model identifier |

---

## 🛡️ Error Handling

Every failure point is handled with a user-friendly message and logged:

| Scenario | User Message |
|---|---|
| Non-PDF uploaded | "Please upload a PDF file only." |
| File > 50 MB | "File too large. Maximum allowed size is 50 MB." |
| Image-only PDF | "This PDF appears to be image-only with no extractable text." |
| Corrupt PDF | "Could not read this PDF — the file may be corrupted." |
| Groq rate limit | "Rate limit reached. Please wait 60 seconds." + countdown timer |
| Groq timeout | "Response timed out. Please try again." |
| Missing API key | "API key not configured. Add GROQ_API_KEY to your .env file." |
| Query before upload | Chat input disabled until document is processed |

---

## 📊 Performance Targets

| Metric | Target |
|---|---|
| Answer accuracy (factual questions) | > 85% |
| Hallucination rate | < 5% |
| Citation accuracy | > 90% correct page numbers |
| Response time | < 5 seconds per query |
| PDF processing (100 pages) | < 30 seconds |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Ali Sher Khan Tareen**

Built as a portfolio project to demonstrate production-grade LLM engineering:
RAG architecture · Vector databases · Prompt engineering · Full-stack AI deployment

---

*DocuMind AI · v1.0.0 · April 2026*