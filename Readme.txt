# 🚀 DocuMind AI

** An intelligent, production-grade RAG-powered document Q&A system built with Machine Learning & Generative AI.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Llama%203%208B-orange)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple)](https://faiss.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

🔗 **Live Application:** https://alysher-docmind-ai.streamlit.app/

This application lets you upload any PDF and have a natural, multi-turn conversation with it. By combining **Semantic Search (FAISS)** with **Generative AI (Llama 3)**, it bypasses LLM token limits and training cutoffs to provide accurate, cited answers grounded strictly in your private documents.

---

## ✨ Key Features

### 🤖 **1. Generative AI Q&A Engine**
* Integrated **Llama 3 8B (via Groq API)** for ultra-fast, near-instant responses (~750 tokens/sec).
* Solves hallucination by strictly grounding answers in the retrieved context.
* Maintains a conversation history buffer for natural multi-turn follow-up questions.

### 🔍 **2. Semantic Vector Search**
* Powered by **FAISS** for high-speed similarity search across document chunks.
* Uses local `sentence-transformers` for robust 384-dimensional text embeddings.
* Bypasses fixed context windows by retrieving only the most relevant sections of massive PDFs.

### 📎 **3. Real-Time Citation System**
* Every generated answer includes explicit page number citations.
* Users can cross-reference the AI's logic directly with the source material.

### 📤 **4. Professional PDF Processing**
* Drag-and-drop upload powered by **PyMuPDF** for high-fidelity extraction of multi-column layouts and tables.
* Intelligent chunking via LangChain respects natural sentence boundaries.

### 🎨 **5. Modern UI & Robust Guardrails**
* Clean, interactive Streamlit frontend with a focus on usability.
* Self-healing logic gracefully handles image-only PDFs, file size limits, and API timeouts.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Indexing Phase
        A[PDF Document] -->|PyMuPDF| B(Text Extraction)
        B -->|LangChain| C(Token-Aware Chunking)
        C -->|sentence-transformers| D(Embedding Generation)
        D -->|IndexFlatL2| E[(FAISS Vector Store)]
    end

    subgraph Query Phase
        F[User Query] --> G(Query Embedding)
        G -->|Similarity Search| E
        E -->|Top-K Chunks| H(Context Retrieval)
        H --> I(Prompt Construction)
        I -->|Llama 3 8B| J[Groq API]
        J --> K(Synthesized Answer + Citations)
    end

## 🛠️ Tech Stack

| Component | Technology | Why We Use It |
|---|---|---|
| **Generative AI** | Groq API (Llama 3 8B) | Open-weights leader with ultra-low latency inference |
| **Orchestration** | LangChain | Standardized abstractions for RAG pipelines |
| **Vector Database**| FAISS | In-memory indexing; no rate limits or external dependencies |
| **Embeddings** | all-MiniLM-L6-v2 | 384-dim local embeddings; balances speed and accuracy |
| **Frontend** | Streamlit (Python) | Rapid, Python-native web UI deployment |
| **Data Processing**| PyMuPDF (fitz) | Handles complex PDF layouts better than standard parsers |

---

## ⚙️ Configuration & Tuning

The pipeline is highly modular. Core parameters can be tuned in their respective modules:

| Parameter | Default | Location | Description |
|---|---|---|---|
| `chunk_size` | 500 tokens | `src/chunker.py` | Max tokens per text block |
| `chunk_overlap` | 50 tokens | `src/chunker.py` | Overlap to prevent cutting sentences |
| `k` | 5 | `src/retriever.py` | Number of chunks retrieved per query |
| `MAX_HISTORY_TURNS` | 3 | `src/llm_chain.py` | Conversation turns kept in context |
| `MAX_FILE_SIZE_MB` | 50 | `utils/validators.py` | Max PDF upload size |

---

## 🛡️ Error Handling

Every failure point is handled with a user-friendly message and logged via Loguru:

| Scenario | User Message |
|---|---|
| Non-PDF uploaded | "Please upload a PDF file only." |
| File > 50 MB | "File too large. Maximum allowed size is 50 MB." |
| Image-only PDF | "This PDF appears to be image-only with no extractable text." |
| Groq rate limit | "Rate limit reached. Please wait 60 seconds." |
| Missing API key | "API key not configured. Add GROQ_API_KEY to your .env file." |

---

## 📊 Performance Targets

| Metric | Target |
|---|---|
| Answer accuracy (factual questions) | > 85% |
| Hallucination rate | < 5% |
| Citation accuracy | > 90% correct page numbers |
| Response time | < 5 seconds per query |

---

## 💻 How to Run Locally

### Installation

Requires Python 3.10+ and a free [Groq API key](https://console.groq.com).

```bash
git clone [https://github.com/YOUR_USERNAME/documind-ai.git](https://github.com/YOUR_USERNAME/documind-ai.git)
cd documind-ai

## Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install dependencies
pip install -r requirements.txt

## Enviornment Setup
cp .env.example .env

## Execute
streamlit run app.py
Open http://localhost:8501 in your browser.

## ☁️ Cloud Deployment
1.Push your code to a public GitHub repository.
2.Go to share.streamlit.io and connect your repo.
3.Under Settings → Secrets, add: GROQ_API_KEY = "your_key_here"
4.Click Deploy.

## Testing
pytest tests/ -v
Tests cover extraction fidelity, chunk sizing, and retriever fallback behavior.

## 👤 Author
Ali Sher Khan Tareen Software Engineer specializing in Artificial Intelligence and Machine Learning.