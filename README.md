# 📚 TechNova RAG Assistant

> A fully functional Retrieval-Augmented Generation (RAG) chatbot built with 100% free, open-source AI — no API keys, no cloud costs, runs entirely locally.

[![Live Demo].(https://huggingface.co/spaces/Indrajithjay/tnova-rag-assistant)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

-----

## 🎯 What is this?

TechNova RAG Assistant is an AI-powered support chatbot for a fictional company called **TechNova**. It reads a structured knowledge base of 34 markdown documents and answers user questions intelligently — retrieving the most relevant context and generating accurate responses.

Built as **CAIEP Project 02** to demonstrate a production-grade RAG pipeline using only open-source tools.

-----

## 🚀 Live Demo

👉 **[Try it live on Hugging Face Spaces]([https://huggingface.co/spaces/Indrajithjay/tenova-rag-assistant](https://huggingface.co/spaces/Indrajithjay/tnova-rag-assistant))**

Example questions to try:

- *“What products does TechNova offer?”*
- *“What is the return and refund policy?”*
- *“What kind of issues do customers commonly report?”*
- *“What is new in NovaOS 5.2?”*
- *“How do I contact TechNova support?”*

-----

## 🏗️ Architecture

```
User Question
     │
     ▼
Sentence Transformers          ← embed the question
(all-MiniLM-L6-v2)
     │
     ▼
ChromaDB Vector Store          ← find top-3 relevant chunks
     │
     ▼
Prompt Builder                 ← combine context + question
     │
     ▼
distilgpt2 (LLM)              ← generate the answer
     │
     ▼
Gradio Chat UI                 ← show answer to user
```

### RAG Pipeline — 4 Steps

|Step       |Component              |Description                                           |
|-----------|-----------------------|------------------------------------------------------|
|1. Ingest  |LangChain Text Splitter|Splits 34 markdown files into 500-char chunks         |
|2. Embed   |all-MiniLM-L6-v2       |Converts chunks into 384-dim vectors                  |
|3. Retrieve|ChromaDB               |Finds top-3 most relevant chunks via cosine similarity|
|4. Generate|distilgpt2             |Produces a grounded answer from retrieved context     |

-----

## 🛠️ Tech Stack

|Tool                   |Purpose       |Why chosen                     |
|-----------------------|--------------|-------------------------------|
|`sentence-transformers`|Embeddings    |Free, fast, runs on CPU        |
|`ChromaDB`             |Vector store  |Lightweight, no server needed  |
|`distilgpt2`           |Language model|Only 82MB, works on any machine|
|`LangChain`            |Text splitting|Smart chunk overlap handling   |
|`Gradio`               |Chat UI       |One-line deployment            |

**No OpenAI. No paid APIs. No GPU required.**

-----

## 📁 Knowledge Base

34 markdown files across 6 categories:

```
knowledge-base/
├── products/          (6 files)  — Nova Phone X1, Laptop Pro, Tablet, Smartwatch, Earbuds, Speaker
├── faqs/              (7 files)  — Shipping, Payments, Returns, Warranty, Trade-in, Software, Account
├── policies/          (5 files)  — Return, Warranty, Shipping, Privacy, Terms of Service
├── company/           (4 files)  — About, Contact, Careers, Sustainability
├── support_tickets/   (8 files)  — Real resolved customer cases
└── release_notes/     (3 files)  — NovaOS 5.1, 5.2, NovaWatch OS 4.1
```

-----

## ⚙️ Run Locally

### Requirements

- Python 3.11
- 500MB free disk space (for model download)
- No GPU needed

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/technova-rag-assistant
cd technova-rag-assistant

# 2. Create virtual environment with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python main.py
```

### Open in browser

```
http://127.0.0.1:7860
```

-----

## 📦 Project Structure

```
technova-rag-assistant/
├── main.py                  # Full RAG pipeline + Gradio UI
├── requirements.txt         # All dependencies
├── README.md                # This file
└── knowledge-base/
    ├── products/
    ├── faqs/
    ├── policies/
    ├── company/
    ├── support_tickets/
    └── release_notes/
```

-----

## 💡 Key Design Decisions

**Why distilgpt2 instead of GPT-4 or TinyLlama?**
distilgpt2 is only 82MB and runs on any machine without GPU. TinyLlama (600MB) caused memory crashes on CPU-only laptops. For a RAG system the retrieval quality matters more than the LLM size.

**Why ChromaDB instead of Pinecone or Weaviate?**
ChromaDB runs fully in-memory with no server setup, perfect for local and Hugging Face deployment.

**Why all-MiniLM-L6-v2 for embeddings?**
It produces high-quality 384-dimension embeddings, runs on CPU in milliseconds, and is one of the most widely benchmarked sentence embedding models available.

-----

## 🗺️ Future Improvements

- [ ] Swap distilgpt2 for a larger model (Mistral 7B) when GPU available
- [ ] Add persistent ChromaDB storage
- [ ] Add evaluation metrics (BLEU, ROUGE, faithfulness score)
- [ ] Support PDF and DOCX knowledge base files
- [ ] Add conversation memory for multi-turn Q&A

-----

## 👨‍💻 Author

**Indrajith Jay**

- Hugging Face: [Indrajithjay](https://huggingface.co/Indrajithjay)
- Live Demo: [tenova-rag-assistant](https://huggingface.co/spaces/Indrajithjay/tenova-rag-assistant)

-----

## 📄 License

MIT License — free to use, modify, and distribute.
