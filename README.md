# TechNova RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant built with Streamlit, ChromaDB, and sentence-transformers.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running locally

```bash
streamlit run app.py
```

## Deployment

This app is deployed on Railway. The start command is:
```
streamlit run 'app..py' --server.port $PORT --server.address 0.0.0.0
```
