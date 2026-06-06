import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechNova RAG Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 TechNova RAG Assistant")
st.caption("Powered by TinyLlama + ChromaDB — 100% free, no API key needed.")

# ── Load models (cached — only loads once) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading TinyLlama (first time ~2 mins)...")
def load_generator():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    return hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=False,
    )

@st.cache_resource(show_spinner="Indexing knowledge base...")
def build_vector_store(_embedding_model):
    base = Path("knowledge-base")
    filenames = list(base.rglob("*.md")) if base.exists() else []

    if not filenames:
        return None

    documents, doc_sources = [], []
    for file in filenames:
        try:
            text = file.read_text(encoding="utf-8").strip()
            if text:
                documents.append(text)
                doc_sources.append(file.name)
        except Exception as e:
            st.warning(f"Could not read {file.name}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks, chunk_sources = [], []
    for doc, source in zip(documents, doc_sources):
        split = splitter.split_text(doc)
        chunks.extend(split)
        chunk_sources.extend([source] * len(split))

    embeddings = _embedding_model.encode(chunks, show_progress_bar=False)

    client = chromadb.Client()
    try:
        client.delete_collection("knowledge")
    except:
        pass
    collection = client.create_collection("knowledge")
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": s} for s in chunk_sources],
        ids=[str(i) for i in range(len(chunks))]
    )
    return collection

# ── Load everything ───────────────────────────────────────────────────────────
embedding_model = load_embedding_model()
generator       = load_generator()
collection      = build_vector_store(embedding_model)

if collection is None:
    st.error("No markdown files found in knowledge-base/ folder.")
    st.stop()

# ── RAG function ──────────────────────────────────────────────────────────────
def answer_question(question: str):
    q_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=min(3, collection.count())
    )
    retrieved_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = "\n\n".join(retrieved_chunks)

    prompt = "\n".join([
        "<|system|>",
        "You are a helpful assistant. Answer using only the context provided. Be concise.",
        "<|user|>",
        "Context:\n" + context,
        "Question: " + question,
        "<|assistant|>"
    ])

    output    = generator(prompt)
    full_text = output[0]["generated_text"]
    answer    = full_text.split("<|assistant|>")[-1].strip()
    return answer, list(set(sources))

# ── Chat UI ───────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about TechNova..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = answer_question(question)
        response = answer
        if sources:
            response += "\n\n📄 **Sources:** " + ", ".join(sources)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
