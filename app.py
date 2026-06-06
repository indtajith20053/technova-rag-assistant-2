"""
TechNova RAG Assistant — Streamlit entry point.
"""
import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

st.set_page_config(page_title="TechNova RAG Assistant", page_icon="🤖", layout="wide")

st.title("🤖 TechNova RAG Assistant")
st.caption("Retrieval-Augmented Generation powered by ChromaDB & Sentence Transformers")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None

# --- Sidebar: document ingestion ---
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a text document", type=["txt", "md", "pdf"])
    chunk_size = st.slider("Chunk size (chars)", 200, 2000, 500, step=100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, 50, step=50)

    if uploaded_file and st.button("Ingest document"):
        raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(raw_text)

        with st.spinner("Embedding chunks…"):
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        client = chromadb.Client(Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection("technova_docs")
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )
        st.session_state.collection = collection
        st.success(f"Ingested {len(chunks)} chunks.")

# --- Chat interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your document…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.collection is None:
            answer = "⚠️ Please upload and ingest a document first using the sidebar."
        else:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = model.encode([prompt]).tolist()
            results = st.session_state.collection.query(
                query_embeddings=query_embedding, n_results=3
            )
            context = "\n\n".join(results["documents"][0])
            answer = f"**Retrieved context:**\n\n{context}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
