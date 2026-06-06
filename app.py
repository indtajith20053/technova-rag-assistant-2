"""
TechNova RAG Assistant — Streamlit entry point.
"""
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

st.set_page_config(page_title="TechNova RAG Assistant", layout="wide")
st.title("TechNova RAG Assistant")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_chroma_client():
    return chromadb.Client(Settings(anonymized_telemetry=False))

model = load_model()
client = get_chroma_client()
collection = client.get_or_create_collection("technova_docs")

st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload text files to index", type=["txt", "md", "pdf"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        embeddings = model.encode(chunks).tolist()
        ids = [f"{uploaded_file.name}_{i}" for i in range(len(chunks))]
        collection.upsert(documents=chunks, embeddings=embeddings, ids=ids)
    st.sidebar.success(f"Indexed {len(uploaded_files)} file(s).")

st.header("Ask a Question")
query = st.text_input("Enter your question:")

if query:
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    documents = results.get("documents", [[]])[0]
    if documents:
        st.subheader("Relevant Context")
        for i, doc in enumerate(documents, 1):
            st.markdown(f"**Chunk {i}:** {doc}")
    else:
        st.info("No relevant documents found. Please upload some documents first.")
