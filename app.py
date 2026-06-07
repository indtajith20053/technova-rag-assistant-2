# ============================================================
#  TechNova RAG Assistant
#  Uses distilgpt2 — ultra lightweight, works on any machine
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import chromadb
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. Load embedding model ───────────────────────────────────────────────────
print("Loading embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Embedding model ready.")

# ── 2. Load lightweight LLM (distilgpt2 — only 82MB, works on any PC) ─────────
print("Loading language model...")
generator = hf_pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=150,
    do_sample=False,
    pad_token_id=50256,
)
print("✅ LLM ready.")

# ── 3. Load markdown files ────────────────────────────────────────────────────
print("Loading knowledge base...")
BASE = Path("knowledge-base")

if not BASE.exists():
    print("WARNING: 'knowledge-base' folder not found.")
    filenames = []
else:
    filenames = list(BASE.rglob("*.md"))
    print(f"Found {len(filenames)} markdown files")

documents, doc_sources = [], []
for file in filenames:
    try:
        text = file.read_text(encoding="utf-8").strip()
        if text:
            documents.append(text)
            doc_sources.append(file.name)
    except Exception as e:
        print(f"Skipping {file.name}: {e}")

# ── 4. Chunk ──────────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks, chunk_sources = [], []
for doc, source in zip(documents, doc_sources):
    split = splitter.split_text(doc)
    chunks.extend(split)
    chunk_sources.extend([source] * len(split))

print(f"Total chunks: {len(chunks)}")

# ── 5. Build ChromaDB vector store ────────────────────────────────────────────
client = chromadb.Client()
try:
    client.delete_collection("knowledge")
except Exception:
    pass

collection = client.create_collection("knowledge")

if chunks:
    print("Generating embeddings...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": s} for s in chunk_sources],
        ids=[str(i) for i in range(len(chunks))]
    )
    print(f"✅ ChromaDB loaded with {collection.count()} chunks")
else:
    print("WARNING: No chunks indexed. Add .md files to knowledge-base/")

# ── 6. RAG answer function ────────────────────────────────────────────────────
def answer_question(question: str, top_k: int = 3):
    if not question.strip():
        return "Please ask a question.", []

    if collection.count() == 0:
        return "No knowledge base found. Please add .md files to knowledge-base/ folder.", []

    # Retrieve relevant chunks
    q_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=min(top_k, collection.count())
    )
    retrieved_chunks = results["documents"][0]
    sources = list(set([m["source"] for m in results["metadatas"][0]]))

    # Use the top chunk as the answer context (distilgpt2 is small so keep prompt short)
    context = retrieved_chunks[0][:400]

    prompt = (
        "Answer the question based on the context below.\n\n"
        "Context: " + context + "\n\n"
        "Question: " + question + "\n\n"
        "Answer:"
    )

    try:
        output = generator(prompt)
        full_text = output[0]["generated_text"]
        # Extract only the part after "Answer:"
        answer = full_text.split("Answer:")[-1].strip()
        # Clean up — stop at first newline to avoid rambling
        answer = answer.split("\n")[0].strip()
        if not answer:
            answer = context  # fallback: return the relevant chunk directly
    except Exception as e:
        answer = f"Retrieved context: {context}"

    return answer, sources

# ── 7. Gradio chat wrapper ────────────────────────────────────────────────────
def chat(message, history):
    answer, sources = answer_question(message)
    if sources:
        return answer + "\n\n📄 Sources: " + ", ".join(sources)
    return answer

# ── 8. Launch UI ──────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=chat,
    title="📚 TechNova RAG Assistant",
    description="Ask anything about the TechNova knowledge base. Free, no API key needed.",
    examples=[
        "What products does TechNova offer?",
        "What is the company's refund policy?",
        "How do I contact support?",
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)
