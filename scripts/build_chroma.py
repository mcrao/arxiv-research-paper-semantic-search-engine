import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------- Load data ----------------
with open("data/arxiv_data.pkl", "rb") as f:
    data = pickle.load(f)

titles = data["titles"]
abstracts = data["abstracts"]

# ---------------- Embeddings ----------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(abstracts, show_progress_bar=True)

# Normalize (CRITICAL)
embeddings = embeddings / np.linalg.norm(
    embeddings, axis=1, keepdims=True
)

# ---------------- Chroma ----------------
client = chromadb.PersistentClient(
    path="./chroma_arxiv"
)

collection = client.get_or_create_collection(
    name="arxiv_abstracts",
    metadata={"hnsw:space": "cosine"}
)

# ---------------- Batched insert ----------------
BATCH_SIZE = 1000

for i in range(0, len(abstracts), BATCH_SIZE):
    end = min(i + BATCH_SIZE, len(abstracts))
    collection.add(
        documents=abstracts[i:end],
        metadatas=[{"title": titles[j]} for j in range(i, end)],
        embeddings=embeddings[i:end].tolist(),
        ids=[str(j) for j in range(i, end)]
    )

print("Chroma DB created with", collection.count(), "documents")
