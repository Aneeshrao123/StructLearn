from sentence_transformers import SentenceTransformer
import numpy as np

# Load a small embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Simple resource dictionary for prototype RAG
RESOURCE_TEXTS = {
    "Variables": "Variables store values. They are the foundation of programming.",
    "Loops": "Loops repeat instructions. Useful for iterating over data.",
    "Functions": "Functions organize code into reusable blocks.",
    "Recursion": "Recursion is when a function calls itself.",
    "Dynamic Programming": "DP is recursion with memoization to optimize overlapping subproblems."
}

# Precompute embeddings for faster retrieval
RESOURCE_EMBEDDINGS = {
    key: model.encode(val) for key, val in RESOURCE_TEXTS.items()
}

def retrieve_context(query: str, top_k: int = 3):
    """Retrieve top-k relevant resources using cosine similarity."""
    query_emb = model.encode(query)

    scores = []
    for key, emb in RESOURCE_EMBEDDINGS.items():
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        scores.append((key, RESOURCE_TEXTS[key], float(sim)))

    # Sort by similarity score
    scores.sort(key=lambda x: x[2], reverse=True)

    return scores[:top_k]
