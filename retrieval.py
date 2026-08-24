"""Retrieval step (Week 3): embed the query and find the most similar chunks.

For a small knowledge base a brute-force scan over all stored vectors is
plenty fast; a dedicated vector index only pays off at much larger scale.
"""

import numpy as np

import config
import db
import llm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_top_chunks(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """Return the top_k most relevant chunks with their similarity scores."""
    query_vec = np.array(llm.embed_texts([query])[0])

    conn = db.get_connection()
    chunks = db.fetch_all_chunks(conn)
    conn.close()

    for chunk in chunks:
        chunk["score"] = cosine_similarity(query_vec, np.array(chunk["embedding"]))

    chunks.sort(key=lambda c: c["score"], reverse=True)
    top = chunks[:top_k]
    return [c for c in top if c["score"] >= config.MIN_SIMILARITY]
