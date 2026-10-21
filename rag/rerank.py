from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker():
    """Load the cross-encoder (downloads about 90 MB the first time)."""
    return CrossEncoder(RERANK_MODEL)


def rerank(query, docs, reranker, top_k=3):
    """Score every (question, chunk) pair and return the best top_k chunks."""
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]