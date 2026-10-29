import os
import re

from rank_bm25 import BM25Okapi

from rag.vectorstore import build_session_vectorstore

# Very common words carry no signal for keyword search
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with",
}


def tokenize(text):
    """Split text into lowercase words for BM25.

    The pattern keeps things like "4.2.1b", "warmup_steps" and "80.5"
    together as one word instead of breaking them apart.
    """
    words = re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", text.lower())
    return [w for w in words if w not in STOPWORDS]


def file_of(doc):
    """The file name a chunk came from (without any folder)."""
    return os.path.basename(doc.metadata.get("source", ""))


def doc_key(doc):
    """An identity for a chunk, so the same chunk found by both searches
    is recognised as one chunk."""
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)


def reciprocal_rank_fusion(result_lists, k=60):
    """Merge several ranked lists of chunks into one ranked list."""
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs.setdefault(key, doc)
    ordered_keys = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ordered_keys]


class HybridRetriever:
    """Holds both indexes and offers dense, BM25 and hybrid search.

    Every search accepts an optional `files` list (file names). When given,
    only chunks from those files are searched.
    """

    def __init__(self, chunks, vector_store):
        self.chunks = chunks
        self.vector_store = vector_store
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

        # Which "source" values in the database belong to which file name
        self.sources_by_file = {}
        for chunk in chunks:
            self.sources_by_file.setdefault(file_of(chunk), set()).add(
                chunk.metadata.get("source")
            )

    def _chroma_filter(self, files):
        """Build the metadata filter ChromaDB understands (or None)."""
        if not files or set(files) >= set(self.sources_by_file):
            return None  # no filtering needed: search everything
        sources = sorted(
            s for name in files for s in self.sources_by_file.get(name, set())
        )
        if len(sources) == 1:
            return {"source": sources[0]}
        return {"source": {"$in": sources}}

    def dense_search(self, query, k=3, files=None):
        """Search by meaning (embeddings + ChromaDB)."""
        return self.vector_store.similarity_search(
            query, k=k, filter=self._chroma_filter(files)
        )

    def bm25_search(self, query, k=3, files=None):
        """Search by keywords (BM25)."""
        scores = self.bm25.get_scores(tokenize(query))
        allowed = [
            i for i, chunk in enumerate(self.chunks)
            if not files or file_of(chunk) in files
        ]
        best = sorted(allowed, key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in best[:k] if scores[i] > 0]

    def hybrid_search(self, query, k=3, fetch=20, rrf_k=60, files=None):
        """Run both searches, merge with RRF, return the best k chunks."""
        dense = self.dense_search(query, fetch, files)
        sparse = self.bm25_search(query, fetch, files)
        return reciprocal_rank_fusion([dense, sparse], k=rrf_k)[:k]


def build_hybrid_retriever(chunks, embeddings):
    """Build the vector database and the BM25 index from the same chunks."""
    vector_store = build_session_vectorstore(chunks, embeddings)
    return HybridRetriever(chunks, vector_store)