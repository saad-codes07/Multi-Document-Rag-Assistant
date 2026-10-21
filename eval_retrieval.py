import os

from rag.loader import load_pdf, split_documents
from rag.vectorstore import get_embeddings
from rag.hybrid import build_hybrid_retriever
from eval_set import TESTS

PDF_FILES = ["data/paper.pdf", "data/bert.pdf"]
MAX_RANK = 20  # each method returns this many chunks for scoring


def normalize(text):
    """Lowercase and collapse line breaks/extra spaces so phrases can match."""
    return " ".join(text.lower().split())


def file_name(doc):
    return os.path.basename(doc.metadata.get("source", ""))


def find_rank(docs, test):
    """Position (1 = best) of the first chunk that holds the answer, or None."""
    phrase = normalize(test["phrase"])
    for position, doc in enumerate(docs, start=1):
        if file_name(doc) == test["file"] and phrase in normalize(doc.page_content):
            return position
    return None


def summarize(name, tests, ranks):
    n = len(tests)
    print(f"\n=== {name} ({n} questions) ===")
    for k in (1, 3, 5, 10):
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"  answer chunk in top {k:>2}: {hits}/{n} ({100 * hits / n:.0f}%)")
    mrr = sum(1 / r for r in ranks if r is not None) / n
    print(f"  MRR (1.0 = always first): {mrr:.3f}")

    misses = [(t["q"], r) for t, r in zip(tests, ranks) if r is None or r > 3]
    if misses:
        print("  Not in the top 3:")
        for question, rank in misses:
            where = f"rank {rank}" if rank else f"not in top {MAX_RANK}"
            print(f"    - {question}  ({where})")


def main():
    chunks = []
    for path in PDF_FILES:
        chunks.extend(split_documents(load_pdf(path)))
    print("Total chunks:", len(chunks))

    # Keep only tests whose answer text really exists in the chunks
    valid_tests = []
    print("\nChecking the test set...")
    for test in TESTS:
        phrase = normalize(test["phrase"])
        exists = any(
            file_name(c) == test["file"] and phrase in normalize(c.page_content)
            for c in chunks
        )
        if exists:
            valid_tests.append(test)
        else:
            print(f"  SKIPPED (phrase not found in {test['file']}): {test['q']}")
    print(f"{len(valid_tests)} of {len(TESTS)} questions are usable.")

    retriever = build_hybrid_retriever(chunks, get_embeddings())

    methods = {
        "dense only (current system)": lambda q: retriever.dense_search(q, MAX_RANK),
        "BM25 only (keywords)": lambda q: retriever.bm25_search(q, MAX_RANK),
        "HYBRID (dense + BM25 with RRF)": lambda q: retriever.hybrid_search(
            q, k=MAX_RANK, fetch=30
        ),
    }

    for name, search in methods.items():
        ranks = [find_rank(search(t["q"]), t) for t in valid_tests]
        summarize(name, valid_tests, ranks)


if __name__ == "__main__":
    main()