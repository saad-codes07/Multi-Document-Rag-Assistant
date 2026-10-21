from rag.loader import load_pdf, split_documents
from rag.vectorstore import get_embeddings
from rag.hybrid import build_hybrid_retriever
from rag.rerank import load_reranker

QUESTION = "What are the two pre-training tasks of BERT?"
KEYWORD = "next sentence prediction"   # text the answer chunk should contain

chunks = []
for path in ["data/paper.pdf", "data/bert.pdf"]:
    chunks.extend(split_documents(load_pdf(path)))

retriever = build_hybrid_retriever(chunks, get_embeddings())
reranker = load_reranker()

# The same steps the app performs
candidates = retriever.hybrid_search(QUESTION, k=20, fetch=30)
scores = reranker.predict([(QUESTION, c.page_content) for c in candidates])
ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)


def has_keyword(doc):
    return KEYWORD in " ".join(doc.page_content.lower().split())


print("QUESTION:", QUESTION)
print("\nAll 20 candidates after re-ranking:")
for rank, (doc, score) in enumerate(ranked, start=1):
    print(f"  rank {rank:>2} | score {score:6.2f} | page {doc.metadata['page_label']:>2} "
          f"| contains '{KEYWORD}': {has_keyword(doc)}")

print("\n\n========== FULL TEXT OF THE TOP 3 CHUNKS (what the LLM sees) ==========")
for rank, (doc, score) in enumerate(ranked[:3], start=1):
    print(f"\n--- rank {rank}, page {doc.metadata['page_label']} ---")
    print(doc.page_content)

print("\n\n========== FIRST CHUNK THAT CONTAINS THE KEYWORD ==========")
for rank, (doc, score) in enumerate(ranked, start=1):
    if has_keyword(doc):
        print(f"\n--- rank {rank}, page {doc.metadata['page_label']} ---")
        print(doc.page_content)
        break
else:
    print("No chunk among the 20 candidates contains the keyword.")