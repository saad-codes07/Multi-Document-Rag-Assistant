from rag.loader import load_pdf, split_documents
from rag.vectorstore import build_vectorstore

# 1. Load and split the PDF (Week 2)
pages = load_pdf("data/paper.pdf")
chunks = split_documents(pages)
print("Chunks to store:", len(chunks))

# 2. Embed the chunks and store them (Week 3)
vector_store = build_vectorstore(chunks)
print("Stored in ChromaDB!")

# 3. Search by meaning
questions = [
    "How long did it take to train the model?",
    "What optimizer was used?",
]

for question in questions:
    print("\n==========================================")
    print("QUESTION:", question)
    results = vector_store.similarity_search_with_score(question, k=3)

    for doc, score in results:
        print(f"\n--- page {doc.metadata['page_label']}, distance {score:.3f} ---")
        print(doc.page_content[:400])