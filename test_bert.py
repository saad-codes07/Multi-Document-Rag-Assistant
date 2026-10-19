from rag.loader import load_pdf, split_documents
from rag.vectorstore import get_embeddings, build_session_vectorstore

chunks = split_documents(load_pdf("data/bert.pdf"))
print("Number of chunks:", len(chunks))

# Which chunks actually contain the answers?
for i, chunk in enumerate(chunks):
    text = chunk.page_content.lower()
    if "stands for" in text or "eleven" in text:
        print(f"\n--- chunk {i}, page {chunk.metadata['page_label']} ---")
        print(chunk.page_content)

store = build_session_vectorstore(chunks, get_embeddings())

tests = [
    ("What does BERT stand for?", "stands for"),
    ("BERT stands for", "stands for"),
    ("On how many NLP tasks did BERT obtain new state-of-the-art results?", "eleven"),
    ("state-of-the-art results on eleven natural language processing tasks", "eleven"),
]

for question, keyword in tests:
    print("\n\nQUESTION:", question)
    results = store.similarity_search_with_score(question, k=10)
    for rank, (doc, score) in enumerate(results, start=1):
        has_answer = keyword in doc.page_content.lower()
        print(f"rank {rank}: page {doc.metadata['page_label']}, "
              f"distance {score:.3f}, contains '{keyword}': {has_answer}")