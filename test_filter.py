from rag.loader import load_pdf, split_documents
from rag.vectorstore import get_embeddings
from rag.hybrid import build_hybrid_retriever, file_of

chunks = []
for path in ["data/paper.pdf", "data/bert.pdf"]:
    chunks.extend(split_documents(load_pdf(path)))

retriever = build_hybrid_retriever(chunks, get_embeddings())

for files in (None, ["paper.pdf"], ["bert.pdf"], ["paper.pdf", "bert.pdf"]):
    results = retriever.hybrid_search("What does BERT stand for?", k=5, files=files)
    print(files, "->", sorted({file_of(doc) for doc in results}))