from rag.loader import load_pdf, split_documents
from rag.vectorstore import get_embeddings
from rag.hybrid import build_hybrid_retriever
from rag.rerank import load_reranker
from rag.chain import answer_question, get_llm

PDF_FILES = ["data/paper.pdf", "data/bert.pdf"]

print("Loading documents and models (this takes a moment)...")
chunks = []
for path in PDF_FILES:
    chunks.extend(split_documents(load_pdf(path)))

retriever = build_hybrid_retriever(chunks, get_embeddings())
reranker = load_reranker()
llm = get_llm()

history = []  # the conversation so far

print("Ask questions about your PDFs. Type 'quit' to exit.")

while True:
    question = input("\nYou: ").strip()

    if question.lower() in ("quit", "exit"):
        break
    if not question:
        continue

    result = answer_question(question, retriever, reranker, llm, history)

    if result["standalone_question"] != question:
        print(f"\n(searched for: {result['standalone_question']})")
    print("\nAI:", result["answer"])
    print("Sources:", "; ".join(result["sources"]))

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result["answer"]})