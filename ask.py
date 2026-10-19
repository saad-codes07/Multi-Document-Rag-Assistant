from rag.vectorstore import load_vectorstore
from rag.chain import answer_question, get_llm

vector_store = load_vectorstore()   # opens the database we built in Week 3
llm = get_llm()

print("Ask questions about your PDF. Type 'quit' to exit.")

while True:
    question = input("\nYou: ").strip()

    if question.lower() in ("quit", "exit"):
        break
    if not question:
        continue

    answer, pages = answer_question(question, vector_store, llm)
    print("\nAI:", answer)
    print("Sources:", "; ".join(pages))