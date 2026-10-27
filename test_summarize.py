import time

from rag.loader import load_pdf, split_documents
from langchain_groq import ChatGroq
from rag.summarize import summarize_documents

chunks = []
for path in ["data/paper.pdf", "data/bert.pdf"]:
    chunks.extend(split_documents(load_pdf(path)))

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, reasoning_effort="low")

start = time.time()
summaries = summarize_documents(chunks, llm, only_file="bert.pdf")
elapsed = time.time() - start

for name, summary in summaries.items():
    print(f"\n===== SUMMARY OF {name} =====\n")
    print(summary)

print(f"\n(took {elapsed:.0f} seconds)")