from rag.loader import load_pdf, split_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import uuid

chunks = split_documents(load_pdf("data/bert.pdf"))
print("Chunks:", len(chunks))

# (question, text that must appear in the chunk that holds the answer)
tests = [
    ("What does BERT stand for?", "bidirectional encoder representations"),
    ("On how many NLP tasks did BERT obtain new state-of-the-art results?",
     "eleven natural language"),
]

models = [
    "sentence-transformers/all-MiniLM-L6-v2",   # current model (small)
    "BAAI/bge-small-en-v1.5",                   # small, usually better at search
    "sentence-transformers/all-mpnet-base-v2",  # bigger (about 420 MB download)
]


def normalize(text):
    # Collapse line breaks and repeated spaces so phrases can be matched
    return " ".join(text.lower().split())


for model_name in models:
    print("\n==========", model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma(
        collection_name=f"test_{uuid.uuid4().hex}",
        embedding_function=embeddings,
    )
    store.add_documents(chunks)

    for question, needed_text in tests:
        results = store.similarity_search(question, k=20)
        rank = "not in top 20"
        for position, doc in enumerate(results, start=1):
            if needed_text in normalize(doc.page_content):
                rank = position
                break
        print(f"rank of answer chunk: {rank}  |  {question}")