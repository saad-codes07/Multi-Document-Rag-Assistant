import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from rag.rerank import rerank

load_dotenv()  # reads GROQ_API_KEY from the .env file

LLM_MODEL = "openai/gpt-oss-120b"
TOP_K = 3       # how many chunks the LLM receives
FETCH_K = 20    # how many candidates are re-ranked to find those chunks

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions about a document.\n"
        "Use ONLY the context below to answer. Do not use outside knowledge.\n"
        "If the answer is not in the context, reply exactly: "
        "\"I couldn't find that in the document.\"\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])


def get_llm():
    """Create the LLM connection."""
    return ChatGroq(model=LLM_MODEL, temperature=0)


def retrieve(question, retriever, reranker=None, k=TOP_K):
    """Hybrid search for candidates, then re-rank and keep the best k."""
    candidates = retriever.hybrid_search(question, k=FETCH_K, fetch=30)
    if reranker is None:
        return candidates[:k]
    return rerank(question, candidates, reranker, top_k=k)


def format_context(docs):
    """Join the chunks into one text block, labelled with file and page."""
    parts = []
    for doc in docs:
        name = os.path.basename(doc.metadata.get("source", "document"))
        page = doc.metadata.get("page_label", "?")
        parts.append(f"[{name}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question, retriever, reranker=None, llm=None):
    """Full RAG: retrieve chunks, ask the LLM.

    Returns (answer text, list of source labels, list of retrieved chunks).
    """
    if llm is None:
        llm = get_llm()

    docs = retrieve(question, retriever, reranker)
    context = format_context(docs)

    chain = PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    sources = []
    for doc in docs:
        name = os.path.basename(doc.metadata.get("source", "document"))
        page = doc.metadata.get("page_label", "?")
        label = f"{name}, page {page}"
        if label not in sources:
            sources.append(label)

    return response.content, sources, docs