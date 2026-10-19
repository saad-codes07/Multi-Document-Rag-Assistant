import os


from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # reads GROQ_API_KEY from the .env file

LLM_MODEL = "openai/gpt-oss-120b"
TOP_K = 3  # how many chunks to retrieve per question

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


def retrieve(question, vector_store, k=TOP_K):
    """Find the k chunks most similar in meaning to the question."""
    return vector_store.similarity_search(question, k=k)


def format_context(docs):
    """Join the chunks into one text block, labelled with file and page."""
    parts = []
    for doc in docs:
        name = os.path.basename(doc.metadata.get("source", "document"))
        page = doc.metadata.get("page_label", "?")
        parts.append(f"[{name}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question, vector_store, llm=None):
    """Full RAG: retrieve chunks, ask the LLM, return answer + sources."""
    if llm is None:
        llm = get_llm()

    docs = retrieve(question, vector_store)
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

    return response.content, sources