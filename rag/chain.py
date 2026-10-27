import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.rerank import rerank

load_dotenv()  # reads GROQ_API_KEY from the .env file

LLM_MODEL = "openai/gpt-oss-120b"
TOP_K = 3               # how many chunks the LLM receives
FETCH_K = 20            # how many candidates are re-ranked to find those chunks
MAX_HISTORY_TURNS = 4   # how many past question/answer pairs are remembered

# Prompt that answers the question from the retrieved chunks
PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions about the user's documents.\n"
        "Use ONLY the context below to answer. Do not use outside knowledge.\n"
        "The chat history is only there to help you understand follow-up "
        "questions. Never use it as a source of facts.\n"
        "Write formulas and symbols in plain text, not LaTeX.\n"
        "If the answer is not in the context, reply exactly: "
        "\"I couldn't find that in the document.\"\n\n"
        "Context:\n{context}",
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

# Prompt that turns a follow-up question into a standalone question
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and a follow-up question, rewrite the follow-up "
        "as a standalone question that can be understood without the chat history. "
        "Use only the context the follow-up actually needs, mainly from the most "
        "recent topic. Include the specific names, terms and numbers from the "
        "history that the question refers to. Do not answer the question and do "
        "not add new information. If the question is already standalone, return "
        "it unchanged. Return ONLY the rewritten question.",
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def get_llm(reasoning_effort=None):
    """Create the LLM connection.

    reasoning_effort ("low", "medium" or "high") controls how long the model
    thinks before answering. Lower is faster.
    """
    options = {}
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort
    return ChatGroq(model=LLM_MODEL, temperature=0, **options)


def to_messages(history, max_turns=MAX_HISTORY_TURNS):
    """Turn our history (a list of {'role', 'content'} dicts) into the message
    objects LangChain expects, keeping only the most recent turns."""
    messages = []
    for item in history[-2 * max_turns:]:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages


def condense_question(question, history, llm):
    """Rewrite a follow-up question into a standalone one (if there is history)."""
    messages = to_messages(history)
    if not messages:
        return question
    chain = REWRITE_PROMPT | llm
    result = chain.invoke({"history": messages, "question": question})
    return result.content.strip() or question


def retrieve(question, retriever, reranker=None, k=TOP_K, files=None):
    """Hybrid search for candidates, then re-rank and keep the best k."""
    candidates = retriever.hybrid_search(question, k=FETCH_K, fetch=30, files=files)
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


def answer_question(question, retriever, reranker=None, llm=None,
                    history=None, files=None):
    """Full RAG with conversation memory and an optional file filter.

    Returns a dictionary with:
      answer              - the answer text
      sources             - list of "file, page N" labels
      docs                - the retrieved chunks
      standalone_question - the question as it was actually searched
    """
    if llm is None:
        llm = get_llm()
    history = history or []

    standalone = condense_question(question, history, llm)
    docs = retrieve(standalone, retriever, reranker, files=files)
    context = format_context(docs)

    chain = PROMPT | llm
    response = chain.invoke({
        "context": context,
        "history": to_messages(history),
        "question": question,
    })

    sources = []
    for doc in docs:
        name = os.path.basename(doc.metadata.get("source", "document"))
        page = doc.metadata.get("page_label", "?")
        label = f"{name}, page {page}"
        if label not in sources:
            sources.append(label)

    return {
        "answer": response.content,
        "sources": sources,
        "docs": docs,
        "standalone_question": standalone,
    }