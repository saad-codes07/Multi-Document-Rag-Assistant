import os

import streamlit as st

from rag.loader import load_uploaded_pdf, split_documents
from rag.vectorstore import get_embeddings
from rag.hybrid import build_hybrid_retriever
from rag.rerank import load_reranker
from rag.chain import answer_question, get_llm
from rag.summarize import summarize_documents

st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄")

# On Streamlit Cloud the API key comes from "Secrets"; locally it comes from .env
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # no secrets file locally: the .env file is used instead

st.title("📄 Multi-Document AI Assistant")
st.caption("Upload PDFs, then ask questions. Answers come only from your documents.")


# Load the heavy models once and reuse them on every re-run
@st.cache_resource
def load_embeddings():
    return get_embeddings()


@st.cache_resource
def load_reranker_model():
    return load_reranker()


@st.cache_resource
def load_llm():
    return get_llm()


@st.cache_resource
def load_summary_llm():
    return get_llm(reasoning_effort="low")  # thinks less, so it is faster


def friendly_error(error):
    """A message that is safe to show to visitors (no technical details)."""
    text = str(error).lower()
    limit_words = ("429", "413", "rate limit", "rate_limit", "request too large")
    if any(word in text for word in limit_words):
        return (
            "The free language-model limit was reached. "
            "Please wait about a minute and try again."
        )
    return "Something went wrong while contacting the language model. Please try again."


# Set up the things we need to remember between re-runs
defaults = {
    "retriever": None,        # hybrid search over all uploaded documents
    "chunks": [],             # the text chunks (needed for summaries)
    "messages": [],           # the chat so far
    "processed_files": [],    # names of the processed files
    "summaries": {},          # {file name: summary text}
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------- Sidebar: upload, process, choose files, summarize ----------
summarize_clicked = False
selected_files = []

with st.sidebar:
    st.header("Your documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("Process documents", disabled=not uploaded_files):
        with st.spinner("Reading and indexing your documents..."):
            all_chunks = []
            for uploaded_file in uploaded_files:
                pages = load_uploaded_pdf(uploaded_file)
                all_chunks.extend(split_documents(pages))

            if not all_chunks:
                st.error("No readable text found. Scanned PDFs are not supported.")
            else:
                st.session_state.retriever = build_hybrid_retriever(
                    all_chunks, load_embeddings()
                )
                st.session_state.chunks = all_chunks
                st.session_state.messages = []   # start a fresh chat
                st.session_state.summaries = {}  # and fresh summaries
                st.session_state.processed_files = [
                    f.name for f in uploaded_files
                ]

    if st.session_state.processed_files:
        st.success("Ready! Documents loaded.")
        selected_files = st.multiselect(
            "Search and summarize these files",
            options=st.session_state.processed_files,
            default=st.session_state.processed_files,
        )

        st.divider()
        st.caption(
            "A summary reads the whole document. On the free API tier this "
            "takes about 5 minutes per paper."
        )
        summarize_clicked = st.button(
            "📝 Summarize selected files", disabled=not selected_files
        )


# ---------- Summaries ----------
if summarize_clicked:
    to_do = [
        n for n in selected_files
        if n not in st.session_state.summaries
    ]
    if not to_do:
        st.info("Those files are already summarized (see below).")
    else:
        with st.status(
            "Summarizing... this can take several minutes. Please keep this page open.",
            expanded=True,
        ) as status:
            try:
                for name in to_do:
                    result = summarize_documents(
                        st.session_state.chunks,
                        load_summary_llm(),
                        only_file=name,
                        progress=st.write,
                    )
                    st.session_state.summaries.update(result)
                status.update(
                    label="Summaries ready",
                    state="complete",
                    expanded=False,
                )
            except Exception as error:
                print("Summarize error:", error)  # details stay in the server log
                status.update(label="Stopped early", state="error")
                st.warning(friendly_error(error))

if st.session_state.summaries:
    st.subheader("📝 Summaries")
    for name, summary in st.session_state.summaries.items():
        with st.expander(f"Summary of {name}", expanded=True):
            st.markdown(summary)
    st.divider()


# ---------- Chat ----------
def show_message(message):
    """Draw one chat message, with its sources and source passages."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("searched_for"):
            st.caption(f"🔎 Searched for: {message['searched_for']}")
        if message.get("sources"):
            st.caption("Sources: " + "; ".join(message["sources"]))
        if message.get("citations"):
            with st.expander("Show source passages"):
                for citation in message["citations"]:
                    st.markdown(f"**{citation['label']}**")
                    st.text(citation["text"])


for message in st.session_state.messages:
    show_message(message)

if st.session_state.retriever is None:
    st.info("👈 Upload a PDF in the sidebar and click **Process documents** to begin.")
elif not selected_files:
    st.warning("Select at least one file in the sidebar to search.")
else:
    question = st.chat_input("Ask a question about your documents")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Thinking..."):
            try:
                result = answer_question(
                    question,
                    st.session_state.retriever,
                    load_reranker_model(),
                    load_llm(),
                    history=st.session_state.messages,
                    files=selected_files,
                )
            except Exception as error:
                print("Answer error:", error)  # details stay in the server log
                result = None
                error_message = friendly_error(error)

        if result is None:
            st.warning(error_message)
        else:
            searched_for = result["standalone_question"]
            user_message = {"role": "user", "content": question}
            assistant_message = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "searched_for": (
                    searched_for if searched_for != question else None
                ),
                "citations": [
                    {
                        "label": "{}, page {}".format(
                            os.path.basename(
                                doc.metadata.get("source", "document")
                            ),
                            doc.metadata.get("page_label", "?"),
                        ),
                        "text": doc.page_content,
                    }
                    for doc in result["docs"]
                ],
            }
            show_message(assistant_message)
            st.session_state.messages.extend(
                [user_message, assistant_message]
            )