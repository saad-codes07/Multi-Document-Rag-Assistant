import os

import streamlit as st

from rag.loader import load_uploaded_pdf, split_documents
from rag.vectorstore import get_embeddings, build_session_vectorstore
from rag.chain import answer_question, get_llm

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
def load_llm():
    return get_llm()


# Set up the things we need to remember between re-runs
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# ---------- Sidebar: upload and process documents ----------
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
                st.session_state.vector_store = build_session_vectorstore(
                    all_chunks, load_embeddings()
                )
                st.session_state.messages = []  # start a fresh chat
                st.session_state.processed_files = [f.name for f in uploaded_files]

    if st.session_state.processed_files:
        st.success("Ready! Documents loaded:")
        for name in st.session_state.processed_files:
            st.write("•", name)


# ---------- Main area: the chat ----------
# Show the conversation so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + "; ".join(message["sources"]))

if st.session_state.vector_store is None:
    st.info("👈 Upload a PDF in the sidebar and click **Process documents** to begin.")
else:
    question = st.chat_input("Ask a question about your documents")

    if question:
        # Show and save the user's question
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get, show and save the answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(
                    question, st.session_state.vector_store, load_llm()
                )
            st.markdown(answer)
            st.caption("Sources: " + "; ".join(sources))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )