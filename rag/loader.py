import os
import tempfile
import unicodedata

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path):
    """Read a PDF and return a list of pages (one item per page)."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

def clean_text(text):
    """Convert special characters such as the 'fi' ligature into plain letters."""
    return unicodedata.normalize("NFKC", text)

def split_documents(pages, chunk_size=1000, chunk_overlap=200):
    """Cut the pages into smaller overlapping chunks."""
    for page in pages:
        page.page_content = clean_text(page.page_content)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    return chunks


def load_uploaded_pdf(uploaded_file):
    """Read a PDF uploaded through Streamlit (it lives in memory, not on disk)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)  # clean up the temporary file

    # Replace the temp file's name with the real file name,
    # so the sources shown to the user are meaningful
    for page in pages:
        page.metadata["source"] = uploaded_file.name

    return pages