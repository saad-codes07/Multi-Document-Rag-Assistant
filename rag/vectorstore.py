import uuid

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "pdf_docs"


def get_embeddings():
    """Create the embedding model (downloads it the first time)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks):
    """Convert chunks to vectors and store them in a fresh ChromaDB."""
    embeddings = get_embeddings()

    # Wipe any old data so re-running doesn't create duplicates
    old_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    old_store.delete_collection()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vector_store.add_documents(chunks)
    return vector_store


def load_vectorstore():
    """Open a database that was already built and saved on disk."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )

def build_session_vectorstore(chunks, embeddings):
    """Build a temporary in-memory database for one user's session."""
    store = Chroma(
        collection_name=f"session_{uuid.uuid4().hex}",
        embedding_function=embeddings,
    )
    store.add_documents(chunks)
    return store