# 📄 Multi-Document AI Assistant (RAG)

Upload your own PDFs and ask questions about them in plain language. The assistant answers **only from your documents**, shows which file and page each answer came from, and says so when the answer isn't there.

**🔗 Live demo:** https://multi-doc-rag-assistant0.streamlit.app/

![App screenshot](screenshot.png)

## What it does

- Upload one or more PDFs and chat with them like in ChatGPT
- Answers are grounded in the documents, so the model is told not to use outside knowledge
- Every answer lists its sources (file name and page number)
- Replies "I couldn't find that in the document." when the documents don't contain the answer

## How it works

This project uses **RAG (Retrieval-Augmented Generation)**:

**Indexing** (when you click "Process documents")
1. `PyPDFLoader` extracts the text from each PDF page
2. `RecursiveCharacterTextSplitter` cuts it into chunks of about 1000 characters with a 200-character overlap
3. A HuggingFace embedding model turns each chunk into a vector
4. The vectors are stored in ChromaDB, together with the file name and page number

**Answering** (each time you ask a question)
1. The question is turned into a vector with the same model
2. ChromaDB returns the 3 most similar chunks
3. The chunks and the question go to the LLM with the instruction to answer only from that context
4. The answer is shown with its source pages

## Tech stack

| Part | Tool |
|---|---|
| Language | Python |
| RAG framework | LangChain |
| Embeddings | BAAI/bge-small-en-v1.5 (HuggingFace, runs locally) |
| Vector database | ChromaDB |
| LLM | openai/gpt-oss-120b via the Groq API |
| Frontend | Streamlit |
| Hosting | Streamlit Community Cloud |

## Choosing the embedding model

My first model (`all-MiniLM-L6-v2`) worked well on one paper but failed on a second one. For "What does BERT stand for?", the chunk containing the answer ranked **14th**, and for another question it wasn't in the top 20, so the LLM never saw it. I wrote a test (`compare_models.py`) that measures where the answer chunk ranks for each model:

| Model | "What does BERT stand for?" | "On how many NLP tasks...?" |
|---|---|---|
| all-MiniLM-L6-v2 | rank 14 | not in top 20 |
| **BAAI/bge-small-en-v1.5** | **rank 1** | **rank 2** |
| all-mpnet-base-v2 | not in top 20 | rank 6 |

I switched to `bge-small-en-v1.5`, and the answers now come back correctly. This was a small test (2 questions, 1 document), so it's evidence and not a full benchmark. It also showed me that a larger model isn't automatically better for retrieval.

## Run it locally

```bash
git clone https://github.com/saad-codes07/Multi-Document-Rag-Assistant.git
cd Multi-Document-Rag-Assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Get a free API key from [console.groq.com](https://console.groq.com) and create a file named `.env` in the project folder:

```
GROQ_API_KEY=your_key_here
```

Then start the app:

```bash
streamlit run app.py
```

## Project structure

```
├── app.py               # Streamlit web interface
├── rag/
│   ├── loader.py        # PDF loading and chunking
│   ├── vectorstore.py   # embeddings and ChromaDB
│   └── chain.py         # retrieval, prompt and LLM
├── compare_models.py    # embedding model comparison experiment
├── ask.py               # terminal version of the assistant
├── test_*.py            # scripts used while building each stage
└── requirements.txt
```

## Limitations

- **Scanned PDFs** (images of text) are not supported, because there is no text to extract. OCR would be needed.
- Each question is answered **independently**, with no memory of earlier questions.
- **Whole-document questions** such as "summarize this PDF" work poorly. RAG retrieves only a few chunks, so it is best at specific questions.
- Tables and figures in PDFs are extracted poorly.
- Retrieval can miss the right chunk. When that happens, the assistant says it couldn't find the answer.
- Document text is sent to an external LLM service (Groq), so it is not suitable for confidential files.

## Ideas for improvement

- Conversation memory, so follow-up questions work
- A "Summarize" feature (map-reduce over all chunks)
- OCR for scanned PDFs
- A reranker and an evaluation set to measure answer accuracy
- Support for Word and text files

## What I learned

- How RAG works end to end: chunking, embeddings, vector search, and prompting
- That retrieval quality limits answer quality, and that it should be **measured**, not assumed
- Deploying a Python app and keeping API keys out of GitHub