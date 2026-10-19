from rag.loader import load_pdf, split_documents

# 1. Load the PDF
pages = load_pdf("data/paper.pdf")
print("Number of pages:", len(pages))
print("Metadata of page 1:", pages[0].metadata)
print("--- First 300 characters of page 1 ---")
print(pages[0].page_content[:300])

# 2. Split into chunks
chunks = split_documents(pages)
print()
print("Number of chunks:", len(chunks))
print("Length of chunk 1:", len(chunks[0].page_content))
print("--- Chunk 1 ---")
print(chunks[0].page_content)
print("--- Chunk 2 ---")
print(chunks[1].page_content)