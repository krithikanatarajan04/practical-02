## DS 4300 Example - from docs

import ollama
import chromadb
import numpy as np
import os
import fitz

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="embedding_index")

VECTOR_DIM = 768

# Used to clear the Chroma database
def clear_chroma_store():
    print("Clearing existing Chroma store...")
    chroma_client.delete_collection("embedding_index")
    print("Chroma store cleared.")

# Function to create a new collection (equivalent to an index)
def create_collection():
    global collection
    collection = chroma_client.get_or_create_collection(name="embedding_index")
    print("Collection created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store the embedding in Chroma
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    key = f"{file}_page_{page}_chunk_{chunk}"
    collection.add(
        ids=[key],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}]
    )
    print(f"Stored embedding for: {chunk}")


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir):

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


# Query Chroma for similar embeddings
def query_chroma(query_text: str):
    query_text = "Efficient search in vector databases"
    embedding = get_embedding(query_text)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=5
    )

    for i, (doc_id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
        print(f"{doc_id} \n ----> {distance}\n")


def main():
    clear_chroma_store()
    create_collection()

    process_pdfs("../data/")
    print("\n---Done processing PDFs---\n")
    query_chroma("What is the capital of France?")


if __name__ == "__main__":
    main()
