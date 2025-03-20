## DS 4300 Example - from docs
from qdrant_client import QdrantClient
import ollama
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
import os
import fitz

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

VECTOR_DIM = 768
COLLECTION_NAME = "embedding_collection"
DISTANCE_METRIC = Distance.COSINE


# used to clear the Qdrant collection
def clear_qdrant_store():
    print("Clearing existing Qdrant collection...")
    client.delete_collection(collection_name=COLLECTION_NAME)
    print("Qdrant collection cleared.")


# Create a collection in Qdrant
def create_qdrant_collection():
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception as e:
        print(f"Error deleting collection: {e}")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,  # Dimension of the vector
            distance=Distance.COSINE,  # Type of distance metric
        ),
    )
    print("Qdrant collection created successfully.")


# Example function to insert a document and its embedding into Qdrant
def insert_document_to_qdrant(doc_id, text, embedding):
    point = PointStruct(id=doc_id, vector=embedding, payload={"text": text})
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )
    print(f"Document {doc_id} inserted into Qdrant.")


# Example usage of inserting a document with its vector embedding
def insert_sample_document():
    doc_id = 1  # Example document ID
    text = "This is an example document."
    embedding = np.random.rand(VECTOR_DIM).astype(np.float32)  # Example random embedding

    insert_document_to_qdrant(doc_id, text, embedding)


# Example function to query Qdrant collection for similar vectors
def query_qdrant(query_vector, top_k=5):
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )

    for result in results:
        print(f"Found document ID {result.id}, score {result.score}")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def store_embedding(file: str, page: str, chunk: str, embedding: list):
    vector = np.array(embedding, dtype=np.float32).tolist()

    # Ensure a positive integer ID
    point_id = abs(hash(f"{file}_page_{page}_chunk_{chunk}"))

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={"file": file, "page": page, "chunk": chunk},
            )
        ]
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


def query_qdrant(query_text: str):
    embedding = get_embedding(query_text)  # Convert text to vector
    vector = np.array(embedding, dtype=np.float32).tolist()  # Convert to list

    # Perform vector search in Qdrant
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=5,  # Get top 5 results
        with_payload=True  # Retrieve metadata
    )

    # Print results
    for result in search_results:
        print(f"ID: {result.id} \n ----> Distance: {result.score}\n")

def main():
    clear_qdrant_store()
    create_qdrant_collection()

    process_pdfs("../data/")
    print("\n---Done processing PDFs---\n")
    query_qdrant("What is the capital of France?")


if __name__ == "__main__":
    main()