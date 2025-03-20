import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, ScoredPoint
import ollama

# Initialize Qdrant Client (Replace with your Qdrant instance details)
qdrant_client = QdrantClient(host="localhost", port=6333)

# Constants
VECTOR_DIM = 768
COLLECTION_NAME = "embedding_collection"

# Ensure the collection exists (create if needed)
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={"size": VECTOR_DIM, "distance": "Cosine"},
)


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Generate an embedding for the given text."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):
    """Search for the most similar embeddings in Qdrant."""
    query_embedding = get_embedding(query)

    try:
        # Perform the vector search in Qdrant
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,  # Get the top K results
        )

        # Process search results
        top_results = [
            {
                "file": result.payload.get("file", "Unknown file"),
                "page": result.payload.get("page", "Unknown page"),
                "chunk": result.payload.get("chunk", "Unknown chunk"),
                "similarity": result.score,  # Cosine similarity score
            }
            for result in search_results
        ]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']:.4f}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):
    """Generate a response using retrieved context."""
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    print(f"context_str: {context_str}")

    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model="llama3.2:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search()