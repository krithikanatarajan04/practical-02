import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
import chromadb

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="embedding_index")

VECTOR_DIM = 768


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):
    query_embedding = get_embedding(query)
    
    try:
        # Perform vector similarity search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Transform results into the expected format
        top_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            similarity = results["distances"][0][i]
            top_results.append({
                "file": metadata.get("file", "Unknown file"),
                "page": metadata.get("page", "Unknown page"),
                "chunk": metadata.get("chunk", "Unknown chunk"),
                "similarity": similarity,
            })
        
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
    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )
    
    print(f"context_str: {context_str}")
    
    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""
    
    # Generate response using Ollama
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
