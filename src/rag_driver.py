#!/usr/bin/env python3
"""
RAG System Driver Script - Configurable Vector Database and LLM Interface

This script allows you to choose:
1. Vector Database (Redis, Chroma, Qdrant)
2. Embedding Model
3. LLM for retrieval and generation
4. Text chunking parameters

It handles both the ingestion and search processes based on your preferences.
"""

import os
import sys
import fitz
import ollama
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Import conditionally based on user's database choice
try:
    import redis
    from redis.commands.search.query import Query
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Constants
VECTOR_DIM = 768  # Default dimension for embeddings


class RAGSystem:
    def __init__(
        self,
        vector_db_type: str,
        llm_model: str,
        embedding_model: str,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        data_dir: str = "../data/",
    ):
        self.vector_db_type = vector_db_type.lower()
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data_dir = data_dir
        
        # Initialize the appropriate vector database
        self._init_vector_db()
    
    def _init_vector_db(self):
        """Initialize the selected vector database."""
        if self.vector_db_type == "redis":
            if not REDIS_AVAILABLE:
                raise ImportError("Redis package not installed. Run 'pip install redis'")
            
            self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
            self.INDEX_NAME = "embedding_index"
            self.DOC_PREFIX = "doc:"
            self.DISTANCE_METRIC = "COSINE"
        
        elif self.vector_db_type == "chroma":
            if not CHROMA_AVAILABLE:
                raise ImportError("ChromaDB package not installed. Run 'pip install chromadb'")
            
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(name="embedding_index")
        
        elif self.vector_db_type == "qdrant":
            if not QDRANT_AVAILABLE:
                raise ImportError("Qdrant package not installed. Run 'pip install qdrant-client'")
            
            self.qdrant_client = QdrantClient(host="localhost", port=6333)
            self.COLLECTION_NAME = "embedding_collection"
            self.DISTANCE_METRIC = Distance.COSINE
        
        else:
            raise ValueError(f"Unsupported vector database: {self.vector_db_type}")
    
    def clear_database(self):
        """Clear the existing database."""
        print(f"Clearing existing {self.vector_db_type} database...")
        
        if self.vector_db_type == "redis":
            self.redis_client.flushdb()
        
        elif self.vector_db_type == "chroma":
            try:
                self.chroma_client.delete_collection("embedding_index")
                self.collection = self.chroma_client.get_or_create_collection(name="embedding_index")
            except Exception as e:
                print(f"Error clearing Chroma collection: {e}")
        
        elif self.vector_db_type == "qdrant":
            try:
                self.qdrant_client.delete_collection(collection_name=self.COLLECTION_NAME)
            except Exception as e:
                print(f"Error clearing Qdrant collection: {e}")
        
        print(f"{self.vector_db_type.capitalize()} database cleared.")
    
    def create_database(self):
        """Create a new database or collection."""
        print(f"Creating new {self.vector_db_type} database...")
        
        if self.vector_db_type == "redis":
            try:
                self.redis_client.execute_command(f"FT.DROPINDEX {self.INDEX_NAME} DD")
            except redis.exceptions.ResponseError:
                pass
            
            self.redis_client.execute_command(
                f"""
                FT.CREATE {self.INDEX_NAME} ON HASH PREFIX 1 {self.DOC_PREFIX}
                SCHEMA text TEXT
                embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {self.DISTANCE_METRIC}
                """
            )
        
        elif self.vector_db_type == "chroma":
            self.collection = self.chroma_client.get_or_create_collection(name="embedding_index")
        
        elif self.vector_db_type == "qdrant":
            try:
                self.qdrant_client.delete_collection(collection_name=self.COLLECTION_NAME)
            except Exception:
                pass
            
            self.qdrant_client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_DIM,
                    distance=self.DISTANCE_METRIC,
                ),
            )
        
        print(f"{self.vector_db_type.capitalize()} database created successfully.")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding using the specified model."""
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]
    
    def store_embedding(self, file: str, page: str, chunk: str, embedding: List[float]):
        """Store the embedding in the appropriate database."""
        if self.vector_db_type == "redis":
            key = f"{self.DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
            self.redis_client.hset(
                key,
                mapping={
                    "file": file,
                    "page": page,
                    "chunk": chunk,
                    "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                },
            )
        
        elif self.vector_db_type == "chroma":
            key = f"{file}_page_{page}_chunk_{chunk}"
            self.collection.add(
                ids=[key],
                embeddings=[embedding],
                metadatas=[{"file": file, "page": page, "chunk": chunk}]
            )
        
        elif self.vector_db_type == "qdrant":
            vector = np.array(embedding, dtype=np.float32).tolist()
            point_id = abs(hash(f"{file}_page_{page}_chunk_{chunk}"))
            
            self.qdrant_client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={"file": file, "page": page, "chunk": chunk},
                    )
                ]
            )
        
        print(f"Stored embedding for chunk from {file}, page {page}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Extract text from a PDF file."""
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text_by_page.append((page_num, page.get_text()))
        return text_by_page
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks of appropriate size with overlap."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
        return chunks
    
    def process_pdfs(self):
        """Process all PDF files in the data directory."""
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist.")
            return
        
        pdf_count = 0
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".pdf"):
                pdf_count += 1
                pdf_path = os.path.join(self.data_dir, file_name)
                text_by_page = self.extract_text_from_pdf(pdf_path)
                
                for page_num, text in text_by_page:
                    chunks = self.split_text_into_chunks(text)
                    for chunk_index, chunk in enumerate(chunks):
                        embedding = self.get_embedding(chunk)
                        self.store_embedding(
                            file=file_name,
                            page=str(page_num),
                            chunk=str(chunk),
                            embedding=embedding,
                        )
                
                print(f" -----> Processed {file_name}")
        
        if pdf_count == 0:
            print(f"No PDF files found in {self.data_dir}")
        else:
            print(f"\n---Processed {pdf_count} PDF files---\n")
    
    def search_embeddings(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar embeddings in the database."""
        query_embedding = self.get_embedding(query)
        
        if self.vector_db_type == "redis":
            query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
            
            try:
                q = (
                    Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
                    .sort_by("vector_distance")
                    .return_fields("id", "file", "page", "chunk", "vector_distance")
                    .dialect(2)
                )
                
                results = self.redis_client.ft(self.INDEX_NAME).search(
                    q, query_params={"vec": query_vector}
                )
                
                top_results = [
                    {
                        "file": getattr(result, "file", "Unknown file"),
                        "page": getattr(result, "page", "Unknown page"),
                        "chunk": getattr(result, "chunk", "Unknown chunk"),
                        "similarity": getattr(result, "vector_distance", 0),
                    }
                    for result in results.docs
                ][:top_k]
            
            except Exception as e:
                print(f"Redis search error: {e}")
                return []
        
        elif self.vector_db_type == "chroma":
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                
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
            
            except Exception as e:
                print(f"Chroma search error: {e}")
                return []
        
        elif self.vector_db_type == "qdrant":
            try:
                search_results = self.qdrant_client.search(
                    collection_name=self.COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=top_k,
                    with_payload=True
                )
                
                top_results = [
                    {
                        "file": result.payload.get("file", "Unknown file"),
                        "page": result.payload.get("page", "Unknown page"),
                        "chunk": result.payload.get("chunk", "Unknown chunk"),
                        "similarity": result.score,
                    }
                    for result in search_results
                ]
            
            except Exception as e:
                print(f"Qdrant search error: {e}")
                return []
        
        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, "
                f"Similarity: {float(result.get('similarity', 0)):.4f}"
            )
        
        return top_results
    
    def generate_rag_response(self, query: str, context_results: List[Dict[str, Any]]) -> str:
        """Generate a response using the LLM and retrieved context."""
        context_str = "\n".join(
            [
                f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
                f"with similarity {float(result.get('similarity', 0)):.2f}"
                for result in context_results
            ]
        )
        
        prompt = f"""You are a helpful AI assistant. 
        Use the following context to answer the query as accurately as possible. If the context is 
        not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""
        
        response = ollama.chat(
            model=self.llm_model, messages=[{"role": "user", "content": prompt}]
        )
        
        return response["message"]["content"]
    
    def ingest_documents(self):
        """Ingest documents into the vector database."""
        self.clear_database()
        self.create_database()
        self.process_pdfs()
    
    def interactive_search(self):
        """Run an interactive search interface."""
        print(f"🔍 RAG Search Interface ({self.vector_db_type.capitalize()} + {self.llm_model})")
        print("Type 'exit' to quit")
        
        while True:
            query = input("\nEnter your search query: ")
            
            if query.lower() == "exit":
                break
            
            # Search for relevant embeddings
            context_results = self.search_embeddings(query)
            
            if not context_results:
                print("\n--- No relevant results found ---")
                continue
            
            # Generate RAG response
            response = self.generate_rag_response(query, context_results)
            
            print("\n--- Response ---")
            print(response)


def get_user_preferences() -> Dict[str, Any]:
    """Get user preferences for the RAG system."""
    # Available options
    db_options = ["redis", "chroma", "qdrant"]
    llm_options = ["llama3.2:latest", "phi3:latest", "mistral:latest", "gemma:latest"]
    embedding_options = ["nomic-embed-text", "all-MiniLM-L6-v2"]
    
    # Check which databases are available
    available_dbs = []
    if REDIS_AVAILABLE:
        available_dbs.append("redis")
    if CHROMA_AVAILABLE:
        available_dbs.append("chroma")
    if QDRANT_AVAILABLE:
        available_dbs.append("qdrant")
    
    if not available_dbs:
        print("Error: No vector database packages are installed.")
        print("Please install at least one of: redis, chromadb, qdrant-client")
        sys.exit(1)
    
    # Get user input for vector database
    print("\n=== RAG System Configuration ===")
    
    # Vector database selection
    print("\nAvailable Vector Databases:")
    for i, db in enumerate(available_dbs, 1):
        print(f"{i}. {db.capitalize()}")
    
    while True:
        try:
            db_choice = int(input("\nSelect vector database (number): "))
            if 1 <= db_choice <= len(available_dbs):
                vector_db = available_dbs[db_choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(available_dbs)}")
        except ValueError:
            print("Please enter a valid number")
    
    # LLM model selection
    print("\nAvailable LLM Models (via Ollama):")
    for i, llm in enumerate(llm_options, 1):
        print(f"{i}. {llm}")
    
    while True:
        try:
            llm_choice = int(input("\nSelect LLM model (number): "))
            if 1 <= llm_choice <= len(llm_options):
                llm_model = llm_options[llm_choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(llm_options)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Embedding model selection
    print("\nAvailable Embedding Models:")
    for i, emb in enumerate(embedding_options, 1):
        print(f"{i}. {emb}")
    
    while True:
        try:
            emb_choice = int(input("\nSelect embedding model (number): "))
            if 1 <= emb_choice <= len(embedding_options):
                embedding_model = embedding_options[emb_choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(embedding_options)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Chunking parameters
    while True:
        try:
            chunk_size = int(input("\nEnter chunk size (words, recommended 300-500): "))
            if chunk_size > 0:
                break
            else:
                print("Chunk size must be positive")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        try:
            chunk_overlap = int(input("Enter chunk overlap (words, recommended 50-100): "))
            if 0 <= chunk_overlap < chunk_size:
                break
            else:
                print(f"Chunk overlap must be between 0 and {chunk_size-1}")
        except ValueError:
            print("Please enter a valid number")
    
    # Data directory
    data_dir = input("\nEnter data directory path (default: ../data/): ").strip() or "../data/"
    
    # Action selection
    print("\nAvailable Actions:")
    print("1. Ingest documents + Search")
    print("2. Search only (use existing index)")
    
    while True:
        try:
            action_choice = int(input("\nSelect action (number): "))
            if action_choice in [1, 2]:
                break
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Please enter a valid number")
    
    return {
        "vector_db": vector_db,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "data_dir": data_dir,
        "action": "ingest_and_search" if action_choice == 1 else "search_only",
    }


def main():
    """Main function to run the RAG system."""
    preferences = get_user_preferences()
    
    try:
        # Initialize the RAG system with user preferences
        rag_system = RAGSystem(
            vector_db_type=preferences["vector_db"],
            llm_model=preferences["llm_model"],
            embedding_model=preferences["embedding_model"],
            chunk_size=preferences["chunk_size"],
            chunk_overlap=preferences["chunk_overlap"],
            data_dir=preferences["data_dir"],
        )
        
        # Perform the selected action
        if preferences["action"] == "ingest_and_search":
            print("\n=== Ingesting Documents ===")
            rag_system.ingest_documents()
        
        print("\n=== Starting Search Interface ===")
        rag_system.interactive_search()
    
    except KeyboardInterrupt:
        print("\nExiting RAG system...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()