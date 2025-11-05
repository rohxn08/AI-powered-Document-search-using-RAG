

import os
import pickle
from typing import List, Dict, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False


class EmbeddingStore:
    """Generate embeddings and manage vector store."""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_store_path: str = "models/vector_store",
                 use_openai: bool = False,
                 openai_api_key: Optional[str] = None):
        """
        Initialize embedding store.
        
        Args:
            embedding_model: Model name for sentence-transformers or OpenAI model
            vector_store_path: Path to store vector database
            use_openai: Whether to use OpenAI embeddings
            openai_api_key: OpenAI API key (if using OpenAI)
        """
        self.vector_store_path = vector_store_path
        self.use_openai = use_openai
        self.embedding_model_name = embedding_model
        
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Initialize embedding model
        if use_openai:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai library required. Install with: pip install openai")
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)
            else:
                self.openai_client = OpenAI()  # Uses OPENAI_API_KEY env var
            self.embedding_model = None
            self.embedding_dim = 1536  # OpenAI ada-002 dimension
        else:
            self.openai_client = None
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS
        if faiss is None:
            raise ImportError("faiss-cpu required. Install with: pip install faiss-cpu")
        
        self.index = None
        self.chunks_metadata = []
        self.load_or_create_index()
    
    def load_or_create_index(self):
        """Load existing FAISS index or create a new one."""
        index_path = os.path.join(self.vector_store_path, "faiss.index")
        metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # Load existing index
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.chunks_metadata = pickle.load(f)
            print(f"Loaded existing index with {len(self.chunks_metadata)} chunks")
        else:
            # Create new index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.chunks_metadata = []
            print("Created new FAISS index")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text."""
        if self.use_openai:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        else:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of chunks with 'text' and 'metadata'
        """
        if not chunks:
            return
        
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = []
        
        if self.use_openai:
            # OpenAI handles batching
            for text in texts:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
        else:
            # Batch encode with sentence-transformers
            embeddings = self.embedding_model.encode(texts, 
                                                     show_progress_bar=True,
                                                     convert_to_numpy=True)
            embeddings = [emb.astype(np.float32) for emb in embeddings]
        
        # Add to FAISS index
        embeddings_array = np.vstack(embeddings)
        self.index.add(embeddings_array)
        
        # Store metadata
        for chunk in chunks:
            self.chunks_metadata.append(chunk['metadata'])
        
        print(f"Added {len(chunks)} chunks to vector store. Total: {len(self.chunks_metadata)}")
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        index_path = os.path.join(self.vector_store_path, "faiss.index")
        metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
        
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunks_metadata, f)
        
        print(f"Saved index and metadata to {self.vector_store_path}")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of results with 'chunk', 'metadata', and 'score'
        """
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks_metadata):
                # Find the original chunk text - we need to store this
                # For now, return metadata and distance
                results.append({
                    'metadata': self.chunks_metadata[idx],
                    'distance': float(dist),
                    'index': int(idx)
                })
        
        return results
    
    def get_chunk_by_index(self, index: int) -> Optional[str]:
        """Get chunk text by index. Note: We need to store chunks separately."""
        # This is a limitation - we should store chunks separately
        # For now, return None and handle in RAG pipeline
        return None

