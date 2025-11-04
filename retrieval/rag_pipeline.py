"""
RAG pipeline: Retrieval-Augmented Generation.
"""

from typing import List, Dict, Optional, Any
import os
import pickle

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

from ingestion.embed_store import EmbeddingStore


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""
    
    def __init__(self, 
                 embedding_store: EmbeddingStore,
                 llm_model: str = "gpt-3.5-turbo",
                 openai_api_key: Optional[str] = None,
                 chunks_storage_path: str = "models/chunks_storage.pkl",
                 llm_provider: str = "openai",
                 gemini_api_key: Optional[str] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_store: EmbeddingStore instance
            llm_model: LLM model name (OpenAI)
            openai_api_key: OpenAI API key
            chunks_storage_path: Path to store chunks for retrieval
        """
        self.embedding_store = embedding_store
        self.llm_model = llm_model
        self.chunks_storage_path = chunks_storage_path
        self.llm_provider = (llm_provider or "openai").lower()

        self.openai_client = None
        self.gemini_model = None

        if self.llm_provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai library required. Install with: pip install openai")
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)
            else:
                self.openai_client = OpenAI()  # Uses OPENAI_API_KEY env var
        elif self.llm_provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai required. Install with: pip install google-generativeai")
            configured_key = gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not configured_key:
                raise ValueError("Google Gemini API key is required. Set GOOGLE_API_KEY or pass gemini_api_key.")
            genai.configure(api_key=configured_key)
            self.gemini_model = genai.GenerativeModel(self.llm_model)
        else:
            raise ValueError(f"Unsupported llm_provider: {self.llm_provider}. Use 'openai' or 'gemini'.")
        
        # Load stored chunks
        self.stored_chunks = self._load_chunks()
    
    def _load_chunks(self) -> Dict[int, str]:
        """Load stored chunks from disk."""
        if os.path.exists(self.chunks_storage_path):
            with open(self.chunks_storage_path, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def save_chunks(self, chunks: List[Dict], start_index: int = None):
        """
        Save chunks to disk for later retrieval.
        
        Args:
            chunks: List of chunks to save
            start_index: Starting index (should match FAISS index). If None, uses current length.
        """
        if start_index is None:
            start_index = len(self.stored_chunks)
        
        for idx, chunk in enumerate(chunks):
            self.stored_chunks[start_index + idx] = chunk['text']
        
        os.makedirs(os.path.dirname(self.chunks_storage_path), exist_ok=True)
        with open(self.chunks_storage_path, 'wb') as f:
            pickle.dump(self.stored_chunks, f)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User question
            k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with text and metadata
        """
        results = self.embedding_store.search(query, k=k)
        
        # Enrich with chunk text
        retrieved_chunks = []
        for result in results:
            idx = result['index']
            chunk_text = self.stored_chunks.get(idx, "")
            
            retrieved_chunks.append({
                'text': chunk_text,
                'metadata': result['metadata'],
                'score': result['distance']
            })
        
        return retrieved_chunks
    
    def generate_answer(self, 
                       query: str, 
                       retrieved_chunks: List[Dict],
                       max_tokens: int = 500) -> Dict[str, any]:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            query: User question
            retrieved_chunks: Retrieved document chunks
            max_tokens: Maximum tokens for response
            
        Returns:
            Dictionary with 'answer', 'sources', and 'context_used'
        """
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for chunk in retrieved_chunks:
            context_parts.append(chunk['text'])
            source_info = {
                'filename': chunk['metadata'].get('filename', 'Unknown'),
                'chunk_index': chunk['metadata'].get('chunk_index', 0),
                'source': chunk['metadata'].get('source', 'Unknown')
            }
            sources.append(source_info)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context from documents.

Context from documents:
{context}

Question: {query}

Instructions:
1. Answer the question based ONLY on the provided context.
2. If the answer is not in the context, say "I cannot find the answer in the provided documents."
3. Be concise and accurate.
4. Cite which document/section you used (if available).

Answer:"""
        
        # Call LLM
        try:
            if self.llm_provider == "openai":
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on document context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                answer_text = response.choices[0].message.content.strip()
            else:  # gemini
                # Gemini uses a single prompt and returns text
                gemini_response = self.gemini_model.generate_content(prompt)
                answer_text = (getattr(gemini_response, 'text', None) or "").strip()
                if not answer_text and hasattr(gemini_response, 'candidates'):
                    # Fallback extraction
                    for c in gemini_response.candidates:
                        if getattr(c, 'content', None) and getattr(c.content, 'parts', None):
                            answer_text = "".join(getattr(p, 'text', '') for p in c.content.parts).strip()
                            if answer_text:
                                break

            return {
                'answer': answer_text or "I could not generate an answer.",
                'sources': sources,
                'context_used': context_parts,
                'model': f"{self.llm_provider}:{self.llm_model}"
            }
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': sources,
                'context_used': context_parts,
                'error': str(e)
            }
    
    def query(self, question: str, k: int = 5) -> Dict[str, any]:
        """
        Complete RAG pipeline: retrieve + generate.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            
        Returns:
            Complete response with answer and sources
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(question, k=k)
        
        if not retrieved_chunks:
            return {
                'answer': "No relevant documents found. Please upload and index documents first.",
                'sources': [],
                'context_used': []
            }
        
        # Generate answer
        response = self.generate_answer(question, retrieved_chunks)
        
        return response

