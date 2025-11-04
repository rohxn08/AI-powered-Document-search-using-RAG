"""
Text chunking module with overlapping segments.
"""

from typing import List, Dict
import tiktoken


class TextChunker:
    """Chunk text into overlapping segments for embedding."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, encoding_name: str = "cl100k_base"):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            encoding_name: Tokenizer encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except:
            # Fallback to a simple character-based chunking if tiktoken fails
            self.tokenizer = None
            print("Warning: tiktoken not available, using character-based chunking")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunks, each with 'text' and 'metadata'
        """
        if metadata is None:
            metadata = {}
        
        # If no tokenizer, use character-based chunking
        if self.tokenizer is None:
            return self._chunk_by_characters(text, metadata)
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.chunk_size:
            # Text is small enough, return as single chunk
            return [{
                'text': text,
                'metadata': {
                    **metadata,
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            }]
        
        chunks = []
        total_chunks = (len(tokens) - self.chunk_overlap) // (self.chunk_size - self.chunk_overlap) + 1
        
        start_idx = 0
        chunk_index = 0
        
        while start_idx < len(tokens):
            end_idx = start_idx + self.chunk_size
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk with metadata
            chunk = {
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': chunk_index,
                    'total_chunks': total_chunks,
                    'start_token': start_idx,
                    'end_token': min(end_idx, len(tokens))
                }
            }
            
            chunks.append(chunk)
            chunk_index += 1
            
            # Move start position (with overlap)
            start_idx = end_idx - self.chunk_overlap
            
            # Prevent infinite loop
            if start_idx >= len(tokens):
                break
        
        # Update total_chunks in all chunks
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
    
    def _chunk_by_characters(self, text: str, metadata: Dict) -> List[Dict]:
        """Fallback character-based chunking."""
        # Approximate: 4 characters per token
        char_chunk_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4
        
        if len(text) <= char_chunk_size:
            return [{
                'text': text,
                'metadata': {
                    **metadata,
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            }]
        
        chunks = []
        start_idx = 0
        chunk_index = 0
        
        while start_idx < len(text):
            end_idx = start_idx + char_chunk_size
            chunk_text = text[start_idx:end_idx]
            
            chunk = {
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': chunk_index,
                    'start_char': start_idx,
                    'end_char': min(end_idx, len(text))
                }
            }
            
            chunks.append(chunk)
            chunk_index += 1
            start_idx = end_idx - char_overlap
            
            if start_idx >= len(text):
                break
        
        # Update total_chunks
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks

