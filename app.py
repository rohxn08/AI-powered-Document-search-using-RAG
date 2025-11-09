"""
Streamlit web interface for AI-Powered Document Search using RAG.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
try:
    import google.generativeai as genai
    _GEMINI_IMPORT_OK = True
except Exception:
    genai = None
    _GEMINI_IMPORT_OK = False

from ingestion.extract_text import DocumentExtractor
from ingestion.chunker import TextChunker
from ingestion.embed_store import EmbeddingStore
from retrieval.rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="AI Document Search & QA",
    page_icon="🧠",
    layout="wide"
)


load_dotenv()

# Initialize session state
if 'embedding_store' not in st.session_state:
    st.session_state.embedding_store = None
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'documents_indexed' not in st.session_state:
    st.session_state.documents_indexed = []


def initialize_rag_system(use_openai_embeddings: bool = False, 
                          openai_api_key: str = None,
                          use_openai_llm: bool = False,
                          llm_model: str = "gemini-2.5-flash",
                          llm_provider: str = "gemini",
                          gemini_api_key: str = None):
    """Initialize RAG system components."""
    try:
        # Initialize embedding store
        embedding_store = EmbeddingStore(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_store_path="models/vector_store",
            use_openai=use_openai_embeddings,
            openai_api_key=openai_api_key
        )
        
        # Initialize RAG pipeline with provider
        provider = (llm_provider or "gemini").lower()
        if provider == "gemini":
            api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("Google Gemini API key required when using Gemini LLM")
                return False
            rag_pipeline = RAGPipeline(
                embedding_store=embedding_store,
                llm_model=llm_model,
                llm_provider="gemini",
                gemini_api_key=api_key
            )
        else:
            st.error(f"Unsupported LLM provider: {provider}")
            return False
        
        st.session_state.embedding_store = embedding_store
        st.session_state.rag_pipeline = rag_pipeline
        
        return True
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return False


def process_document(file_path: str = None, url: str = None):
    """Process a document: extract, chunk, and store."""
    try:
        extractor = DocumentExtractor()
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        
        # Extract text
        if file_path:
            doc_data = extractor.extract_from_file(file_path)
        elif url:
            doc_data = extractor.extract_from_url(url)
        else:
            return False, "No file or URL provided"
        
        # Chunk text
        chunks = chunker.chunk_text(doc_data['text'], metadata=doc_data['metadata'])
        
        # Store chunks in embedding store
        if st.session_state.embedding_store is None:
            if not initialize_rag_system():
                return False, "Failed to initialize RAG system"
        
        embedding_store = st.session_state.embedding_store
        rag_pipeline = st.session_state.rag_pipeline
        
        # Get current FAISS index size before adding
        start_index = embedding_store.index.ntotal if embedding_store.index else 0
        
        # Add to vector store first (this updates the index)
        embedding_store.add_chunks(chunks)
        
        # Save chunks for retrieval with correct starting index
        rag_pipeline.save_chunks(chunks, start_index=start_index)
        
        # Save the updated index
        embedding_store.save_index()
        
        # Update session state
        doc_name = doc_data['metadata'].get('filename', url if url else 'Document')
        st.session_state.documents_indexed.append({
            'name': doc_name,
            'type': doc_data['metadata'].get('type', 'unknown'),
            'chunks': len(chunks)
        })
        
        return True, f"Successfully processed {len(chunks)} chunks from {doc_name}"
        
    except Exception as e:
        return False, f"Error processing document: {str(e)}"


def main():
    """Main application."""
    st.title("🧠 AI-Powered Document Search & Question Answering")
    st.markdown("Upload documents and ask questions using RAG (Retrieval-Augmented Generation)")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Provider fixed to Gemini
        st.caption("Using Google Gemini for generation")
        provider = "Gemini"
        
        st.markdown("---")
        
        gemini_env_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
        if gemini_env_key:
            st.info("✅ Google Gemini API key found in environment")
        else:
            st.error("❌ No Google Gemini API key found. Add it to your .env as GOOGLE_API_KEY.")

        # Show current key source/status
        effective_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if effective_key:
            st.caption("Gemini key detected from environment ✅")
        else:
            st.warning("No Gemini API key detected. Configure .env and restart.")
        llm_model = st.selectbox(
            "LLM Model",
            ["gemini-2.5-flash", "gemini-2.5-pro"],
            index=0
        )
        selected_provider = "gemini"
        selected_openai_key = None
        selected_gemini_key = None
        
        if st.button("Initialize System"):
            with st.spinner("Initializing RAG system..."):
                if initialize_rag_system(
                    use_openai_embeddings=False,
                    openai_api_key=None,
                    use_openai_llm=False,
                    llm_model=llm_model,
                    llm_provider=selected_provider,
                    gemini_api_key=None
                ):
                    st.success("✅ System initialized!")
        
        st.divider()
        
        st.header("📚 Indexed Documents")
        if st.session_state.documents_indexed:
            for doc in st.session_state.documents_indexed:
                st.text(f"• {doc['name']} ({doc['chunks']} chunks)")
        else:
            st.text("No documents indexed yet")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📄 Upload Documents", "❓ Ask Questions", "📜 History"])
    
    # Tab 1: Document Upload
    with tab1:
        st.header("Upload or Index Documents")
        
        upload_method = st.radio(
            "Select upload method:",
            ["Upload File", "Enter URL"],
            horizontal=True
        )
        
        if upload_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'txt', 'docx'],
                help="Supported formats: PDF, TXT, DOCX"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        success, message = process_document(file_path=tmp_path)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        else:  # URL method
            url = st.text_input("Enter document URL", placeholder="https://example.com/document.pdf")
            
            if url and st.button("Process URL"):
                with st.spinner("Downloading and processing..."):
                    success, message = process_document(url=url)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    # Tab 2: Question Answering
    with tab2:
        st.header("Ask Questions About Your Documents")
        
        if st.session_state.rag_pipeline is None:
            st.warning("⚠️ Please initialize the system in the sidebar first, then upload documents.")
        elif len(st.session_state.documents_indexed) == 0:
            st.warning("⚠️ Please upload and index documents first.")
        else:
            question = st.text_input(
                "Enter your question:",
                placeholder="What are the key points discussed in the document?"
            )
            
            k_chunks = st.slider("Number of chunks to retrieve", 1, 10, 5)
            
            if question and st.button("Get Answer"):
                with st.spinner("Searching documents and generating answer..."):
                    response = st.session_state.rag_pipeline.query(question, k=k_chunks)
                    
                    # Display answer
                    st.subheader("Answer")
                    st.write(response['answer'])
                    
                    # Display sources
                    if response.get('sources'):
                        st.subheader("Sources")
                        for i, source in enumerate(response['sources'], 1):
                            with st.expander(f"Source {i}: {source.get('filename', 'Unknown')}"):
                                st.text(f"Document: {source.get('source', 'Unknown')}")
                                st.text(f"Chunk: {source.get('chunk_index', 'N/A')}")
                        
                        # Show context used
                        st.subheader("Context Used")
                        for i, context in enumerate(response.get('context_used', [])[:3], 1):
                            with st.expander(f"Context Excerpt {i}"):
                                st.text(context[:500] + "..." if len(context) > 500 else context)
                    
                    # Save to history
                    st.session_state.conversation_history.append({
                        'question': question,
                        'answer': response['answer'],
                        'sources': response.get('sources', [])
                    })
    
    # Tab 3: Conversation History
    with tab3:
        st.header("Conversation History")
        
        if st.session_state.conversation_history:
            for i, conversation in enumerate(reversed(st.session_state.conversation_history), 1):
                with st.expander(f"Q{i}: {conversation['question'][:50]}..."):
                    st.write("**Question:**", conversation['question'])
                    st.write("**Answer:**", conversation['answer'])
                    if conversation.get('sources'):
                        st.write("**Sources:**", len(conversation['sources']), "documents")
        else:
            st.info("No conversation history yet. Start asking questions!")


if __name__ == "__main__":
    main()

