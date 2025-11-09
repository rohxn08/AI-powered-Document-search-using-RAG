# 🧠 AI-Powered Document Search and Question Answering using RAG

A complete Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask natural language questions, with answers generated from the uploaded document content.

## 🚀 Features

- **Document Ingestion**: Upload PDFs, TXT files, DOCX files, or fetch from URLs
- **Text Extraction**: Automatic text extraction from various document formats
- **Intelligent Chunking**: Split documents into overlapping segments for better retrieval
- **Vector Embeddings**: Generate embeddings using sentence-transformers or OpenAI
- **Vector Store**: FAISS-based local vector database for fast similarity search
- **RAG Pipeline**: Retrieve relevant chunks and generate context-grounded answers
- **Web Interface**: User-friendly Streamlit interface
- **Conversation History**: Save and view Q&A history
- **Source Citation**: See which parts of documents were used for answers

## 📋 Prerequisites

### For Docker:
- Docker and Docker Compose installed
- OpenAI API key (**required** for LLM question answering)

### For Local Installation:
- Python 3.8+
- Google Gemini api key (**required** for LLM question answering)

> **💡 Need an API key?** Get one free at https://platform.openai.com/api-keys (you get free credits to start!)

## 🛠️ Installation

### Option 1: Docker (Recommended)

1. Clone or download this repository

2. Build and run with Docker Compose:
```bash
# Set your OpenAI API key (optional)
export OPENAI_API_KEY="your-api-key-here"

# Build and run
docker-compose up --build
```

Or using Docker directly:
```bash
# Build the image
docker build -t ai-doc-search-rag .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -e OPENAI_API_KEY="your-api-key-here" \
  ai-doc-search-rag
```

The application will be available at `http://localhost:8501`

### Option 2: Local Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set OpenAI API key (required for LLM):
   
   **Option A: Environment Variable (Recommended)**
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY="your-api-key-here"
   
   # Windows CMD
   set OPENAI_API_KEY=your-api-key-here
   
   # Linux/Mac
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   **Option B: .env File**
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```
   
   **Option C: Streamlit UI**
   - Enter it in the sidebar when the app runs
   
   📖 **See [SETUP_API_KEY.md](SETUP_API_KEY.md) for detailed instructions**

## 🎯 Usage

### Docker
```bash
docker-compose up
```
Access at `http://localhost:8501`

### Local
1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. **Configure the system** (in the sidebar):
   - Choose whether to use OpenAI embeddings (default: sentence-transformers)
   - Choose LLM model (default: GPT-3.5-turbo)
   - Enter OpenAI API key if using OpenAI services
   - Click "Initialize System"

3. **Upload documents**:
   - Go to "Upload Documents" tab
   - Upload PDF/TXT/DOCX files or enter a URL
   - Click "Process Document"
   - Wait for indexing to complete

4. **Ask questions**:
   - Go to "Ask Questions" tab
   - Enter your question
   - Adjust number of chunks to retrieve (default: 5)
   - Click "Get Answer"
   - View answer with source citations

5. **View history**:
   - Check "History" tab for past Q&A sessions

## 📁 Project Structure

```
ai_doc_search_rag/
│
├── app.py                 # Main Streamlit web interface
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── Dockerfile            # Docker container definition
├── docker-compose.yml    # Docker Compose configuration
├── docker-run.sh         # Convenience script (Linux/Mac)
├── docker-run.bat        # Convenience script (Windows)
├── .dockerignore         # Files to exclude from Docker build
│
├── ingestion/
│   ├── extract_text.py   # PDF/DOCX/URL text extraction
│   ├── chunker.py        # Text chunking logic
│   └── embed_store.py    # Embedding + vector store setup
│
├── retrieval/
│   └── rag_pipeline.py   # RAG: retrieve + generate answer
│
└── models/
    ├── vector_store/     # FAISS index and metadata
    └── chunks_storage.pkl # Stored chunk texts
```

## 🔧 Architecture

### Document Processing Flow

1. **Extraction**: Document → Text (using PyMuPDF, python-docx, or BeautifulSoup)
2. **Chunking**: Text → Overlapping chunks (500 tokens, 50 overlap)
3. **Embedding**: Chunks → Vector embeddings (sentence-transformers or OpenAI)
4. **Storage**: Embeddings → FAISS vector database

### Question Answering Flow

1. **Query Embedding**: Question → Vector embedding
2. **Retrieval**: Search FAISS for top-k similar chunks
3. **Context Building**: Combine retrieved chunks as context
4. **Generation**: LLM generates answer from context + question
5. **Response**: Answer + source citations

## ⚙️ Configuration Options

### Embedding Models
- **Default**: `sentence-transformers/all-MiniLM-L6-v2` (free, local)
- **OpenAI**: `text-embedding-ada-002` (requires API key)

### LLM Models
- `gpt-3.5-turbo` (default, cost-effective)
- `gpt-4` (higher quality, more expensive)
- `gpt-4-turbo-preview` (latest GPT-4 variant)

### Chunking Parameters
- Chunk size: 500 tokens
- Overlap: 50 tokens
- Adjustable in code

## 🎨 Features in Detail

### Document Formats Supported
- **PDF**: Extracts text from all pages
- **TXT**: Plain text files (UTF-8 or Latin-1)
- **DOCX**: Microsoft Word documents
- **URLs**: HTML pages or PDFs from URLs

### Vector Search
- Uses FAISS (Facebook AI Similarity Search) for fast similarity search
- L2 distance metric for finding closest embeddings
- Stores metadata (filename, chunk index, source) with each vector

### RAG Pipeline
- Retrieves top-k most relevant document chunks
- Builds context prompt with retrieved content
- Generates answers that are grounded in document content
- Provides source attribution for transparency

## 🔍 Example Use Cases

- **Research Paper Q&A**: Upload research papers and ask specific questions
- **Documentation Search**: Index documentation and find answers quickly
- **Legal Document Analysis**: Extract information from contracts or legal docs
- **Educational Content**: Upload textbooks and get study answers
- **Corporate Knowledge Base**: Index internal documents for employee queries

## 🐛 Troubleshooting

### Docker Issues
- **Port already in use**: Change the port mapping in `docker-compose.yml` (e.g., `"8502:8501"`)
- **Permission errors**: Ensure Docker has access to the `models` directory
- **Build fails**: Make sure you're in the project root directory when building
- **Memory issues**: Increase Docker memory allocation if processing large documents

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- For FAISS on Mac M1/M2: Use `faiss-cpu` (already in requirements)
- In Docker: All dependencies are pre-installed in the image

### OpenAI API Issues
- Verify API key is correct
- Check API quota/billing
- Try using sentence-transformers instead (free alternative)
- In Docker: Set `OPENAI_API_KEY` as environment variable or in docker-compose.yml

### Document Processing Errors
- Ensure PDFs are not corrupted or password-protected
- Check URL accessibility for web scraping
- Verify file formats are supported
- In Docker: Use volume mounts to access local files (see docker-compose.yml)

## 📝 License

This project is open source and available for educational and commercial use.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests.

## 🙏 Acknowledgments

- Built with Streamlit, FAISS, sentence-transformers, and OpenAI
- Inspired by RAG architecture from Retrieval-Augmented Generation papers

---

**Enjoy your AI-powered document search system! 🚀**

