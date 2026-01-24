# AI POWERED Document Search

## Contents
- [Introduction](#1-introduction)
- [Model Summary](#2-model-summary)
- [Features](#3-features)
- [Tech Stack](#4-tech-stack)
- [Project Structure](#5-project-structure)
- [How to Run the App](#6-how-to-run-the-app)
- [Difficulties Faced](#7-difficulties-faced)
- [Future Improvements](#8-future-improvements)

## 1. Introduction
The **AI POWERED Document Search** is a robust Retrieval-Augmented Generation (RAG) system designed to allow users to chat with their documents. Whether it's a technical PDF spec, a research paper, or a live website URL, this system leverages a hybrid AI approach—combining specific semantic search (FAISS + SentenceTransformers) and modern LLM capabilities (Google Gemini 2.5)—to ensure high-accuracy answers with traceable sources.

The application features a polished Streamlit interface that supports drag-and-drop file ingestion, URL processing, and real-time interactive Q&A with transparent source citation.

## 2. Model Summary
The system employs a two-stage pipeline to handle large-scale document understanding:

### A. Generation: Google Gemini 2.5
The primary engine for generating natural language responses is the Google Gemini family of models.

- **Models Supported**: `gemini-2.5-flash` (Speed/Cost optimized), `gemini-2.5-pro` (Reasoning optimized).
- **Role**: Synthesizes answers based on the context chunks retrieved from the vector store.
- **Key Behavior**: Configured to provide concise, context-aware answers while strictly adhering to the provided source material.

### B. Retrieval: Sentence Transformers + FAISS
For semantic search and context retrieval, we rely on a high-efficiency embedding pipeline.

- **Encoder**: `sentence-transformers/all-MiniLM-L6-v2` to capture semantic meaning of text chunks.
- **Store**: FAISS (Facebook AI Similarity Search) for millisecond-scale similarity search.
- **Strategy**: Chunks documents into 500-token segments with 50-token overlap to maintain context continuity.

## 3. Features

### Dual-Import Interface
**File Ingestion**:
- **Multi-Format Support**: Native handling for PDF, DOCX, and TXT files.
- **Secure Processing**: Files are processed locally/temporarily and converted to vector embeddings without permanent storage of raw text.

**Web Processing**:
- **Direct URL Fetching**: Scrapes and parses text content directly from provided URLs for instant analysis of web pages.

### Research-Oriented Q&A
- **Source Citation**: Every answer is accompanied by an "Expandable" source list, showing exactly which document and chunk the information came from.
- **Context Transparency**: Users can view the exact raw text segments the LLM used to generate the answer, ensuring trust and verifiability.
- **History Tracking**: Automatically saves the session's Q&A history for easy reference.

### Advanced Logic
- **Lazy Loading Implementation**: Critical components (RAG Pipeline, Embedding Models) are imported only when needed, reducing initial app startup time significantly.
- **Dynamic Configuration**: Users can switch between Gemini models or update API keys on the fly via the sidebar without restarting the server.

## 4. Tech Stack

### Frontend & Application
- **Streamlit**: For the interactive web interface, session state management, and responsive layout.

### AI & Backend
- **Google Generative AI SDK**: Direct integration with Gemini models.
- **Sentence Transformers**: For generating local, high-quality text embeddings.
- **FAISS**: For efficient vector storage and similarity search.

### Infrastructure & Processing
- **PyMuPDF / Python-Docx**: Robust parsing libraries for complex document formats.
- **BeautifulSoup4**: For cleaning and extracting text from web URLs.
- **Numpy**: For handling vector operations.
- **Python-Dotenv**: for secure environment variable management.

## 5. Project Structure
```
AI POWERED Document Search/
├── app.py                     # Streamlit Frontend & Main Entry Point
├── ingestion/
│   ├── chunker.py             # Text splitting and windowing logic
│   ├── embed_store.py         # FAISS vector store wrapper
│   └── extract_text.py        # PDF/DOCX/URL parsing logic
├── retrieval/
│   └── rag_pipeline.py        # RAG orchestration (Retrieve + Generate)
├── models/
│   └── vector_store/          # Local FAISS index artifacts
├── .env                       # API Keys and Config
└── requirements.txt           # Project Dependencies
```

## 6. How to Run the App

### Prerequisites
- Python 3.10 or higher
- A Google Gemini API Key

### Quick Start
1. **Clone the repository** (if applicable) or navigate to the project folder.

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   - Create a `.env` file in the root directory.
   - Add your API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   
### Accessing the App
- **Web Interface**: http://localhost:8501

## 7. Difficulties Faced
1. **Startup Latency**: Initial loading of `sentence-transformers` and FAISS indexes caused the Streamlit app to hang on start.
   - **Solution**: We implemented "Lazy Imports" inside the initialization functions. Heavy libraries are only loaded into memory when the user clicks "Initialize System," keeping the UI snappy on first load.

2. **Context Window Limitations**: Handling large PDFs often exceeded the token limits of the embedding models.
   - **Solution**: Implemented a robust `TextChunker` with sliding windows (500 limit, 50 overlap) to ensure no information is lost at chunk boundaries.

3. **Deployment Consistency**: Managing API keys across different environments (local vs. cloud) was error-prone.
   - **Solution**: Added a hybrid check that looks for `.env` files first, but falls back to user UI input if no environment variables are detected.

## 8. Future Improvements
- **Multimodal Support**: Integrate support for Image and Video ingestion to allow "asking questions" about diagrams or visual content.
- **Persistent Vector Database**: Migrate from local FAISS files to a cloud-based solution (like Pinecone) to allow knowledge bases to persist across server restarts.
- **Conversational Memory**: Enhance the RAG pipeline to be fully conversational, where the model remembers context from previous questions in the session.
