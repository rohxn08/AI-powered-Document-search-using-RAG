"""
Document text extraction module.
Supports PDF, TXT, DOCX files and URLs.
"""

import os
import re
from typing import Optional, Dict, List, Any
import requests
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


class DocumentExtractor:
    """Extract text from various document formats."""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.txt', '.docx'}
        
    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with 'text', 'metadata', and 'pages' (for PDFs)
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif extension == '.txt':
            return self._extract_from_txt(file_path)
        elif extension == '.docx':
            return self._extract_from_docx(file_path)
    
    def extract_from_url(self, url: str) -> Dict[str, Any]:
        """
        Extract text from a URL.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Dictionary with 'text' and 'metadata'
        """
        try:
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            # Try to extract from PDF if URL points to PDF
            if url.lower().endswith('.pdf') or 'pdf' in response.headers.get('content-type', '').lower():
                # Save temporarily and extract
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name
                
                try:
                    result = self._extract_from_pdf(Path(tmp_path))
                    result['metadata']['source'] = url
                    return result
                finally:
                    os.unlink(tmp_path)
            
            # Otherwise treat as HTML/text
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text)
            
            return {
                'text': text,
                'metadata': {
                    'source': url,
                    'title': soup.title.string if soup.title else url,
                    'type': 'url'
                },
                'pages': []
            }
        except Exception as e:
            raise Exception(f"Error extracting from URL: {str(e)}")
    
    def _extract_from_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF file."""
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required for PDF extraction. Install with: pip install PyMuPDF")
        
        doc = fitz.open(file_path)
        pages_text = []
        full_text = []
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            pages_text.append({
                'page_number': page_num + 1,
                'text': text
            })
            full_text.append(text)
        
        doc.close()
        
        combined_text = '\n\n'.join(full_text)
        
        return {
            'text': combined_text,
            'metadata': {
                'source': str(file_path),
                'filename': file_path.name,
                'type': 'pdf',
                'total_pages': total_pages
            },
            'pages': pages_text
        }
    
    def _extract_from_txt(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
        
        return {
            'text': text,
            'metadata': {
                'source': str(file_path),
                'filename': file_path.name,
                'type': 'txt'
            },
            'pages': []
        }
    
    def _extract_from_docx(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from DOCX file."""
        if DocxDocument is None:
            raise ImportError("python-docx is required for DOCX extraction. Install with: pip install python-docx")
        
        doc = DocxDocument(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        text = '\n\n'.join(paragraphs)
        
        return {
            'text': text,
            'metadata': {
                'source': str(file_path),
                'filename': file_path.name,
                'type': 'docx'
            },
            'pages': []
        }

