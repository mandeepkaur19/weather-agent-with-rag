"""PDF processing module for extracting and chunking text from PDF documents."""
import os
from typing import List, Dict, Optional
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, PDF_UPLOAD_DIR


class PDFProcessor:
    """Processor for extracting and chunking text from PDF files."""
    
    def __init__(self):
        """Initialize PDFProcessor with text splitter."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self._ensure_upload_dir()
    
    def _ensure_upload_dir(self):
        """Ensure the upload directory exists."""
        os.makedirs(PDF_UPLOAD_DIR, exist_ok=True)
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        chunks = self.text_splitter.split_text(text)
        
        chunked_docs = []
        for idx, chunk in enumerate(chunks):
            chunk_metadata = {
                "chunk_index": idx,
                "total_chunks": len(chunks),
                **(metadata or {})
            }
            chunked_docs.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        return chunked_docs
    
    def process_pdf(self, pdf_path: str, source_name: Optional[str] = None) -> List[Dict]:
        """
        Process a PDF file: extract text and chunk it.
        
        Args:
            pdf_path: Path to the PDF file
            source_name: Optional name for the source document
            
        Returns:
            List of chunked documents with metadata
        """
        text = self.extract_text(pdf_path)
        metadata = {
            "source": source_name or os.path.basename(pdf_path),
            "file_path": pdf_path
        }
        return self.chunk_text(text, metadata)

