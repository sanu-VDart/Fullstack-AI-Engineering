import os
import PyPDF2
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class PDFProcessor:
    """Handles PDF text extraction and chunking."""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Extract text from PDF and split into chunks."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
            
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

        if not text.strip():
            return []

        # Create documents with metadata
        filename = os.path.basename(pdf_path)
        chunks = self.text_splitter.split_text(text)
        
        documents = [
            Document(
                page_content=chunk, 
                metadata={"source": filename, "page": i//1} # Simplistic page tracking
            ) 
            for i, chunk in enumerate(chunks)
        ]
        
        return documents
