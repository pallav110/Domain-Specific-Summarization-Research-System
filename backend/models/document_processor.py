"""
Document Processing Module
Handles PDF extraction, text chunking, and preprocessing
"""
import PyPDF2
import pdfplumber
from typing import List, Dict, Tuple
from pathlib import Path
import re
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangChainDocument


class DocumentProcessor:
    """Process documents for summarization"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        """Initialize document processor"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str, method: str = "pdfplumber") -> str:
        """
        Extract text from PDF
        
        Args:
            pdf_path: Path to PDF file
            method: 'pdfplumber' or 'pypdf2'
            
        Returns:
            Extracted text
        """
        try:
            if method == "pdfplumber":
                return self._extract_with_pdfplumber(pdf_path)
            else:
                return self._extract_with_pypdf2(pdf_path)
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            # Try fallback method
            try:
                return self._extract_with_pypdf2(pdf_path)
            except:
                raise ValueError(f"Could not extract text from PDF: {pdf_path}")
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract using pdfplumber (better for tables and complex layouts)"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text.strip()
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract using PyPDF2 (fallback method)"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text.strip()
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?-]', '', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenation
        
        return text.strip()
    
    def detect_sections(self, text: str, domain: str = "general") -> List[Dict[str, str]]:
        """
        Detect and extract sections from document
        
        Args:
            text: Document text
            domain: Document domain (legal, medical, general)
            
        Returns:
            List of sections with headers and content
        """
        sections = []
        
        if domain == "legal":
            # Legal documents often have numbered sections
            pattern = r'(?:^|\n)((?:Article|Section|Clause|\d+\.)\s+.+?)(?=\n(?:Article|Section|Clause|\d+\.)|$)'
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                section_text = match.group(1).strip()
                lines = section_text.split('\n', 1)
                header = lines[0] if lines else "Section"
                content = lines[1] if len(lines) > 1 else section_text
                
                sections.append({
                    "header": header,
                    "content": content
                })
        
        elif domain == "medical":
            # Medical documents often have standard sections
            medical_headers = [
                "chief complaint", "history of present illness", "past medical history",
                "medications", "allergies", "physical examination", "assessment",
                "plan", "diagnosis", "treatment", "summary", "findings"
            ]
            
            for header in medical_headers:
                pattern = rf'(?:^|\n)({header}[:\s]+)(.*?)(?=\n(?:{"|".join(medical_headers)})|$)'
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                
                if match:
                    sections.append({
                        "header": match.group(1).strip(),
                        "content": match.group(2).strip()
                    })
        
        # If no sections detected, treat entire document as one section
        if not sections:
            sections.append({
                "header": "Document",
                "content": text
            })
        
        return sections
    
    def chunk_text(self, text: str, preserve_structure: bool = True) -> List[str]:
        """
        Split text into chunks for processing
        
        Args:
            text: Input text
            preserve_structure: Try to preserve sentence/paragraph boundaries
            
        Returns:
            List of text chunks
        """
        if preserve_structure:
            # Use LangChain splitter
            docs = self.text_splitter.create_documents([text])
            chunks = [doc.page_content for doc in docs]
        else:
            # Simple chunking
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.chunk_size):
                chunk = " ".join(words[i:i + self.chunk_size])
                chunks.append(chunk)
        
        return chunks
    
    def chunk_with_metadata(self, text: str, domain: str = "general") -> List[Dict]:
        """
        Chunk text while preserving metadata
        
        Returns:
            List of dicts with 'text', 'chunk_id', 'domain', 'section'
        """
        sections = self.detect_sections(text, domain)
        chunks_with_metadata = []
        
        chunk_id = 0
        for section in sections:
            section_chunks = self.chunk_text(section['content'])
            
            for chunk in section_chunks:
                chunks_with_metadata.append({
                    'chunk_id': chunk_id,
                    'text': chunk,
                    'section': section['header'],
                    'domain': domain,
                    'word_count': len(chunk.split())
                })
                chunk_id += 1
        
        return chunks_with_metadata
    
    def get_word_count(self, text: str) -> int:
        """Get word count of text"""
        return len(text.split())
    
    def get_statistics(self, text: str) -> Dict:
        """Get text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'char_count': len(text),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }


class LegalDocumentProcessor(DocumentProcessor):
    """Specialized processor for legal documents"""
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities (parties, dates, amounts, etc.)"""
        entities = {
            'parties': [],
            'dates': [],
            'amounts': [],
            'references': []
        }
        
        # Extract dates (simple pattern)
        date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+ \d{1,2}, \d{4})\b'
        entities['dates'] = re.findall(date_pattern, text)
        
        # Extract monetary amounts
        amount_pattern = r'\$[\d,]+(?:\.\d{2})?'
        entities['amounts'] = re.findall(amount_pattern, text)
        
        # Extract case references (simple pattern)
        case_pattern = r'\b\d{2,4}\s+\w+\s+\d+\b'
        entities['references'] = re.findall(case_pattern, text)
        
        return entities


class MedicalDocumentProcessor(DocumentProcessor):
    """Specialized processor for medical documents"""
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities (medications, diagnoses, procedures)"""
        entities = {
            'medications': [],
            'diagnoses': [],
            'measurements': []
        }
        
        # Extract measurements (BP, temp, etc.)
        measurement_pattern = r'\b\d{2,3}/\d{2,3}\s*mmHg|\b\d{2,3}\s*bpm|\b\d{2,3}(?:\.\d)?\s*°[CF]\b'
        entities['measurements'] = re.findall(measurement_pattern, text)
        
        # Simple medication detection (drugs often end in specific suffixes)
        med_pattern = r'\b\w+(?:ine|ol|azole|mycin|cillin|pril|sartan|statin)\b'
        entities['medications'] = re.findall(med_pattern, text, re.IGNORECASE)
        
        return entities
