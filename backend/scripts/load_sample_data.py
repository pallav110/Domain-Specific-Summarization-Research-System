"""
Script to load sample data into the system
"""
import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import init_db, async_session_maker
from models.db_models import Document, DomainType
from models.domain_classifier import DomainClassifier
from models.document_processor import DocumentProcessor

async def load_sample_data():
    """Load sample documents into the database"""
    
    # Initialize database
    await init_db()
    
    # Initialize components
    classifier = DomainClassifier()
    processor = DocumentProcessor()
    
    sample_dir = Path(__file__).parent.parent.parent / 'sample_data'
    
    documents_to_load = [
        {
            'path': sample_dir / 'legal' / 'employment_contract.txt',
            'original_name': 'Sample Employment Contract',
            'expected_domain': 'legal'
        },
        {
            'path': sample_dir / 'medical' / 'patient_record.txt',
            'original_name': 'Sample Patient Record',
            'expected_domain': 'medical'
        }
    ]
    
    async with async_session_maker() as session:
        for doc_info in documents_to_load:
            if not doc_info['path'].exists():
                print(f"Warning: File not found: {doc_info['path']}")
                continue
            
            # Read file
            with open(doc_info['path'], 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # Clean text
            raw_text = processor.clean_text(raw_text)
            word_count = processor.get_word_count(raw_text)
            
            # Classify domain
            domain, confidence, _ = classifier.classify(raw_text)
            
            # Create document
            document = Document(
                filename=doc_info['path'].name,
                original_filename=doc_info['original_name'],
                file_path=str(doc_info['path']),
                file_size=len(raw_text),
                file_type='txt',
                raw_text=raw_text,
                word_count=word_count,
                detected_domain=DomainType(domain),
                domain_confidence=confidence,
                processed=1
            )
            
            session.add(document)
            
            print(f"Loaded: {doc_info['original_name']}")
            print(f"  Domain: {domain} ({confidence:.2%} confidence)")
            print(f"  Words: {word_count}")
            print()
        
        await session.commit()
        print("Sample data loaded successfully!")

if __name__ == "__main__":
    asyncio.run(load_sample_data())
