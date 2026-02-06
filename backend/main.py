"""
FastAPI Application - Main Entry Point
"""
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
import aiofiles
import os
from pathlib import Path
from datetime import datetime
import uuid
from loguru import logger

from config import settings
from database import get_db, init_db, close_db
from models.db_models import Document, Summary, Experiment, Evaluation, DomainType, ModelType
from models.domain_classifier import DomainClassifier
from models.document_processor import DocumentProcessor
from models.summarizers import SummarizationEngine
from models.evaluator import EvaluationMetrics, ComparativeAnalysis
from schemas import (
    DocumentUploadResponse,
    DocumentDetail,
    SummarizationRequest,
    SummaryResponse,
    EvaluationResponse,
    ExperimentRequest,
    ExperimentResponse,
    ComparisonResponse,
    DashboardStats,
    ErrorResponse
)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Research system for domain-specific document summarization"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODELS_CACHE_DIR, exist_ok=True)
os.makedirs(settings.EXPERIMENT_LOG_DIR, exist_ok=True)
os.makedirs(settings.RESULTS_DIR, exist_ok=True)

# Initialize components
domain_classifier = None
document_processor = None
summarization_engine = None
evaluation_metrics = None
comparative_analysis = None


@app.on_event("startup")
async def startup_event():
    """Initialize database and models on startup"""
    global domain_classifier, document_processor, summarization_engine
    global evaluation_metrics, comparative_analysis
    
    logger.info("Starting application...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Initialize ML components
    logger.info("Initializing ML components...")
    domain_classifier = DomainClassifier()
    document_processor = DocumentProcessor(
        chunk_size=settings.MAX_CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    summarization_engine = SummarizationEngine()
    evaluation_metrics = EvaluationMetrics()
    comparative_analysis = ComparativeAnalysis()
    
    logger.info("Application started successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")
    await close_db()
    logger.info("Application shut down")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Domain-Specific Summarization Research System",
        "version": settings.VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================== DOCUMENT ENDPOINTS ====================

@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a document (PDF or TXT)
    Automatically extracts text and classifies domain
    """
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        file_size = len(content)
        
        # Check file size
        if file_size > settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_DOCUMENT_SIZE_MB}MB"
            )
        
        # Extract text
        if file_ext == ".pdf":
            raw_text = document_processor.extract_text_from_pdf(file_path)
        elif file_ext == ".txt":
            raw_text = document_processor.extract_text_from_txt(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Clean text
        raw_text = document_processor.clean_text(raw_text)
        word_count = document_processor.get_word_count(raw_text)
        
        # Classify domain
        domain, confidence, _ = domain_classifier.classify(raw_text)
        
        # Create document record
        document = Document(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file_ext[1:],  # Remove dot
            raw_text=raw_text,
            word_count=word_count,
            detected_domain=DomainType(domain),
            domain_confidence=confidence,
            processed=1
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        logger.info(f"Document uploaded: {file.filename} -> {domain} ({confidence:.2f})")
        
        return DocumentUploadResponse(
            document_id=document.id,
            filename=file.filename,
            file_size=file_size,
            word_count=word_count,
            detected_domain=DomainType(domain),
            domain_confidence=confidence,
            message="Document uploaded and processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: int, db: AsyncSession = Depends(get_db)):
    """Get document details"""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document


@app.get("/api/v1/documents", response_model=List[DocumentDetail])
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    domain: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all documents with optional filtering"""
    query = select(Document)
    
    if domain:
        query = query.where(Document.detected_domain == DomainType(domain))
    
    query = query.offset(skip).limit(limit).order_by(Document.upload_timestamp.desc())
    
    result = await db.execute(query)
    documents = result.scalars().all()
    
    return documents


# ==================== SUMMARIZATION ENDPOINTS ====================

@app.post("/api/v1/summarize", response_model=List[SummaryResponse])
async def generate_summaries(
    request: SummarizationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate summaries using specified models
    """
    try:
        # Get document
        result = await db.execute(
            select(Document).where(Document.id == request.document_id)
        )
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Determine models to use
        models_to_use = request.model_types
        
        # Add domain-specific model if requested
        if request.use_domain_specific and document.detected_domain != DomainType.UNKNOWN:
            if document.detected_domain == DomainType.LEGAL:
                models_to_use.append(ModelType.LEGAL_BERT_PEGASUS)
            elif document.detected_domain == DomainType.MEDICAL:
                models_to_use.append(ModelType.CLINICAL_BERT_PEGASUS)
        
        summaries = []
        
        # Generate summaries
        for model_type in models_to_use:
            try:
                logger.info(f"Generating summary with {model_type}")
                
                result_dict = summarization_engine.summarize(
                    text=document.raw_text,
                    model_type=model_type.value,
                    domain=document.detected_domain.value,
                    max_length=request.max_length,
                    min_length=request.min_length
                )
                
                # Create summary record
                summary = Summary(
                    document_id=document.id,
                    model_type=ModelType(result_dict['model_type']),
                    model_name=result_dict['model_name'],
                    summary_text=result_dict['summary'],
                    summary_length=result_dict['summary_length'],
                    generation_time=result_dict['generation_time'],
                    generation_params={
                        'max_length': request.max_length,
                        'min_length': request.min_length
                    }
                )
                
                db.add(summary)
                await db.commit()
                await db.refresh(summary)
                
                summaries.append(summary)
                
            except Exception as e:
                logger.error(f"Error generating summary with {model_type}: {e}")
                continue
        
        if not summaries:
            raise HTTPException(status_code=500, detail="Failed to generate any summaries")
        
        return summaries
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/summaries/{summary_id}", response_model=SummaryResponse)
async def get_summary(summary_id: int, db: AsyncSession = Depends(get_db)):
    """Get summary by ID"""
    result = await db.execute(
        select(Summary).where(Summary.id == summary_id)
    )
    summary = result.scalar_one_or_none()
    
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    return summary


@app.get("/api/v1/documents/{document_id}/summaries", response_model=List[SummaryResponse])
async def get_document_summaries(
    document_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get all summaries for a document"""
    result = await db.execute(
        select(Summary)
        .where(Summary.document_id == document_id)
        .order_by(Summary.created_at.desc())
    )
    summaries = result.scalars().all()
    
    return summaries


# ==================== EVALUATION ENDPOINTS ====================

@app.post("/api/v1/evaluate/{summary_id}", response_model=EvaluationResponse)
async def evaluate_summary(
    summary_id: int,
    compute_factuality: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """
    Evaluate a summary using ROUGE, BERTScore, and factuality metrics
    """
    try:
        # Get summary and document
        result = await db.execute(
            select(Summary).where(Summary.id == summary_id)
        )
        summary = result.scalar_one_or_none()
        
        if not summary:
            raise HTTPException(status_code=404, detail="Summary not found")
        
        result = await db.execute(
            select(Document).where(Document.id == summary.document_id)
        )
        document = result.scalar_one_or_none()
        
        # Evaluate
        metrics = evaluation_metrics.evaluate_summary(
            summary=summary.summary_text,
            reference=document.raw_text,
            compute_factuality=compute_factuality,
            factuality_method="overlap"  # Faster method
        )
        
        # Create evaluation record
        evaluation = Evaluation(
            summary_id=summary.id,
            rouge_1_f=metrics.get('rouge_1_f'),
            rouge_1_p=metrics['rouge'].get('rouge1', {}).get('precision'),
            rouge_1_r=metrics['rouge'].get('rouge1', {}).get('recall'),
            rouge_2_f=metrics.get('rouge_2_f'),
            rouge_2_p=metrics['rouge'].get('rouge2', {}).get('precision'),
            rouge_2_r=metrics['rouge'].get('rouge2', {}).get('recall'),
            rouge_l_f=metrics.get('rouge_l_f'),
            rouge_l_p=metrics['rouge'].get('rougeL', {}).get('precision'),
            rouge_l_r=metrics['rouge'].get('rougeL', {}).get('recall'),
            bertscore_f1=metrics.get('bertscore_f1'),
            bertscore_precision=metrics['bertscore'].get('precision'),
            bertscore_recall=metrics['bertscore'].get('recall'),
            factuality_score=metrics.get('factuality_score'),
            factuality_method=metrics.get('factuality', {}).get('method'),
            compression_ratio=metrics.get('compression_ratio'),
            semantic_similarity=metrics.get('semantic_similarity'),
            full_metrics=metrics,
            evaluation_time=metrics.get('evaluation_time')
        )
        
        db.add(evaluation)
        await db.commit()
        await db.refresh(evaluation)
        
        return EvaluationResponse(
            evaluation_id=evaluation.id,
            summary_id=summary.id,
            metrics={
                "rouge_1_f": evaluation.rouge_1_f,
                "rouge_2_f": evaluation.rouge_2_f,
                "rouge_l_f": evaluation.rouge_l_f,
                "bertscore_f1": evaluation.bertscore_f1,
                "factuality_score": evaluation.factuality_score,
                "compression_ratio": evaluation.compression_ratio,
                "semantic_similarity": evaluation.semantic_similarity
            },
            evaluated_at=evaluation.evaluated_at,
            evaluation_time=evaluation.evaluation_time
        )
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== GET EVALUATION BY SUMMARY ====================

@app.get("/api/v1/evaluations/summary/{summary_id}")
async def get_evaluation_by_summary(
    summary_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get evaluation metrics for a summary"""
    result = await db.execute(
        select(Evaluation)
        .where(Evaluation.summary_id == summary_id)
        .order_by(Evaluation.evaluated_at.desc())
    )
    evaluation = result.scalars().first()
    
    if not evaluation:
        raise HTTPException(status_code=404, detail="No evaluation found for this summary")
    
    return {
        "id": evaluation.id,
        "summary_id": evaluation.summary_id,
        "rouge_1_f": evaluation.rouge_1_f,
        "rouge_2_f": evaluation.rouge_2_f,
        "rouge_l_f": evaluation.rouge_l_f,
        "bertscore_f1": evaluation.bertscore_f1,
        "factuality_score": evaluation.factuality_score,
        "compression_ratio": evaluation.compression_ratio,
        "semantic_similarity": evaluation.semantic_similarity,
        "evaluated_at": evaluation.evaluated_at,
        "evaluation_time": evaluation.evaluation_time,
    }


# ==================== DELETE DOCUMENT ====================

@app.delete("/api/v1/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a document and all its summaries/evaluations"""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    await db.delete(document)
    await db.commit()
    
    return {"message": f"Document {document_id} deleted successfully"}


# Include extended API routes
from api_extended import router as extended_router
app.include_router(extended_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
