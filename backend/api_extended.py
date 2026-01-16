"""
FastAPI Application - Additional Endpoints
Experiment, Comparison, and Dashboard endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger

from database import get_db
from models.db_models import Document, Summary, Experiment, Evaluation, DomainType, ModelType
from schemas import (
    ExperimentRequest,
    ExperimentResponse,
    ComparisonResponse,
    DashboardStats,
    ModelComparison,
    EvaluationMetrics as EvaluationMetricsSchema
)

router = APIRouter(prefix="/api/v1", tags=["extended"])


# ==================== EXPERIMENT ENDPOINTS ====================

@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a research experiment
    Tests multiple models on a document and compares results
    """
    from main import summarization_engine, evaluation_metrics
    
    try:
        # Get document
        result = await db.execute(
            select(Document).where(Document.id == request.document_id)
        )
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Create experiment record
        experiment = Experiment(
            document_id=document.id,
            experiment_name=request.experiment_name,
            description=request.description,
            domain=document.detected_domain,
            models_tested=[m.value for m in request.models_to_test],
            status="running"
        )
        
        db.add(experiment)
        await db.commit()
        await db.refresh(experiment)
        
        # Generate summaries for each model
        all_results = []
        
        for model_type in request.models_to_test:
            try:
                # Generate summary
                result_dict = summarization_engine.summarize(
                    text=document.raw_text,
                    model_type=model_type.value,
                    domain=document.detected_domain.value,
                    max_length=512,
                    min_length=100
                )
                
                # Save summary
                summary = Summary(
                    document_id=document.id,
                    model_type=model_type,
                    model_name=result_dict['model_name'],
                    summary_text=result_dict['summary'],
                    summary_length=result_dict['summary_length'],
                    generation_time=result_dict['generation_time']
                )
                
                db.add(summary)
                await db.flush()
                
                # Evaluate if requested
                if request.evaluate:
                    metrics = evaluation_metrics.evaluate_summary(
                        summary=summary.summary_text,
                        reference=document.raw_text,
                        compute_factuality=True,
                        factuality_method="overlap"
                    )
                    
                    evaluation = Evaluation(
                        summary_id=summary.id,
                        rouge_1_f=metrics.get('rouge_1_f'),
                        rouge_2_f=metrics.get('rouge_2_f'),
                        rouge_l_f=metrics.get('rouge_l_f'),
                        bertscore_f1=metrics.get('bertscore_f1'),
                        factuality_score=metrics.get('factuality_score'),
                        compression_ratio=metrics.get('compression_ratio'),
                        semantic_similarity=metrics.get('semantic_similarity'),
                        full_metrics=metrics
                    )
                    
                    db.add(evaluation)
                    
                    all_results.append({
                        'model': model_type.value,
                        'rouge_l': metrics.get('rouge_l_f', 0),
                        'bertscore': metrics.get('bertscore_f1', 0),
                        'generation_time': result_dict['generation_time']
                    })
                
            except Exception as e:
                logger.error(f"Error with model {model_type}: {e}")
                continue
        
        # Update experiment with results
        if all_results:
            best_model = max(all_results, key=lambda x: x['rouge_l'])
            
            experiment.results_summary = {
                'models': all_results,
                'total_models_tested': len(all_results)
            }
            experiment.best_model = best_model['model']
        
        experiment.status = "completed"
        experiment.completed_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(experiment)
        
        return experiment
        
    except Exception as e:
        logger.error(f"Experiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: int, db: AsyncSession = Depends(get_db)):
    """Get experiment details"""
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return experiment


@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """List all experiments"""
    result = await db.execute(
        select(Experiment)
        .offset(skip)
        .limit(limit)
        .order_by(Experiment.started_at.desc())
    )
    experiments = result.scalars().all()
    
    return experiments


# ==================== COMPARISON ENDPOINTS ====================

@router.get("/compare/{document_id}", response_model=ComparisonResponse)
async def compare_models(document_id: int, db: AsyncSession = Depends(get_db)):
    """
    Compare all models tested on a document
    """
    from main import comparative_analysis
    
    try:
        # Get document
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get all summaries with evaluations
        result = await db.execute(
            select(Summary, Evaluation)
            .outerjoin(Evaluation, Summary.id == Evaluation.summary_id)
            .where(Summary.document_id == document_id)
        )
        
        summaries_data = result.all()
        
        if not summaries_data:
            raise HTTPException(
                status_code=404,
                detail="No summaries found for this document"
            )
        
        # Build comparison
        models = []
        for summary, evaluation in summaries_data:
            if evaluation:
                model_comp = ModelComparison(
                    model_name=summary.model_name,
                    model_type=summary.model_type,
                    summary_length=summary.summary_length,
                    generation_time=summary.generation_time or 0.0,
                    metrics=EvaluationMetricsSchema(
                        rouge_1_f=evaluation.rouge_1_f,
                        rouge_2_f=evaluation.rouge_2_f,
                        rouge_l_f=evaluation.rouge_l_f,
                        bertscore_f1=evaluation.bertscore_f1,
                        factuality_score=evaluation.factuality_score,
                        compression_ratio=evaluation.compression_ratio,
                        semantic_similarity=evaluation.semantic_similarity
                    )
                )
                models.append(model_comp)
        
        # Determine best model
        if models:
            best_model = max(
                models,
                key=lambda m: (m.metrics.rouge_l_f or 0) + (m.metrics.bertscore_f1 or 0)
            )
            best_overall = best_model.model_name
        else:
            best_overall = "N/A"
        
        # Generate recommendations
        recommendations = []
        if document.detected_domain == DomainType.LEGAL:
            recommendations.append("Legal documents benefit from Legal-BERT models")
        elif document.detected_domain == DomainType.MEDICAL:
            recommendations.append("Medical documents benefit from Clinical-BERT models")
        
        if models:
            avg_rouge = sum(m.metrics.rouge_l_f or 0 for m in models) / len(models)
            if avg_rouge > 0.5:
                recommendations.append("High ROUGE scores indicate good extractive quality")
        
        return ComparisonResponse(
            document_id=document.id,
            domain=document.detected_domain,
            models=models,
            best_overall=best_overall,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== DASHBOARD ENDPOINTS ====================

@router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    """
    Get aggregate statistics for research dashboard
    """
    try:
        # Total counts
        total_docs = await db.scalar(select(func.count(Document.id)))
        total_summaries = await db.scalar(select(func.count(Summary.id)))
        total_experiments = await db.scalar(select(func.count(Experiment.id)))
        
        # Domain distribution
        domain_result = await db.execute(
            select(Document.detected_domain, func.count(Document.id))
            .group_by(Document.detected_domain)
        )
        domains_distribution = {
            str(domain): count for domain, count in domain_result.all()
        }
        
        # Model usage
        model_result = await db.execute(
            select(Summary.model_type, func.count(Summary.id))
            .group_by(Summary.model_type)
        )
        model_usage = {
            str(model): count for model, count in model_result.all()
        }
        
        # Average metrics
        avg_metrics = {}
        
        avg_rouge_1 = await db.scalar(
            select(func.avg(Evaluation.rouge_1_f))
        )
        avg_rouge_2 = await db.scalar(
            select(func.avg(Evaluation.rouge_2_f))
        )
        avg_rouge_l = await db.scalar(
            select(func.avg(Evaluation.rouge_l_f))
        )
        avg_bertscore = await db.scalar(
            select(func.avg(Evaluation.bertscore_f1))
        )
        avg_factuality = await db.scalar(
            select(func.avg(Evaluation.factuality_score))
        )
        
        avg_metrics = {
            'rouge_1': float(avg_rouge_1) if avg_rouge_1 else 0.0,
            'rouge_2': float(avg_rouge_2) if avg_rouge_2 else 0.0,
            'rouge_l': float(avg_rouge_l) if avg_rouge_l else 0.0,
            'bertscore': float(avg_bertscore) if avg_bertscore else 0.0,
            'factuality': float(avg_factuality) if avg_factuality else 0.0
        }
        
        return DashboardStats(
            total_documents=total_docs or 0,
            total_summaries=total_summaries or 0,
            total_experiments=total_experiments or 0,
            domains_distribution=domains_distribution,
            model_usage=model_usage,
            average_metrics=avg_metrics
        )
        
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/research/results")
async def get_research_results(
    domain: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed research results for analysis
    Returns data suitable for graphs and tables
    """
    try:
        # Build query
        query = (
            select(Experiment, Summary, Evaluation, Document)
            .join(Document, Experiment.document_id == Document.id)
            .join(Summary, and_(
                Summary.document_id == Document.id,
                Experiment.id == Experiment.id
            ))
            .outerjoin(Evaluation, Summary.id == Evaluation.summary_id)
        )
        
        if domain:
            query = query.where(Document.detected_domain == DomainType(domain))
        
        result = await db.execute(query)
        data = result.all()
        
        # Format results
        results = []
        for exp, summary, evaluation, doc in data:
            if evaluation:
                results.append({
                    'experiment_id': exp.id,
                    'experiment_name': exp.experiment_name,
                    'domain': str(doc.detected_domain.value),
                    'model_type': str(summary.model_type.value),
                    'model_name': summary.model_name,
                    'rouge_1': evaluation.rouge_1_f or 0.0,
                    'rouge_2': evaluation.rouge_2_f or 0.0,
                    'rouge_l': evaluation.rouge_l_f or 0.0,
                    'bertscore': evaluation.bertscore_f1 or 0.0,
                    'factuality': evaluation.factuality_score or 0.0,
                    'generation_time': summary.generation_time or 0.0,
                    'compression_ratio': evaluation.compression_ratio or 0.0,
                    'timestamp': exp.started_at.isoformat()
                })
        
        return {
            'total_results': len(results),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Research results error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
