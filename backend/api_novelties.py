"""
API Endpoints for Research Novelties
- Novelty 1: Sentence-Level Ensemble Summarization
- Novelty 2: Cross-Model Consensus Metric
- Novelty 3: Adaptive Model Recommender
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from loguru import logger

from database import get_db
from models.db_models import Document, Summary, Evaluation, ModelType
from schemas import (
    EnsembleSummaryResponse,
    ConsensusResponse,
    RecommendationResponse,
    TrainRecommenderResponse,
)

router = APIRouter(prefix="/api/v1", tags=["novelties"])


# ==================== NOVELTY 1: ENSEMBLE ====================


@router.post("/ensemble/{document_id}", response_model=EnsembleSummaryResponse)
async def generate_ensemble_summary(
    document_id: int,
    max_sentences: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a sentence-level ensemble summary from all existing
    model summaries for a document. Requires 2+ summaries.
    """
    from main import sentence_ensemble

    if sentence_ensemble is None:
        raise HTTPException(status_code=503, detail="Ensemble engine not initialized")

    # Fetch document
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Fetch existing summaries (exclude any prior ensemble)
    result = await db.execute(
        select(Summary)
        .where(
            and_(
                Summary.document_id == document_id,
                Summary.model_type != ModelType.ENSEMBLE,
            )
        )
    )
    summaries = result.scalars().all()

    if len(summaries) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 summaries, found {len(summaries)}. Generate more summaries first.",
        )

    summary_dicts = [
        {
            "model_type": s.model_type.value,
            "summary": s.summary_text,
        }
        for s in summaries
    ]

    try:
        result = sentence_ensemble.ensemble_summarize(
            summary_dicts, document.raw_text, max_sentences
        )
    except Exception as e:
        logger.error(f"Ensemble generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Delete any previous ensemble for this document
    old = await db.execute(
        select(Summary).where(
            and_(
                Summary.document_id == document_id,
                Summary.model_type == ModelType.ENSEMBLE,
            )
        )
    )
    for old_summary in old.scalars().all():
        await db.delete(old_summary)

    # Save new ensemble summary
    new_summary = Summary(
        document_id=document_id,
        model_type=ModelType.ENSEMBLE,
        model_name=result["model_name"],
        summary_text=result["summary"],
        summary_length=result["summary_length"],
        generation_time=result["generation_time"],
        generation_params=result.get("metadata", {}),
    )
    db.add(new_summary)
    await db.commit()
    await db.refresh(new_summary)

    metadata = result.get("metadata", {})

    return EnsembleSummaryResponse(
        summary_id=new_summary.id,
        summary_text=result["summary"],
        summary_length=result["summary_length"],
        generation_time=result["generation_time"],
        source_models=metadata.get("source_models", []),
        clusters_formed=metadata.get("clusters_formed", 0),
        clusters_selected=metadata.get("clusters_selected", 0),
    )


# ==================== NOVELTY 2: CONSENSUS ====================


@router.get("/consensus/{document_id}", response_model=ConsensusResponse)
async def get_consensus_metrics(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Compute cross-model consensus for all summaries of a document.
    Returns agreement matrix, consensus score, and high-agreement sentences.
    """
    from main import consensus_analyzer

    if consensus_analyzer is None:
        raise HTTPException(status_code=503, detail="Consensus analyzer not initialized")

    # Fetch document
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Fetch summaries (exclude ensemble)
    result = await db.execute(
        select(Summary)
        .where(
            and_(
                Summary.document_id == document_id,
                Summary.model_type != ModelType.ENSEMBLE,
            )
        )
    )
    summaries = result.scalars().all()

    if len(summaries) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 summaries for consensus, found {len(summaries)}.",
        )

    summary_dicts = [
        {
            "model_type": s.model_type.value,
            "summary": s.summary_text,
        }
        for s in summaries
    ]

    try:
        consensus = consensus_analyzer.compute_consensus(summary_dicts)
    except Exception as e:
        logger.error(f"Consensus computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return ConsensusResponse(
        document_id=document_id,
        **consensus,
    )


# ==================== NOVELTY 3: RECOMMENDER ====================


@router.get("/recommend/{document_id}", response_model=RecommendationResponse)
async def recommend_model(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Recommend the best summarization model for a document
    based on document features and historical performance.
    """
    from main import model_recommender

    if model_recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")

    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_dict = {
        "raw_text": document.raw_text,
        "word_count": document.word_count,
        "detected_domain": document.detected_domain.value if document.detected_domain else "unknown",
        "domain_confidence": document.domain_confidence,
    }

    try:
        recommendation = model_recommender.recommend(doc_dict)
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return RecommendationResponse(
        document_id=document_id,
        **recommendation,
    )


@router.post("/recommender/train", response_model=TrainRecommenderResponse)
async def train_recommender(db: AsyncSession = Depends(get_db)):
    """
    Train the model recommender from all historical evaluation data.
    For each document, finds which model achieved the highest weighted score.
    """
    from main import model_recommender

    if model_recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")

    # Query all evaluations with their summaries and documents
    result = await db.execute(
        select(Evaluation, Summary, Document)
        .join(Summary, Evaluation.summary_id == Summary.id)
        .join(Document, Summary.document_id == Document.id)
        .where(Summary.model_type != ModelType.ENSEMBLE)
    )
    rows = result.all()

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No evaluation data found. Run evaluations first.",
        )

    # Group by document, find best model per document
    doc_model_scores: dict = {}  # doc_id -> {model_type -> weighted_score}
    doc_info: dict = {}  # doc_id -> document data

    for evaluation, summary, document in rows:
        doc_id = document.id
        if doc_id not in doc_info:
            doc_info[doc_id] = {
                "raw_text": document.raw_text,
                "word_count": document.word_count,
                "detected_domain": document.detected_domain.value if document.detected_domain else "unknown",
                "domain_confidence": document.domain_confidence,
            }
        if doc_id not in doc_model_scores:
            doc_model_scores[doc_id] = {}

        model_type = summary.model_type.value
        weighted = (
            0.3 * (evaluation.rouge_l_f or 0)
            + 0.3 * (evaluation.bertscore_f1 or 0)
            + 0.2 * (evaluation.semantic_similarity or 0)
            + 0.2 * (evaluation.factuality_score or 0)
        )

        # Keep the highest score if multiple evaluations for same model
        if model_type not in doc_model_scores[doc_id]:
            doc_model_scores[doc_id][model_type] = weighted
        else:
            doc_model_scores[doc_id][model_type] = max(
                doc_model_scores[doc_id][model_type], weighted
            )

    # Build training data
    training_data = []
    for doc_id, scores in doc_model_scores.items():
        if not scores:
            continue
        best_model = max(scores, key=scores.get)
        training_data.append({
            "features": doc_info[doc_id],
            "best_model": best_model,
        })

    try:
        stats = model_recommender.train(training_data)
    except Exception as e:
        logger.error(f"Recommender training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return TrainRecommenderResponse(**stats)
