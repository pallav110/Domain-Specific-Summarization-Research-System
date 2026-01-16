"""
Experiment Runner Script
Run comprehensive experiments on documents
"""
import asyncio
import sys
import os
import argparse
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import async_session_maker
from sqlalchemy import select
from models.db_models import Document, Summary, Evaluation, Experiment, ModelType, DomainType
from models.summarizers import SummarizationEngine
from models.evaluator import EvaluationMetrics
from datetime import datetime
import json

async def run_experiment(
    document_id: int,
    models: List[str],
    experiment_name: str = None
):
    """
    Run a complete experiment on a document
    
    Args:
        document_id: ID of document to process
        models: List of model types to test
        experiment_name: Optional name for experiment
    """
    
    summarization_engine = SummarizationEngine()
    evaluator = EvaluationMetrics()
    
    async with async_session_maker() as session:
        # Get document
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if not document:
            print(f"Error: Document {document_id} not found")
            return
        
        print(f"\n{'='*60}")
        print(f"Starting Experiment: {experiment_name or 'Unnamed'}")
        print(f"Document: {document.original_filename}")
        print(f"Domain: {document.detected_domain.value}")
        print(f"Words: {document.word_count}")
        print(f"{'='*60}\n")
        
        # Create experiment record
        experiment = Experiment(
            document_id=document.id,
            experiment_name=experiment_name or f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Testing models: {', '.join(models)}",
            domain=document.detected_domain,
            models_tested=models,
            status="running"
        )
        
        session.add(experiment)
        await session.flush()
        
        all_results = []
        
        # Test each model
        for model_type in models:
            print(f"\n--- Testing Model: {model_type.upper()} ---")
            
            try:
                # Generate summary
                print("Generating summary...")
                result_dict = summarization_engine.summarize(
                    text=document.raw_text,
                    model_type=model_type,
                    domain=document.detected_domain.value,
                    max_length=512,
                    min_length=100
                )
                
                print(f"Generation time: {result_dict['generation_time']:.2f}s")
                print(f"Summary length: {result_dict['summary_length']} words")
                
                # Save summary
                summary = Summary(
                    document_id=document.id,
                    model_type=ModelType(result_dict['model_type']),
                    model_name=result_dict['model_name'],
                    summary_text=result_dict['summary'],
                    summary_length=result_dict['summary_length'],
                    generation_time=result_dict['generation_time']
                )
                
                session.add(summary)
                await session.flush()
                
                # Evaluate
                print("Evaluating summary...")
                metrics = evaluator.evaluate_summary(
                    summary=summary.summary_text,
                    reference=document.raw_text,
                    compute_factuality=True,
                    factuality_method="overlap"
                )
                
                print(f"ROUGE-1: {metrics.get('rouge_1_f', 0):.4f}")
                print(f"ROUGE-2: {metrics.get('rouge_2_f', 0):.4f}")
                print(f"ROUGE-L: {metrics.get('rouge_l_f', 0):.4f}")
                print(f"BERTScore: {metrics.get('bertscore_f1', 0):.4f}")
                print(f"Factuality: {metrics.get('factuality_score', 0):.4f}")
                
                # Save evaluation
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
                
                session.add(evaluation)
                
                all_results.append({
                    'model': model_type,
                    'rouge_l': metrics.get('rouge_l_f', 0),
                    'bertscore': metrics.get('bertscore_f1', 0),
                    'factuality': metrics.get('factuality_score', 0),
                    'generation_time': result_dict['generation_time']
                })
                
            except Exception as e:
                print(f"Error testing {model_type}: {e}")
                continue
        
        # Update experiment
        if all_results:
            best_model = max(all_results, key=lambda x: x['rouge_l'])
            experiment.results_summary = {
                'models': all_results,
                'total_models_tested': len(all_results)
            }
            experiment.best_model = best_model['model']
        
        experiment.status = "completed"
        experiment.completed_at = datetime.utcnow()
        
        await session.commit()
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        print(f"\nResults Summary:")
        print(f"Models Tested: {len(all_results)}")
        if all_results:
            print(f"Best Model (by ROUGE-L): {best_model['model']}")
            print(f"\nDetailed Results:")
            for result in sorted(all_results, key=lambda x: x['rouge_l'], reverse=True):
                print(f"\n{result['model'].upper()}:")
                print(f"  ROUGE-L: {result['rouge_l']:.4f}")
                print(f"  BERTScore: {result['bertscore']:.4f}")
                print(f"  Factuality: {result['factuality']:.4f}")
                print(f"  Time: {result['generation_time']:.2f}s")
        
        print(f"\nExperiment ID: {experiment.id}")
        print(f"View results at: http://localhost:3000/experiments/{experiment.id}")


async def run_batch_experiments(domain: str = None):
    """Run experiments on all documents"""
    
    async with async_session_maker() as session:
        query = select(Document)
        if domain:
            query = query.where(Document.detected_domain == DomainType(domain))
        
        result = await session.execute(query)
        documents = result.scalars().all()
        
        if not documents:
            print("No documents found")
            return
        
        print(f"\nFound {len(documents)} documents to process\n")
        
        for doc in documents:
            # Determine models to test based on domain
            models = ['bart', 'pegasus']
            
            if doc.detected_domain == DomainType.LEGAL:
                models.append('legal_bert_pegasus')
            elif doc.detected_domain == DomainType.MEDICAL:
                models.append('clinical_bert_pegasus')
            
            # Add GPT if API key is configured
            from config import settings
            if settings.OPENAI_API_KEY:
                models.append('gpt')
            
            await run_experiment(
                document_id=doc.id,
                models=models,
                experiment_name=f"Batch_{doc.original_filename}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run summarization experiments')
    parser.add_argument('--document_id', type=int, help='Single document ID to process')
    parser.add_argument('--domain', choices=['legal', 'medical'], help='Filter by domain for batch')
    parser.add_argument('--models', nargs='+', default=['bart', 'pegasus', 'gpt'], help='Models to test')
    parser.add_argument('--name', type=str, help='Experiment name')
    
    args = parser.parse_args()
    
    if args.document_id:
        asyncio.run(run_experiment(
            document_id=args.document_id,
            models=args.models,
            experiment_name=args.name
        ))
    else:
        asyncio.run(run_batch_experiments(domain=args.domain))
