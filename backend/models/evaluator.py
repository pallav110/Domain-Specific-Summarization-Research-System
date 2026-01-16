"""
Evaluation Metrics Module
Implements ROUGE, BERTScore, and Factuality Checking
"""
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
import time


class EvaluationMetrics:
    """Comprehensive evaluation metrics for summarization"""
    
    def __init__(self):
        """Initialize evaluation models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing EvaluationMetrics on {self.device}")
        
        # ROUGE scorer
        self.rouge = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        # Sentence transformer for semantic similarity
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # For factuality checking
        self.nli_model = None  # Lazy load
    
    def compute_rouge(self, summary: str, reference: str) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE scores
        
        Args:
            summary: Generated summary
            reference: Reference text (original document or gold summary)
            
        Returns:
            Dict with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        try:
            scores = self.rouge.score(reference, summary)
            
            result = {
                'rouge1': {
                    'precision': scores['rouge1'].precision,
                    'recall': scores['rouge1'].recall,
                    'fmeasure': scores['rouge1'].fmeasure
                },
                'rouge2': {
                    'precision': scores['rouge2'].precision,
                    'recall': scores['rouge2'].recall,
                    'fmeasure': scores['rouge2'].fmeasure
                },
                'rougeL': {
                    'precision': scores['rougeL'].precision,
                    'recall': scores['rougeL'].recall,
                    'fmeasure': scores['rougeL'].fmeasure
                }
            }
            
            logger.info(f"ROUGE-1 F1: {result['rouge1']['fmeasure']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"ROUGE computation error: {e}")
            return {}
    
    def compute_bertscore(
        self, 
        summary: str, 
        reference: str,
        model_type: str = "microsoft/deberta-xlarge-mnli"
    ) -> Dict[str, float]:
        """
        Compute BERTScore
        
        Args:
            summary: Generated summary
            reference: Reference text
            model_type: Model for BERTScore
            
        Returns:
            Dict with precision, recall, F1
        """
        try:
            P, R, F1 = bert_score(
                [summary],
                [reference],
                lang="en",
                model_type=model_type,
                device=self.device,
                verbose=False
            )
            
            result = {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
            
            logger.info(f"BERTScore F1: {result['f1']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"BERTScore computation error: {e}")
            # Fallback to lighter model
            try:
                P, R, F1 = bert_score(
                    [summary],
                    [reference],
                    lang="en",
                    device=self.device,
                    verbose=False
                )
                return {
                    'precision': P.mean().item(),
                    'recall': R.mean().item(),
                    'f1': F1.mean().item()
                }
            except:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def compute_semantic_similarity(self, summary: str, reference: str) -> float:
        """
        Compute semantic similarity using sentence transformers
        
        Args:
            summary: Generated summary
            reference: Reference text
            
        Returns:
            Cosine similarity score
        """
        try:
            # Encode texts
            summary_emb = self.semantic_model.encode([summary])
            reference_emb = self.semantic_model.encode([reference])
            
            # Compute cosine similarity
            similarity = cosine_similarity(summary_emb, reference_emb)[0][0]
            
            logger.info(f"Semantic similarity: {similarity:.4f}")
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Semantic similarity error: {e}")
            return 0.0
    
    def compute_compression_ratio(self, summary: str, reference: str) -> float:
        """
        Compute compression ratio
        
        Returns:
            Ratio of summary length to reference length
        """
        summary_words = len(summary.split())
        reference_words = len(reference.split())
        
        if reference_words == 0:
            return 0.0
        
        ratio = summary_words / reference_words
        logger.info(f"Compression ratio: {ratio:.4f}")
        return ratio
    
    def compute_factuality_score(
        self, 
        summary: str, 
        reference: str,
        method: str = "nli"
    ) -> Dict[str, float]:
        """
        Compute factuality/consistency score
        
        Args:
            summary: Generated summary
            reference: Source document
            method: 'nli' or 'qa' based
            
        Returns:
            Factuality score
        """
        try:
            if method == "nli":
                return self._nli_factuality(summary, reference)
            else:
                # Simple overlap-based factuality
                return self._overlap_factuality(summary, reference)
        except Exception as e:
            logger.error(f"Factuality computation error: {e}")
            return {'score': 0.0, 'method': method}
    
    def _nli_factuality(self, summary: str, reference: str) -> Dict[str, float]:
        """NLI-based factuality checking"""
        from transformers import pipeline
        
        # Lazy load NLI model
        if self.nli_model is None:
            self.nli_model = pipeline(
                "text-classification",
                model="microsoft/deberta-base-mnli",
                device=0 if self.device == "cuda" else -1
            )
        
        # Split summary into sentences
        summary_sentences = [s.strip() for s in summary.split('.') if s.strip()]
        
        # Check each sentence against reference
        entailment_scores = []
        for sentence in summary_sentences[:10]:  # Limit to 10 sentences
            try:
                result = self.nli_model(
                    f"{reference} [SEP] {sentence}",
                    truncation=True,
                    max_length=512
                )
                
                # Check if entailed
                if result[0]['label'] == 'ENTAILMENT':
                    entailment_scores.append(result[0]['score'])
                else:
                    entailment_scores.append(0.0)
            except:
                entailment_scores.append(0.0)
        
        avg_score = np.mean(entailment_scores) if entailment_scores else 0.0
        
        return {
            'score': float(avg_score),
            'method': 'nli',
            'num_sentences_checked': len(entailment_scores)
        }
    
    def _overlap_factuality(self, summary: str, reference: str) -> Dict[str, float]:
        """Simple n-gram overlap factuality"""
        summary_words = set(summary.lower().split())
        reference_words = set(reference.lower().split())
        
        if not summary_words:
            return {'score': 0.0, 'method': 'overlap'}
        
        overlap = len(summary_words.intersection(reference_words))
        score = overlap / len(summary_words)
        
        return {
            'score': float(score),
            'method': 'overlap'
        }
    
    def evaluate_summary(
        self, 
        summary: str, 
        reference: str,
        compute_factuality: bool = True,
        factuality_method: str = "overlap"  # Use 'nli' for better quality
    ) -> Dict:
        """
        Comprehensive evaluation of summary
        
        Args:
            summary: Generated summary
            reference: Source document
            compute_factuality: Whether to compute factuality
            factuality_method: Method for factuality ('nli' or 'overlap')
            
        Returns:
            Complete evaluation metrics
        """
        start_time = time.time()
        logger.info("Starting comprehensive evaluation...")
        
        metrics = {}
        
        # ROUGE scores
        rouge_scores = self.compute_rouge(summary, reference)
        metrics['rouge'] = rouge_scores
        metrics['rouge_1_f'] = rouge_scores.get('rouge1', {}).get('fmeasure', 0.0)
        metrics['rouge_2_f'] = rouge_scores.get('rouge2', {}).get('fmeasure', 0.0)
        metrics['rouge_l_f'] = rouge_scores.get('rougeL', {}).get('fmeasure', 0.0)
        
        # BERTScore (use lighter model for speed)
        bertscore = self.compute_bertscore(summary, reference, model_type="distilbert-base-uncased")
        metrics['bertscore'] = bertscore
        metrics['bertscore_f1'] = bertscore.get('f1', 0.0)
        
        # Semantic similarity
        semantic_sim = self.compute_semantic_similarity(summary, reference)
        metrics['semantic_similarity'] = semantic_sim
        
        # Compression ratio
        compression = self.compute_compression_ratio(summary, reference)
        metrics['compression_ratio'] = compression
        
        # Factuality (optional, can be slow)
        if compute_factuality:
            factuality = self.compute_factuality_score(summary, reference, method=factuality_method)
            metrics['factuality'] = factuality
            metrics['factuality_score'] = factuality.get('score', 0.0)
        
        evaluation_time = time.time() - start_time
        metrics['evaluation_time'] = evaluation_time
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        return metrics
    
    def batch_evaluate(
        self, 
        summaries: List[Dict],
        reference: str
    ) -> List[Dict]:
        """
        Evaluate multiple summaries
        
        Args:
            summaries: List of summary dicts with 'text' and 'model' keys
            reference: Source document
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for summary_dict in summaries:
            summary_text = summary_dict.get('summary', summary_dict.get('text', ''))
            model_name = summary_dict.get('model_name', 'unknown')
            
            logger.info(f"Evaluating summary from {model_name}")
            
            metrics = self.evaluate_summary(
                summary_text,
                reference,
                compute_factuality=True,
                factuality_method="overlap"  # Faster method
            )
            
            results.append({
                'model_name': model_name,
                'model_type': summary_dict.get('model_type', 'unknown'),
                'metrics': metrics
            })
        
        return results


class ComparativeAnalysis:
    """Compare multiple summarization models"""
    
    def __init__(self):
        self.evaluator = EvaluationMetrics()
    
    def compare_models(
        self, 
        summaries: List[Dict],
        reference: str
    ) -> Dict:
        """
        Compare multiple models on same document
        
        Args:
            summaries: List of summaries from different models
            reference: Source document
            
        Returns:
            Comparison results with rankings
        """
        # Evaluate all summaries
        evaluations = self.evaluator.batch_evaluate(summaries, reference)
        
        # Aggregate metrics
        comparison = {
            'models': [],
            'rankings': {},
            'best_model': {}
        }
        
        for eval_result in evaluations:
            model_info = {
                'model_name': eval_result['model_name'],
                'model_type': eval_result['model_type'],
                'rouge_1': eval_result['metrics'].get('rouge_1_f', 0),
                'rouge_2': eval_result['metrics'].get('rouge_2_f', 0),
                'rouge_l': eval_result['metrics'].get('rouge_l_f', 0),
                'bertscore': eval_result['metrics'].get('bertscore_f1', 0),
                'semantic_similarity': eval_result['metrics'].get('semantic_similarity', 0),
                'factuality': eval_result['metrics'].get('factuality_score', 0),
                'compression_ratio': eval_result['metrics'].get('compression_ratio', 0)
            }
            comparison['models'].append(model_info)
        
        # Rank models
        if comparison['models']:
            # Rank by ROUGE-L
            sorted_by_rouge = sorted(
                comparison['models'],
                key=lambda x: x['rouge_l'],
                reverse=True
            )
            comparison['rankings']['rouge_l'] = [m['model_name'] for m in sorted_by_rouge]
            
            # Rank by BERTScore
            sorted_by_bert = sorted(
                comparison['models'],
                key=lambda x: x['bertscore'],
                reverse=True
            )
            comparison['rankings']['bertscore'] = [m['model_name'] for m in sorted_by_bert]
            
            # Overall best (weighted average)
            for model in comparison['models']:
                model['overall_score'] = (
                    0.3 * model['rouge_l'] +
                    0.3 * model['bertscore'] +
                    0.2 * model['semantic_similarity'] +
                    0.2 * model['factuality']
                )
            
            sorted_overall = sorted(
                comparison['models'],
                key=lambda x: x['overall_score'],
                reverse=True
            )
            
            comparison['best_model'] = {
                'name': sorted_overall[0]['model_name'],
                'overall_score': sorted_overall[0]['overall_score']
            }
            comparison['rankings']['overall'] = [m['model_name'] for m in sorted_overall]
        
        return comparison
