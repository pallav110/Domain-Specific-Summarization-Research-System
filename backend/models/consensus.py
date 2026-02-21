"""
Cross-Model Consensus Metric (Novelty 2)

Measures inter-model agreement as a reliability signal.
Unlike ROUGE/BERTScore which compare summary-to-source,
this metric compares summary-to-summary across models.
"""
import re
import numpy as np
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


class ConsensusAnalyzer:
    """
    Computes cross-model consensus metrics:
    - Agreement matrix (NxN pairwise model similarity)
    - Per-sentence consensus (how many models agree on each idea)
    - Consensus score (fraction of high-agreement content)
    - Unique content ratio (per-model uniqueness)
    """

    def __init__(self, semantic_model):
        self.semantic_model = semantic_model

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def compute_consensus(
        self,
        summaries: List[Dict],
    ) -> Dict:
        """
        Compute cross-model consensus from multiple summaries.

        Args:
            summaries: list of {"model_type": str, "summary": str}

        Returns:
            Dict with consensus_score, agreement_matrix,
            unique_content_ratio, high_agreement_sentences
        """
        n_models = len(summaries)
        if n_models < 2:
            return {
                "consensus_score": 0.0,
                "unique_content_ratio": {},
                "agreement_matrix": {},
                "total_sentences_analyzed": 0,
                "high_agreement_count": 0,
                "high_agreement_sentences": [],
            }

        # Parse sentences per model and compute embeddings
        model_sentences: Dict[str, List[str]] = {}
        model_embeddings: Dict[str, np.ndarray] = {}

        for s in summaries:
            model = s["model_type"]
            sents = self._split_sentences(s["summary"])
            model_sentences[model] = sents
            if sents:
                model_embeddings[model] = self.semantic_model.encode(sents)
            else:
                model_embeddings[model] = np.array([])

        model_names = list(model_sentences.keys())

        # 1. Agreement Matrix: pairwise model-level similarity
        agreement_matrix: Dict[str, Dict[str, float]] = {}
        for mA in model_names:
            agreement_matrix[mA] = {}
            for mB in model_names:
                if mA == mB:
                    agreement_matrix[mA][mB] = 1.0
                    continue
                embA = model_embeddings[mA]
                embB = model_embeddings[mB]
                if len(embA) == 0 or len(embB) == 0:
                    agreement_matrix[mA][mB] = 0.0
                    continue
                sims = cosine_similarity(embA, embB)
                avg_max_sim = float(np.mean(np.max(sims, axis=1)))
                agreement_matrix[mA][mB] = round(avg_max_sim, 4)

        # 2. Per-sentence consensus
        per_sentence_consensus = []
        high_agreement_sentences = []

        for model in model_names:
            for idx, sent in enumerate(model_sentences[model]):
                if len(model_embeddings[model]) == 0:
                    continue
                sent_emb = model_embeddings[model][idx : idx + 1]
                agreeing_models = [model]

                for other_model in model_names:
                    if other_model == model:
                        continue
                    if len(model_embeddings[other_model]) == 0:
                        continue
                    sims = cosine_similarity(
                        sent_emb, model_embeddings[other_model]
                    )[0]
                    if np.max(sims) > 0.7:
                        agreeing_models.append(other_model)

                consensus_count = len(agreeing_models)
                entry = {
                    "sentence": sent,
                    "source_model": model,
                    "agreeing_models": agreeing_models,
                    "consensus_count": consensus_count,
                    "consensus_ratio": round(
                        consensus_count / n_models, 4
                    ),
                }
                per_sentence_consensus.append(entry)

                if consensus_count >= 2:
                    high_agreement_sentences.append(entry)

        # 3. Consensus Score
        consensus_score = (
            len(high_agreement_sentences) / len(per_sentence_consensus)
            if per_sentence_consensus
            else 0.0
        )

        # 4. Unique content ratio per model
        unique_content_ratio: Dict[str, float] = {}
        for model in model_names:
            model_sents = [
                p
                for p in per_sentence_consensus
                if p["source_model"] == model
            ]
            unique_sents = [
                p for p in model_sents if p["consensus_count"] == 1
            ]
            unique_content_ratio[model] = round(
                len(unique_sents) / len(model_sents) if model_sents else 0.0,
                4,
            )

        logger.info(
            f"Consensus: {len(per_sentence_consensus)} sentences analyzed, "
            f"score={consensus_score:.2%}, "
            f"{len(high_agreement_sentences)} high-agreement"
        )

        return {
            "consensus_score": round(consensus_score, 4),
            "unique_content_ratio": unique_content_ratio,
            "agreement_matrix": agreement_matrix,
            "total_sentences_analyzed": len(per_sentence_consensus),
            "high_agreement_count": len(high_agreement_sentences),
            "high_agreement_sentences": sorted(
                high_agreement_sentences,
                key=lambda x: x["consensus_count"],
                reverse=True,
            )[:10],
        }
