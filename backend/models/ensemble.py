"""
Sentence-Level Ensemble Summarization (Novelty 1)

Fuses outputs from multiple summarization models at the sentence level
using semantic clustering and source-relevance scoring.
"""
import re
import time
import numpy as np
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


class SentenceLevelEnsemble:
    """
    Creates an ensemble summary by:
    1. Extracting sentences from all model summaries
    2. Clustering semantically similar sentences across models
    3. Scoring clusters by (model agreement) * (source relevance)
    4. Selecting top clusters and ordering by position
    """

    def __init__(self, semantic_model):
        self.semantic_model = semantic_model

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def ensemble_summarize(
        self,
        summaries: List[Dict],
        source_text: str,
        max_sentences: int = 10,
    ) -> Dict:
        start_time = time.time()

        # Step 1: Extract sentences with provenance
        all_sentences = []
        for summary_dict in summaries:
            sents = self._split_sentences(summary_dict["summary"])
            for i, sent in enumerate(sents):
                all_sentences.append({
                    "text": sent,
                    "model": summary_dict["model_type"],
                    "position": i,
                })

        if len(all_sentences) < 2:
            fallback = summaries[0]["summary"] if summaries else ""
            return {
                "summary": fallback,
                "model_name": "sentence_ensemble",
                "model_type": "ensemble",
                "generation_time": time.time() - start_time,
                "summary_length": len(fallback.split()),
                "metadata": {"error": "Not enough sentences to ensemble"},
            }

        # Step 2: Compute embeddings
        sentence_texts = [s["text"] for s in all_sentences]
        embeddings = self.semantic_model.encode(sentence_texts)
        source_embedding = self.semantic_model.encode(
            [source_text[:5000]]
        )[0]

        # Step 3: Greedy agglomerative clustering (cosine > 0.7)
        sim_matrix = cosine_similarity(embeddings)
        assigned = set()
        clusters: List[List[int]] = []

        for i in range(len(all_sentences)):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, len(all_sentences)):
                if j in assigned:
                    continue
                if sim_matrix[i][j] > 0.7:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)

        # Step 4: Score each cluster
        cluster_scores = []
        for cluster_indices in clusters:
            models_in_cluster = set(
                all_sentences[idx]["model"] for idx in cluster_indices
            )
            agreement_count = len(models_in_cluster)

            cluster_embs = embeddings[cluster_indices]
            source_sims = cosine_similarity(
                cluster_embs, [source_embedding]
            ).flatten()
            avg_source_sim = float(np.mean(source_sims))

            score = agreement_count * avg_source_sim

            best_idx = cluster_indices[int(np.argmax(source_sims))]
            representative = all_sentences[best_idx]

            earliest_position = min(
                all_sentences[idx]["position"] for idx in cluster_indices
            )

            cluster_scores.append({
                "score": score,
                "representative": representative["text"],
                "agreement_count": agreement_count,
                "avg_source_sim": round(avg_source_sim, 4),
                "models": list(models_in_cluster),
                "position": earliest_position,
            })

        # Step 5: Select top-k by score, order by position
        cluster_scores.sort(key=lambda x: x["score"], reverse=True)
        selected = cluster_scores[:max_sentences]
        selected.sort(key=lambda x: x["position"])

        ensemble_text = " ".join(c["representative"] for c in selected)
        generation_time = time.time() - start_time

        logger.info(
            f"Ensemble: {len(all_sentences)} sentences → "
            f"{len(clusters)} clusters → {len(selected)} selected "
            f"in {generation_time:.2f}s"
        )

        return {
            "summary": ensemble_text,
            "model_name": "sentence_ensemble",
            "model_type": "ensemble",
            "generation_time": generation_time,
            "summary_length": len(ensemble_text.split()),
            "metadata": {
                "total_input_sentences": len(all_sentences),
                "clusters_formed": len(clusters),
                "clusters_selected": len(selected),
                "source_models": list(
                    set(s["model"] for s in all_sentences)
                ),
                "cluster_details": [
                    {
                        "score": round(c["score"], 4),
                        "agreement": c["agreement_count"],
                        "source_sim": c["avg_source_sim"],
                        "models": c["models"],
                    }
                    for c in selected
                ],
            },
        }
