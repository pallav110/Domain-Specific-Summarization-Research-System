"""
Adaptive Model Recommender (Novelty 3)

Predicts the best summarization model for a document based on
document features and historical evaluation performance.
Uses RandomForest when sklearn is available, falls back to k-NN.
"""
import re
import numpy as np
from typing import Dict, List, Optional
from collections import Counter
from loguru import logger


class ModelRecommender:
    """
    Recommends the best summarization model for a document by:
    1. Extracting document features (word count, sentence stats, domain, etc.)
    2. Training on historical data (which model scored best per document)
    3. Predicting best model for new documents
    """

    def __init__(self):
        self.is_trained = False
        self.training_features: List[np.ndarray] = []
        self.training_labels: List[str] = []
        self.model = None
        self.feature_names = [
            "word_count",
            "sentence_count",
            "avg_sentence_length",
            "domain_confidence",
            "type_token_ratio",
            "is_legal",
            "is_medical",
        ]
        self.sklearn_available = False
        self._try_load_sklearn()

    def _try_load_sklearn(self):
        try:
            from sklearn.ensemble import RandomForestClassifier  # noqa: F401
            self.sklearn_available = True
            logger.info("sklearn available — using RandomForest recommender")
        except ImportError:
            logger.info("sklearn not available — using numpy k-NN recommender")

    def extract_features(self, doc: Dict) -> np.ndarray:
        """Extract a 7-dimensional feature vector from document data."""
        text = doc.get("raw_text", "")
        words = text.split() if text else []
        word_count = doc.get("word_count", len(words))

        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = len(sentences)
        avg_sentence_length = (
            word_count / sentence_count if sentence_count > 0 else 0
        )

        unique_words = set(w.lower() for w in words)
        type_token_ratio = len(unique_words) / len(words) if words else 0

        domain_confidence = doc.get("domain_confidence", 0.0) or 0.0
        domain = doc.get("detected_domain", "unknown")
        if hasattr(domain, "value"):
            domain = domain.value
        is_legal = 1.0 if domain == "legal" else 0.0
        is_medical = 1.0 if domain == "medical" else 0.0

        return np.array([
            word_count,
            sentence_count,
            avg_sentence_length,
            domain_confidence,
            type_token_ratio,
            is_legal,
            is_medical,
        ])

    def train(self, training_data: List[Dict]) -> Dict:
        """
        Train from historical evaluation data.

        Args:
            training_data: list of {
                "features": {raw_text, word_count, detected_domain, domain_confidence},
                "best_model": str
            }

        Returns:
            Dict with training stats
        """
        if len(training_data) < 3:
            logger.warning(
                f"Not enough training data ({len(training_data)} samples). "
                "Need at least 3."
            )
            return {
                "samples_used": len(training_data),
                "models_in_training": [],
                "message": "Not enough data to train (need 3+ documents with evaluations)",
            }

        self.training_features = []
        self.training_labels = []

        for item in training_data:
            features = self.extract_features(item["features"])
            self.training_features.append(features)
            self.training_labels.append(item["best_model"])

        X = np.array(self.training_features)
        y = self.training_labels
        unique_models = list(set(y))

        if self.sklearn_available:
            from sklearn.ensemble import RandomForestClassifier

            self.model = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42
            )
            self.model.fit(X, y)
            logger.info(
                f"Trained RandomForest on {len(X)} samples, "
                f"{len(unique_models)} classes"
            )
        else:
            self.model = None
            logger.info(
                f"Stored {len(X)} samples for k-NN lookup"
            )

        self.is_trained = True

        return {
            "samples_used": len(X),
            "models_in_training": unique_models,
            "message": f"Trained on {len(X)} samples with {len(unique_models)} model classes",
        }

    def recommend(self, doc: Dict) -> Dict:
        """
        Recommend best model for a document.

        Returns:
            Dict with recommended_model, confidence, reasoning, all_scores
        """
        if not self.is_trained:
            return self._rule_based_recommendation(doc)

        features = self.extract_features(doc)

        if self.sklearn_available and self.model is not None:
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            classes = self.model.classes_
            confidence = float(max(probabilities))
            all_scores = {
                cls: round(float(prob), 4)
                for cls, prob in zip(classes, probabilities)
            }
            return {
                "recommended_model": prediction,
                "confidence": round(confidence, 4),
                "reasoning": (
                    f"RandomForest prediction based on "
                    f"{len(self.training_features)} historical evaluations"
                ),
                "all_scores": all_scores,
            }
        else:
            # Numpy k-NN fallback
            X = np.array(self.training_features)
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-8
            X_norm = (X - mean) / std
            f_norm = (features - mean) / std

            distances = np.linalg.norm(X_norm - f_norm, axis=1)
            k = min(3, len(distances))
            nearest_indices = np.argsort(distances)[:k]

            votes = Counter(
                self.training_labels[i] for i in nearest_indices
            )
            best_model = votes.most_common(1)[0][0]
            confidence = votes.most_common(1)[0][1] / k

            return {
                "recommended_model": best_model,
                "confidence": round(confidence, 4),
                "reasoning": (
                    f"k-NN (k={k}) from "
                    f"{len(self.training_features)} historical evaluations"
                ),
                "all_scores": {
                    m: round(c / k, 4) for m, c in votes.items()
                },
            }

    def _rule_based_recommendation(self, doc: Dict) -> Dict:
        """Fallback when no training data exists."""
        domain = doc.get("detected_domain", "unknown")
        if hasattr(domain, "value"):
            domain = domain.value
        word_count = doc.get("word_count", 0)
        confidence = doc.get("domain_confidence", 0.0) or 0.0

        if domain == "legal" and confidence > 0.4:
            model = "legal_bert_pegasus"
            reason = "Legal document with domain confidence — domain-specific model recommended"
        elif domain == "medical" and confidence > 0.4:
            model = "clinical_bert_pegasus"
            reason = "Medical document with domain confidence — domain-specific model recommended"
        elif word_count > 3000:
            model = "gemini"
            reason = "Long document — LLM handles longer context better"
        else:
            model = "bart"
            reason = "General document — BART provides reliable baseline"

        return {
            "recommended_model": model,
            "confidence": 0.5,
            "reasoning": f"Rule-based (no training data yet): {reason}",
            "all_scores": {},
        }
