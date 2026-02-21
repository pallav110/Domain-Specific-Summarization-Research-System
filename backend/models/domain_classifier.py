"""
Domain Classifier Module
Classifies documents as LEGAL or MEDICAL using fine-tuned BERT
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import Tuple, Dict
import re
from loguru import logger


class DomainClassifier:
    """
    Domain Classifier for Legal and Medical Documents
    Uses zero-shot classification or fine-tuned BERT
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """Initialize domain classifier"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing DomainClassifier on {self.device}")
        
        # Using zero-shot classification for initial implementation
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        self.candidate_labels = ["legal document", "medical document", "general document"]
        
        # Domain-specific keywords
        self.legal_keywords = {
            "contract", "agreement", "hereby", "plaintiff", "defendant", "court",
            "jurisdiction", "pursuant", "clause", "amendment", "covenant", "warranty",
            "liability", "indemnify", "attorney", "litigation", "statute", "legal",
            "law", "judgment", "precedent", "petition", "affidavit", "subpoena",
            "testimony", "verdict", "settlement", "damages", "injunction"
        }
        
        self.medical_keywords = {
            "patient", "diagnosis", "treatment", "symptoms", "physician", "clinical",
            "medical", "therapy", "prescription", "disease", "syndrome", "medication",
            "dosage", "prognosis", "pathology", "radiology", "laboratory", "vital signs",
            "examination", "hospital", "surgery", "anesthesia", "postoperative",
            "cardiovascular", "respiratory", "neurological", "chronic", "acute"
        }
    
    def _keyword_analysis(self, text: str) -> Dict[str, float]:
        """Analyze text using keyword matching"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_set = set(words)
        
        legal_count = len(word_set.intersection(self.legal_keywords))
        medical_count = len(word_set.intersection(self.medical_keywords))
        
        total_keywords = legal_count + medical_count
        
        if total_keywords == 0:
            return {"legal": 0.0, "medical": 0.0}
        
        return {
            "legal": legal_count / total_keywords,
            "medical": medical_count / total_keywords
        }
    
    def classify(self, text: str, use_keywords: bool = True) -> Tuple[str, float, Dict]:
        """
        Classify document domain
        
        Args:
            text: Document text (first 512 tokens recommended)
            use_keywords: Whether to combine with keyword analysis
            
        Returns:
            Tuple of (domain, confidence, detailed_scores)
        """
        # Truncate text for classification (use first 1000 words)
        words = text.split()
        sample_text = " ".join(words[:1000])
        
        # Zero-shot classification
        try:
            result = self.classifier(
                sample_text,
                candidate_labels=self.candidate_labels,
                multi_label=False
            )
            
            # Parse results
            scores = {
                label.split()[0]: score 
                for label, score in zip(result['labels'], result['scores'])
            }
            
            # Keyword analysis
            if use_keywords:
                keyword_scores = self._keyword_analysis(sample_text)
                
                # Weighted combination (70% ML, 30% keywords)
                combined_legal = 0.7 * scores.get('legal', 0) + 0.3 * keyword_scores.get('legal', 0)
                combined_medical = 0.7 * scores.get('medical', 0) + 0.3 * keyword_scores.get('medical', 0)
                
                scores['legal'] = combined_legal
                scores['medical'] = combined_medical
            
            # Determine domain - pick the higher-scoring domain unless
            # both scores are very low (below 0.25)
            legal = scores['legal']
            medical = scores['medical']

            if legal > medical and legal > 0.25:
                domain = "legal"
                confidence = legal
            elif medical >= legal and medical > 0.25:
                domain = "medical"
                confidence = medical
            else:
                domain = "unknown"
                confidence = max(legal, medical)
            
            detailed_scores = {
                "legal_score": scores['legal'],
                "medical_score": scores['medical'],
                "general_score": scores.get('general', 0)
            }
            
            logger.info(f"Classified as {domain} with confidence {confidence:.2f}")
            return domain, confidence, detailed_scores
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "unknown", 0.0, {}
    
    def batch_classify(self, texts: list) -> list:
        """Classify multiple documents"""
        results = []
        for text in texts:
            domain, confidence, scores = self.classify(text)
            results.append({
                "domain": domain,
                "confidence": confidence,
                "scores": scores
            })
        return results


# Fine-tunable version (for research improvements)
class FineTunedDomainClassifier:
    """
    Fine-tuned BERT classifier for domain classification
    Can be trained on labeled legal/medical documents
    """
    
    def __init__(self, model_path: str = "bert-base-uncased"):
        """Initialize fine-tuned classifier"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # For fine-tuning, load pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2  # Legal, Medical
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.label_map = {0: "legal", 1: "medical"}
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict domain using fine-tuned model"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
            
            domain = self.label_map[predicted_class]
            
        return domain, confidence
    
    def train(self, train_data, train_labels, epochs: int = 3):
        """
        Fine-tune the classifier
        train_data: List of texts
        train_labels: List of labels (0 for legal, 1 for medical)
        """
        # Implementation for fine-tuning
        # This would include DataLoader, Optimizer, Training Loop
        # Left as placeholder for research improvements
        pass
