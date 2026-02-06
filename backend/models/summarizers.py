"""
Summarization Engines Module
Implements multiple summarization approaches
"""
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
    pipeline
)
from typing import List, Dict, Optional
from loguru import logger
import time
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from config import settings


class BaseSummarizer:
    """Base class for summarizers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing {self.__class__.__name__} on {self.device}")
    
    def summarize(self, text: str, max_length: int = 512, min_length: int = 100) -> Dict:
        """Generate summary - to be implemented by subclasses"""
        raise NotImplementedError


class BARTSummarizer(BaseSummarizer):
    """BART-based summarizer (Generic)"""
    
    def __init__(self):
        super().__init__(settings.BART_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def summarize(self, text: str, max_length: int = 512, min_length: int = 100) -> Dict:
        """Generate summary using BART"""
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            generation_time = time.time() - start_time
            
            return {
                "summary": summary,
                "model_name": self.model_name,
                "model_type": "bart",
                "generation_time": generation_time,
                "summary_length": len(summary.split())
            }
            
        except Exception as e:
            logger.error(f"BART summarization error: {e}")
            raise


class PegasusSummarizer(BaseSummarizer):
    """PEGASUS-based summarizer"""
    
    def __init__(self):
        super().__init__(settings.PEGASUS_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def summarize(self, text: str, max_length: int = 512, min_length: int = 100) -> Dict:
        """Generate summary using PEGASUS"""
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            # Clean up PEGASUS <n> newline tokens that leak into output
            summary = summary.replace(".<n>", ". ").replace("<n>", " ").strip()
            generation_time = time.time() - start_time
            
            return {
                "summary": summary,
                "model_name": self.model_name,
                "model_type": "pegasus",
                "generation_time": generation_time,
                "summary_length": len(summary.split())
            }
            
        except Exception as e:
            logger.error(f"PEGASUS summarization error: {e}")
            raise


class DomainSpecificSummarizer(BaseSummarizer):
    """
    Domain-specific summarizer using BERT (extractive) + PEGASUS (abstractive).
    
    Two-stage pipeline:
      1. Domain BERT encodes sentences and scores them for domain relevance
         (Legal-BERT for legal docs, Clinical-BERT for medical docs)
      2. Top-ranked sentences are fed to PEGASUS for abstractive summarization
    
    This ensures the domain BERT model genuinely influences the output by
    performing domain-aware extractive pre-selection before abstraction.
    """
    
    def __init__(self, domain: str):
        """
        Initialize domain-specific summarizer
        domain: 'legal' or 'medical'
        """
        self.domain = domain
        
        if domain == "legal":
            bert_model = settings.LEGAL_BERT_MODEL
        elif domain == "medical":
            bert_model = settings.CLINICAL_BERT_MODEL
        else:
            raise ValueError(f"Unknown domain: {domain}")
        
        super().__init__(bert_model)
        
        # Stage 1: Domain BERT for extractive sentence scoring
        from transformers import AutoModel
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert_model = AutoModel.from_pretrained(bert_model)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
        # Stage 2: PEGASUS for abstractive generation
        self.pegasus_tokenizer = AutoTokenizer.from_pretrained(settings.PEGASUS_MODEL)
        self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(settings.PEGASUS_MODEL)
        self.pegasus_model.to(self.device)
        self.pegasus_model.eval()
        
        logger.info(f"Domain-specific pipeline initialized: {bert_model} -> PEGASUS")
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _score_sentences(self, sentences: List[str]) -> List[float]:
        """
        Score sentences using domain BERT embeddings.
        Sentences whose embeddings are closest to the document-level
        embedding (centroid) are considered most domain-relevant.
        """
        if not sentences:
            return []
        
        embeddings = []
        for sentence in sentences:
            inputs = self.bert_tokenizer(
                sentence,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding as sentence representation
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding.squeeze(0))
        
        # Stack all embeddings and compute document centroid
        emb_matrix = torch.stack(embeddings)  # (N, hidden_dim)
        centroid = emb_matrix.mean(dim=0, keepdim=True)  # (1, hidden_dim)
        
        # Cosine similarity of each sentence to the centroid
        similarities = torch.nn.functional.cosine_similarity(emb_matrix, centroid)
        scores = similarities.cpu().tolist()
        
        return scores
    
    def _extract_top_sentences(self, text: str, top_ratio: float = 0.6) -> str:
        """
        Extract top domain-relevant sentences using BERT scoring.
        Returns a condensed version of the text ordered by original position.
        """
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 3:
            return text  # Too short to extract from
        
        scores = self._score_sentences(sentences)
        
        # Pair sentences with scores and original indices
        scored = list(enumerate(zip(sentences, scores)))
        
        # Select top N sentences by score
        n_select = max(3, int(len(sentences) * top_ratio))
        scored_sorted = sorted(scored, key=lambda x: x[1][1], reverse=True)
        top_indices = sorted([idx for idx, _ in scored_sorted[:n_select]])
        
        # Reconstruct text in original order
        extracted = " ".join(sentences[i] for i in top_indices)
        
        logger.info(
            f"Domain BERT extracted {len(top_indices)}/{len(sentences)} sentences "
            f"(top scores: {sorted(scores, reverse=True)[:3]})"
        )
        
        return extracted
    
    def summarize(self, text: str, max_length: int = 512, min_length: int = 100) -> Dict:
        """
        Generate domain-aware summary using two-stage pipeline:
        1. Domain BERT extracts most relevant sentences
        2. PEGASUS generates abstractive summary from extracted content
        """
        start_time = time.time()
        
        try:
            # Stage 1: Domain BERT extractive pre-selection
            logger.info(f"Stage 1: {self.domain} BERT extractive scoring...")
            extracted_text = self._extract_top_sentences(text, top_ratio=0.6)
            
            # Stage 2: PEGASUS abstractive generation on extracted content
            logger.info("Stage 2: PEGASUS abstractive generation...")
            inputs = self.pegasus_tokenizer(
                extracted_text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.pegasus_model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
            
            summary = self.pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            # Clean up PEGASUS <n> newline tokens
            summary = summary.replace(".<n>", ". ").replace("<n>", " ").strip()
            generation_time = time.time() - start_time
            
            return {
                "summary": summary,
                "model_name": f"{self.domain}_bert_pegasus",
                "model_type": f"{self.domain}_bert_pegasus",
                "generation_time": generation_time,
                "summary_length": len(summary.split()),
                "domain": self.domain,
                "pipeline": f"{self.model_name} (extract) + PEGASUS (abstract)"
            }
            
        except Exception as e:
            logger.error(f"Domain-specific summarization error: {e}")
            raise


class GeminiSummarizer(BaseSummarizer):
    """Google Gemini-based summarizer (FREE API available!)"""
    
    def __init__(self):
        super().__init__(settings.GEMINI_MODEL)
        
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured. Get free key at: https://makersuite.google.com/app/apikey")
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.model_name)
    
    def summarize(
        self, 
        text: str, 
        max_length: int = 512, 
        min_length: int = 100,
        domain: Optional[str] = None
    ) -> Dict:
        """Generate summary using Gemini"""
        start_time = time.time()
        
        # Create domain-aware prompt
        if domain == "legal":
            prompt = f"""Summarize the following legal document. Focus on key legal clauses, parties involved, obligations, and important terms. Maintain legal terminology and precision.

Document:
{text[:8000]}

Provide a concise summary in about {max_length} words:"""
        elif domain == "medical":
            prompt = f"""Summarize the following medical document. Focus on patient information, diagnosis, treatment plan, medications, and key findings. Use proper medical terminology.

Document:
{text[:8000]}

Provide a concise summary in about {max_length} words:"""
        else:
            prompt = f"""Provide a concise, comprehensive summary of the following document in about {max_length} words:

{text[:8000]}

Summary:"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=settings.GEMINI_MAX_TOKENS,
                    temperature=settings.GEMINI_TEMPERATURE,
                )
            )
            
            summary = response.text.strip()
            generation_time = time.time() - start_time
            
            return {
                "summary": summary,
                "model_name": self.model_name,
                "model_type": "gemini",
                "generation_time": generation_time,
                "summary_length": len(summary.split()),
                "domain": domain,
                "provider": "google"
            }
            
        except Exception as e:
            logger.error(f"Gemini summarization error: {e}")
            raise


class GPTSummarizer(BaseSummarizer):
    """OpenAI GPT-based summarizer (requires paid API key)"""
    
    def __init__(self):
        super().__init__(settings.OPENAI_MODEL)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed. Run: pip install openai")
        
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    def summarize(
        self, 
        text: str, 
        max_length: int = 512, 
        min_length: int = 100,
        domain: Optional[str] = None
    ) -> Dict:
        """Generate summary using GPT"""
        start_time = time.time()
        
        # Create domain-aware prompt
        if domain == "legal":
            prompt = f"""Summarize the following legal document. Focus on key legal clauses, parties involved, obligations, and important terms. Maintain legal terminology and precision.

Document:
{text}

Summary:"""
        elif domain == "medical":
            prompt = f"""Summarize the following medical document. Focus on patient information, diagnosis, treatment plan, medications, and key findings. Use proper medical terminology.

Document:
{text}

Summary:"""
        else:
            prompt = f"""Provide a concise, comprehensive summary of the following document:

{text}

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert document summarizer. Provide clear, accurate, and concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=settings.OPENAI_TEMPERATURE
            )
            
            summary = response.choices[0].message.content.strip()
            generation_time = time.time() - start_time
            
            return {
                "summary": summary,
                "model_name": self.model_name,
                "model_type": "gpt",
                "generation_time": generation_time,
                "summary_length": len(summary.split()),
                "domain": domain,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"GPT summarization error: {e}")
            raise


class HybridSummarizer:
    """
    Hybrid summarizer that combines multiple approaches
    Can ensemble different models or use them in sequence
    """
    
    def __init__(self):
        self.summarizers = {}
    
    def add_summarizer(self, name: str, summarizer: BaseSummarizer):
        """Add a summarizer to the hybrid system"""
        self.summarizers[name] = summarizer
    
    def summarize_all(
        self, 
        text: str, 
        max_length: int = 512, 
        min_length: int = 100
    ) -> Dict[str, Dict]:
        """Generate summaries using all available summarizers"""
        results = {}
        
        for name, summarizer in self.summarizers.items():
            try:
                logger.info(f"Generating summary with {name}")
                result = summarizer.summarize(text, max_length, min_length)
                results[name] = result
            except Exception as e:
                logger.error(f"Error with {name}: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    def ensemble_summarize(
        self, 
        text: str, 
        max_length: int = 512, 
        min_length: int = 100
    ) -> Dict:
        """
        Generate ensemble summary
        Combines outputs from multiple models
        """
        # Get all summaries
        all_summaries = self.summarize_all(text, max_length, min_length)
        
        # Simple ensemble: concatenate and re-summarize
        # In production, use more sophisticated ensemble methods
        combined_text = "\n\n".join([
            s["summary"] for s in all_summaries.values() 
            if "summary" in s
        ])
        
        # Use best model to summarize the ensemble
        if "gpt" in self.summarizers:
            return self.summarizers["gpt"].summarize(combined_text, max_length, min_length)
        elif "pegasus" in self.summarizers:
            return self.summarizers["pegasus"].summarize(combined_text, max_length, min_length)
        else:
            return list(all_summaries.values())[0]


class SummarizationEngine:
    """Main engine that coordinates all summarizers"""
    
    def __init__(self):
        self.summarizers_cache = {}
    
    def get_summarizer(self, model_type: str, domain: Optional[str] = None):
        """Get or create summarizer instance"""
        cache_key = f"{model_type}_{domain}" if domain else model_type
        
        if cache_key in self.summarizers_cache:
            return self.summarizers_cache[cache_key]
        
        if model_type == "bart":
            summarizer = BARTSummarizer()
        elif model_type == "pegasus":
            summarizer = PegasusSummarizer()
        elif model_type == "gemini":
            summarizer = GeminiSummarizer()
        elif model_type == "gpt":
            summarizer = GPTSummarizer()
        elif model_type in ["legal_bert_pegasus", "clinical_bert_pegasus"]:
            domain = "legal" if "legal" in model_type else "medical"
            summarizer = DomainSpecificSummarizer(domain)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.summarizers_cache[cache_key] = summarizer
        return summarizer
    
    def summarize(
        self, 
        text: str, 
        model_type: str = "bart",
        domain: Optional[str] = None,
        max_length: int = 512,
        min_length: int = 100
    ) -> Dict:
        """Generate summary using specified model"""
        summarizer = self.get_summarizer(model_type, domain)
        
        if hasattr(summarizer, 'summarize'):
            if model_type in ["gpt", "gemini"]:
                return summarizer.summarize(text, max_length, min_length, domain=domain)
            else:
                return summarizer.summarize(text, max_length, min_length)
        else:
            raise ValueError(f"Invalid summarizer for model type: {model_type}")
