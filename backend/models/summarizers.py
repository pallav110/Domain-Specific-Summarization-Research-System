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
    """Domain-specific summarizer using BERT + PEGASUS"""
    
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
        
        # Use PEGASUS for actual summarization
        self.pegasus_tokenizer = AutoTokenizer.from_pretrained(settings.PEGASUS_MODEL)
        self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(settings.PEGASUS_MODEL)
        self.pegasus_model.to(self.device)
        self.pegasus_model.eval()
        
        # BERT for encoding (optional - can enhance context)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
    
    def summarize(self, text: str, max_length: int = 512, min_length: int = 100) -> Dict:
        """Generate domain-aware summary"""
        start_time = time.time()
        
        try:
            # Use PEGASUS with domain context
            # In production, you would fine-tune PEGASUS on domain data
            inputs = self.pegasus_tokenizer(
                text,
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
            generation_time = time.time() - start_time
            
            return {
                "summary": summary,
                "model_name": f"{self.domain}_bert_pegasus",
                "model_type": f"{self.domain}_bert_pegasus",
                "generation_time": generation_time,
                "summary_length": len(summary.split()),
                "domain": self.domain
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
                return summarizer.summarize(text, max_length, min_length, domain)
            else:
                return summarizer.summarize(text, max_length, min_length)
        else:
            raise ValueError(f"Invalid summarizer for model type: {model_type}")
