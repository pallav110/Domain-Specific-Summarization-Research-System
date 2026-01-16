"""
Configuration Management for Research System
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application Settings"""
    
    # Application
    APP_NAME: str = "Domain-Specific Summarization Research System"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API
    API_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./research_system.db"
    # For PostgreSQL: postgresql+asyncpg://user:password@localhost/dbname
    
    # Model Paths & Configuration
    MODELS_CACHE_DIR: str = "./models_cache"
    
    # Generic Models
    BART_MODEL: str = "facebook/bart-large-cnn"
    PEGASUS_MODEL: str = "google/pegasus-cnn_dailymail"
    T5_MODEL: str = "t5-base"
    
    # Domain-Specific Models
    LEGAL_BERT_MODEL: str = "nlpaueb/legal-bert-base-uncased"
    CLINICAL_BERT_MODEL: str = "emilyalsentzer/Bio_ClinicalBERT"
    
    # Classification Model
    DOMAIN_CLASSIFIER_MODEL: str = "bert-base-uncased"  # Will be fine-tuned
    
    # LLM Configuration - Google Gemini (Free tier available!)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"  # Free tier: gemini-1.5-flash (fast) or gemini-1.5-pro (better quality)
    GEMINI_MAX_TOKENS: int = 2048
    GEMINI_TEMPERATURE: float = 0.3
    
    # Optional: OpenAI (if you have API key)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    USE_GEMINI: bool = True  # Set to False to use OpenAI instead
    
    # Processing Configuration
    MAX_CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 128
    MAX_DOCUMENT_SIZE_MB: int = 50
    
    # Summarization
    SUMMARY_MAX_LENGTH: int = 512
    SUMMARY_MIN_LENGTH: int = 100
    
    # File Upload
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_EXTENSIONS: set = {".pdf", ".txt", ".docx"}
    MAX_FILE_SIZE: int = 10485760  # 10MB in bytes
    
    # Experiment Logging
    EXPERIMENT_LOG_DIR: str = "./experiments"
    RESULTS_DIR: str = "./results"
    
    # Evaluation Metrics
    COMPUTE_BERTSCORE: bool = True
    COMPUTE_FACTUALITY: bool = True
    
    # API Metadata
    API_TITLE: str = "Domain-Specific Summarization Research API"
    API_VERSION: str = "1.0.0"
    CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
