"""
Database Models for Research System
"""
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from database import Base


class DomainType(str, enum.Enum):
    """Document Domain Types"""
    LEGAL = "legal"
    MEDICAL = "medical"
    UNKNOWN = "unknown"


class ModelType(str, enum.Enum):
    """Summarization Model Types"""
    BART = "bart"
    PEGASUS = "pegasus"
    LEGAL_BERT_PEGASUS = "legal_bert_pegasus"
    CLINICAL_BERT_PEGASUS = "clinical_bert_pegasus"
    GEMINI = "gemini"  # Google Gemini (FREE!)
    GPT = "gpt"  # OpenAI GPT (optional, paid)
    ENSEMBLE = "ensemble"  # Sentence-level ensemble (Novelty)


class Document(Base):
    """Uploaded Document Model"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=False)  # in bytes
    file_type = Column(String(50), nullable=False)  # pdf, txt, docx
    
    # Content
    raw_text = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=False)
    
    # Domain Classification
    detected_domain = Column(Enum(DomainType), default=DomainType.UNKNOWN)
    domain_confidence = Column(Float, default=0.0)
    
    # Metadata
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    processed = Column(Integer, default=0)  # 0: pending, 1: processed, -1: error
    
    # Relationships
    summaries = relationship("Summary", back_populates="document", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="document", cascade="all, delete-orphan")


class Summary(Base):
    """Generated Summary Model"""
    __tablename__ = "summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Summary Details
    model_type = Column(Enum(ModelType), nullable=False)
    model_name = Column(String(255), nullable=False)
    summary_text = Column(Text, nullable=False)
    summary_length = Column(Integer, nullable=False)
    
    # Generation Parameters
    generation_params = Column(JSON, nullable=True)  # max_length, min_length, temperature, etc.
    
    # Metrics
    generation_time = Column(Float, nullable=True)  # in seconds
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="summaries")
    evaluations = relationship("Evaluation", back_populates="summary", cascade="all, delete-orphan")


class Experiment(Base):
    """Research Experiment Model"""
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Experiment Metadata
    experiment_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    domain = Column(Enum(DomainType), nullable=False)
    
    # Models Tested
    models_tested = Column(JSON, nullable=False)  # List of model names
    
    # Results Summary
    results_summary = Column(JSON, nullable=True)  # Aggregated metrics
    best_model = Column(String(255), nullable=True)
    
    # Metadata
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(50), default="running")  # running, completed, failed
    
    # Relationships
    document = relationship("Document", back_populates="experiments")


class Evaluation(Base):
    """Evaluation Metrics Model"""
    __tablename__ = "evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    summary_id = Column(Integer, ForeignKey("summaries.id"), nullable=False)
    
    # ROUGE Metrics
    rouge_1_f = Column(Float, nullable=True)
    rouge_1_p = Column(Float, nullable=True)
    rouge_1_r = Column(Float, nullable=True)
    
    rouge_2_f = Column(Float, nullable=True)
    rouge_2_p = Column(Float, nullable=True)
    rouge_2_r = Column(Float, nullable=True)
    
    rouge_l_f = Column(Float, nullable=True)
    rouge_l_p = Column(Float, nullable=True)
    rouge_l_r = Column(Float, nullable=True)
    
    # BERTScore
    bertscore_f1 = Column(Float, nullable=True)
    bertscore_precision = Column(Float, nullable=True)
    bertscore_recall = Column(Float, nullable=True)
    
    # Factuality
    factuality_score = Column(Float, nullable=True)
    factuality_method = Column(String(100), nullable=True)  # qags, factcc, etc.
    
    # Additional Metrics
    compression_ratio = Column(Float, nullable=True)
    semantic_similarity = Column(Float, nullable=True)
    
    # Full Metrics JSON
    full_metrics = Column(JSON, nullable=True)
    
    # Metadata
    evaluated_at = Column(DateTime, default=datetime.utcnow)
    evaluation_time = Column(Float, nullable=True)  # in seconds
    
    # Relationships
    summary = relationship("Summary", back_populates="evaluations")
