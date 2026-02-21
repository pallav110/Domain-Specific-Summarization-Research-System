"""
Pydantic Schemas for API Request/Response
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DomainType(str, Enum):
    """Document Domain"""
    LEGAL = "legal"
    MEDICAL = "medical"
    UNKNOWN = "unknown"


class ModelType(str, Enum):
    """Model Types"""
    BART = "bart"
    PEGASUS = "pegasus"
    LEGAL_BERT_PEGASUS = "legal_bert_pegasus"
    CLINICAL_BERT_PEGASUS = "clinical_bert_pegasus"
    GEMINI = "gemini"  # Google Gemini (FREE!)
    GPT = "gpt"  # OpenAI GPT (optional, paid)


# Document Schemas
class DocumentUploadResponse(BaseModel):
    """Response after document upload"""
    document_id: int
    filename: str
    file_size: int
    word_count: int
    detected_domain: DomainType
    domain_confidence: float
    message: str


class DocumentDetail(BaseModel):
    """Detailed document information"""
    id: int
    filename: str
    original_filename: str
    file_size: int
    file_type: str
    word_count: int
    detected_domain: DomainType
    domain_confidence: float
    upload_timestamp: datetime
    processed: int
    
    class Config:
        from_attributes = True


# Summarization Schemas
class SummarizationRequest(BaseModel):
    """Request to generate summary"""
    document_id: int
    model_types: List[ModelType] = Field(
        default=[ModelType.BART, ModelType.GEMINI],
        description="List of models to use for summarization"
    )
    max_length: Optional[int] = Field(default=512, ge=50, le=2048)
    min_length: Optional[int] = Field(default=100, ge=20, le=512)
    use_domain_specific: bool = Field(
        default=True,
        description="Automatically use domain-specific models"
    )


class SummaryResponse(BaseModel):
    """Summary generation response"""
    id: int
    document_id: int
    model_type: ModelType
    model_name: str
    summary_text: str
    summary_length: int
    generation_time: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Evaluation Schemas
class EvaluationMetrics(BaseModel):
    """Evaluation metrics"""
    rouge_1_f: Optional[float] = None
    rouge_1_p: Optional[float] = None
    rouge_1_r: Optional[float] = None
    rouge_2_f: Optional[float] = None
    rouge_l_f: Optional[float] = None
    rouge_l_p: Optional[float] = None
    rouge_l_r: Optional[float] = None
    bertscore_f1: Optional[float] = None
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    factuality_score: Optional[float] = None
    compression_ratio: Optional[float] = None
    semantic_similarity: Optional[float] = None


class EvaluationResponse(BaseModel):
    """Evaluation response"""
    evaluation_id: int
    summary_id: int
    metrics: EvaluationMetrics
    evaluated_at: datetime
    evaluation_time: Optional[float]
    
    class Config:
        from_attributes = True


# Experiment Schemas
class ExperimentRequest(BaseModel):
    """Request to create experiment"""
    document_id: int
    experiment_name: str
    description: Optional[str] = None
    models_to_test: List[ModelType] = Field(
        default=[
            ModelType.BART,
            ModelType.PEGASUS,
            ModelType.GEMINI
        ]
    )
    evaluate: bool = Field(default=True, description="Automatically evaluate summaries")


class ExperimentResponse(BaseModel):
    """Experiment response"""
    id: int
    experiment_name: str
    document_id: int
    domain: DomainType
    models_tested: List[str]
    results_summary: Optional[Dict[str, Any]] = None
    best_model: Optional[str] = None
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Comparison Schemas
class ModelComparison(BaseModel):
    """Model comparison data"""
    model_name: str
    model_type: ModelType
    summary_id: int
    summary_length: int
    generation_time: float
    summary_text: str = ""
    metrics: EvaluationMetrics


class ComparisonResponse(BaseModel):
    """Comparison response"""
    document_id: int
    document_name: str = ""
    domain: DomainType
    word_count: int = 0
    models: List[ModelComparison]
    best_overall: str
    recommendations: List[str]


# Research Dashboard Schemas
class DashboardStats(BaseModel):
    """Dashboard statistics"""
    total_documents: int
    total_summaries: int
    total_experiments: int
    domains_distribution: Dict[str, int]
    model_usage: Dict[str, int]
    average_metrics: Dict[str, float]


class ExperimentResult(BaseModel):
    """Single experiment result for research"""
    experiment_id: int
    experiment_name: str
    domain: DomainType
    model_type: str
    rouge_1: float
    rouge_2: float
    rouge_l: float
    bertscore: float
    factuality: float
    generation_time: float


# Error Schema
class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
