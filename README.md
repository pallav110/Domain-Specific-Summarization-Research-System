# Domain-Specific Abstractive Summarization Research System

A comprehensive research system for comparing generic vs domain-specific NLP models for legal and medical document summarization.

## 📋 Table of Contents
- [Overview](#overview)
- [Research Question](#research-question)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Running Experiments](#running-experiments)
- [API Documentation](#api-documentation)
- [Frontend Features](#frontend-features)
- [Evaluation Metrics](#evaluation-metrics)
- [Sample Data](#sample-data)
- [Research Guidelines](#research-guidelines)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This system enables undergraduate research on domain-specific abstractive summarization by providing:
- Automated document processing and domain classification
- Multiple summarization models (generic and domain-specific)
- Comprehensive evaluation metrics (ROUGE, BERTScore, Factuality)
- Web-based interface for experiment management
- Automated workflows via n8n
- Publication-ready experimental results

## 🔬 Research Question

**"Does domain-specific NLP significantly improve summarization quality over generic models?"**

The system tests this by comparing:
- **Generic Models**: BART, PEGASUS, T5
- **Domain-Specific Models**: Legal-BERT + PEGASUS, Clinical-BERT + PEGASUS
- **LLM Baseline**: GPT-4

On two domains:
- **Legal**: Contracts, court judgments, legal documents
- **Medical**: Clinical notes, patient records, medical research papers

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Next.js Frontend                        │
│  (Upload, View Summaries, Compare Models, Dashboard)        │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API
┌──────────────────────────▼──────────────────────────────────┐
│                      FastAPI Backend                         │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ Document       │  │ Domain       │  │ Summarization   │ │
│  │ Processor      │  │ Classifier   │  │ Engine          │ │
│  └────────────────┘  └──────────────┘  └─────────────────┘ │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ Evaluation     │  │ Experiment   │  │ Database        │ │
│  │ Metrics        │  │ Runner       │  │ (SQLite)        │ │
│  └────────────────┘  └──────────────┘  └─────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    n8n Automation                            │
│  (Batch Processing, Scheduled Experiments)                   │
└─────────────────────────────────────────────────────────────┘
```

## 🛠 Tech Stack

### Backend
- **Framework**: FastAPI
- **Database**: SQLite (PostgreSQL ready)
- **ML/NLP**: 
  - Transformers (Hugging Face)
  - LangChain
  - PyTorch
- **Models**:
  - facebook/bart-large-cnn
  - google/pegasus-cnn_dailymail
  - nlpaueb/legal-bert-base-uncased
  - emilyalsentzer/Bio_ClinicalBERT
  - **Google Gemini Pro (FREE API!)** - Replaces GPT-4
- **Evaluation**:
  - ROUGE (rouge-score)
  - BERTScore (bert-score)
  - Sentence Transformers (semantic similarity)

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **HTTP Client**: Axios


## 📦 Setup & Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- CUDA-compatible GPU (recommended)
- **Google Gemini API key (FREE!)** - Get at: https://makersuite.google.com/app/apikey

### 1. Clone Repository
```bash
cd "Domain-Specific Summarization Research System"
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
copy .env.example .env

# Edit .env and add your FREE Google Gemini API key
# Get key at: https://makersuite.google.com/app/apikey
# GEMINI_API_KEY=your_key_here

# Initialize database
python -c "import asyncio; from database import init_db; asyncio.run(init_db())"

# Start backend server
python main.py
```

The backend will start at http://localhost:8000

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will start at http://localhost:3000

### 4. n8n Setup (Optional)

```bash
# Install n8n globally
npm install -g n8n

# Start n8n
n8n start

# Import workflows from n8n/ directory
```

n8n will start at http://localhost:5678

## 🚀 Usage

### Basic Workflow

1. **Upload Document**
   - Go to http://localhost:3000
   - Upload a PDF or TXT file (legal or medical)
   - System automatically detects domain

2. **Generate Summaries**
   - Navigate to document details
   - Select models to use
   - Click "Generate Summaries"
   - Wait for processing

3. **View Evaluations**
   - System automatically evaluates each summary
   - View ROUGE, BERTScore, and factuality scores

4. **Compare Models**
   - Go to comparison page
   - View side-by-side metrics
   - Identify best-performing model

5. **Research Dashboard**
   - View aggregate statistics
   - Analyze trends across experiments
   - Export data for publication

## 🧪 Running Experiments

### Single Document Experiment

```bash
# Using Python script
cd backend
python experiments/run_single_experiment.py --document_id 1
```

### Batch Experiments

```bash
# Process all documents
python experiments/run_batch_experiments.py

# Process specific domain
python experiments/run_batch_experiments.py --domain legal
```

### Via API

```bash
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": 1,
    "experiment_name": "Legal Contract Comparison",
    "description": "Comparing all models on employment contract",
    "models_to_test": ["bart", "pegasus", "legal_bert_pegasus", "gemini"],
    "evaluate": true
  }'
```

## 📚 API Documentation

### Documents

#### Upload Document
```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: <PDF or TXT file>
```

#### Get Document
```http
GET /api/v1/documents/{document_id}
```

#### List Documents
```http
GET /api/v1/documents?skip=0&limit=50&domain=legal
```

### Summarization

#### Generate Summaries
```http
POST /api/v1/summarize
Content-Type: application/json

{
  "document_id": 1,
  "model_types": ["bart", "pegasus", "gemini"],
  "max_length": 512,
  "min_length": 100,
  "use_domain_specific": true
}
```

#### Get Summary
```http
GET /api/v1/summaries/{summary_id}
```

### Evaluation

#### Evaluate Summary
```http
POST /api/v1/evaluate/{summary_id}?compute_factuality=true
```

### Experiments

#### Create Experiment
```http
POST /api/v1/experiments
Content-Type: application/json

{
  "document_id": 1,
  "experiment_name": "Model Comparison",
  "models_to_test": ["bart", "pegasus", "gemini"],
  "evaluate": true
}
```

#### Get Experiment Results
```http
GET /api/v1/experiments/{experiment_id}
```

### Comparison & Dashboard

#### Compare Models
```http
GET /api/v1/compare/{document_id}
```

#### Dashboard Statistics
```http
GET /api/v1/dashboard/stats
```

#### Research Results
```http
GET /api/v1/research/results?domain=legal
```

## 🖥 Frontend Features

### Pages

1. **Home** (`/`)
   - Quick upload
   - Recent documents
   - Research overview

2. **Documents** (`/documents`)
   - List all documents
   - Filter by domain
   - Upload new documents

3. **Document Detail** (`/documents/[id]`)
   - View document info
   - Generate summaries
   - View all summaries

4. **Experiments** (`/experiments`)
   - Create experiments
   - View experiment results
   - Compare models

5. **Dashboard** (`/dashboard`)
   - Aggregate statistics
   - Charts and visualizations
   - Research insights

## 📊 Evaluation Metrics

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **Range**: 0.0 to 1.0 (higher is better)

### BERTScore
- Semantic similarity using contextual embeddings
- **Precision**: How much of summary is supported
- **Recall**: How much of source is covered
- **F1**: Harmonic mean
- **Range**: 0.0 to 1.0 (higher is better)

### Factuality Score
- NLI-based or overlap-based consistency check
- Measures factual accuracy
- **Range**: 0.0 to 1.0 (higher is better)

### Compression Ratio
- Summary length / Source length
- **Lower values** = more compression

### Semantic Similarity
- Cosine similarity of sentence embeddings
- **Range**: 0.0 to 1.0 (higher is better)

## 📁 Sample Data

Sample documents are provided in `sample_data/`:

### Legal Documents
- `employment_contract.txt` - Employment agreement
- Use for testing legal domain classification

### Medical Documents
- `patient_record.txt` - Patient medical record
- Use for testing medical domain classification

### Loading Sample Data

```bash
cd backend
python scripts/load_sample_data.py
```

## 📖 Research Guidelines

### For Undergraduate Research Projects

#### 1. Hypothesis Formation
- **H1**: Domain-specific models outperform generic models on domain-specific metrics
- **H2**: GPT-4 provides best overall performance but at higher cost

#### 2. Experimental Design
- Test each model on minimum 10 documents per domain (20 total)
- Use same evaluation criteria across all models
- Document all hyperparameters

#### 3. Data Collection
```python
# Export results to CSV
import pandas as pd
from database import async_session_maker
from models.db_models import Experiment, Summary, Evaluation

# Fetch all evaluations
# Create DataFrame
# Export to CSV for analysis
```

#### 4. Statistical Analysis
- Calculate mean and standard deviation for each metric
- Perform paired t-tests between models
- Create visualization comparing models

#### 5. Publication Preparation
- Use dashboard charts for figures
- Export results table for paper
- Document methodology thoroughly

### Recommended Experiments

1. **Experiment 1: Generic Model Comparison**
   - Compare BART, PEGASUS, T5
   - Measure baseline performance

2. **Experiment 2: Domain-Specific Impact**
   - Compare generic vs Legal-BERT for legal docs
   - Compare generic vs Clinical-BERT for medical docs

3. **Experiment 3: LLM Benchmark**
   - Compare GPT-4 with all other models
   - Analyze cost vs quality tradeoff

4. **Experiment 4: Cross-Domain Analysis**
   - Test legal models on medical docs (and vice versa)
   - Measure domain specificity importance

## 🔧 Troubleshooting

### Backend Issues

**Models taking too long to load:**
```bash
# Download models in advance
python scripts/download_models.py
```

**Out of memory errors:**
```python
# In config.py, reduce batch size or use CPU
DEVICE = "cpu"
```

**Database errors:**
```bash
# Reset database
rm research_system.db
python -c "import asyncio; from database import init_db; asyncio.run(init_db())"
```

### Frontend Issues

**API connection errors:**
- Check backend is running on port 8000
- Verify NEXT_PUBLIC_API_URL in .env.local

**Build errors:**
```bash
# Clear cache and reinstall
rm -rf .next node_modules
npm install
npm run dev
```

### Model Issues

**CUDA errors:**
```python
# Force CPU usage in config.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

**OpenAI API errors:**
- Verify API key in backend/.env
- Check API quota and billing

## 📝 Citation

If you use this system for research, please cite:

```bibtex
@software{domain_summarization_2026,
  title={Domain-Specific Abstractive Summarization Research System},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/repo}
}
```

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For questions or issues:
- Open a GitHub issue
- Email: your.email@university.edu

## 🎓 Acknowledgments

- Hugging Face for transformer models
- OpenAI for GPT-4 API
- FastAPI and Next.js teams
- Research advisors and collaborators

---

**Happy Researching! 🚀**
