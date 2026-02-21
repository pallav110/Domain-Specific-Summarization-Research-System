<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Next.js_16-000000?style=for-the-badge&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/React_19-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

# Domain-Specific Abstractive Summarization Research System

A full-stack research platform for comparing **generic vs domain-specific NLP models** on legal and medical document summarization — featuring three **novel research contributions**: Sentence-Level Ensemble Summarization, Cross-Model Consensus Analysis, and an Adaptive Model Recommender. Upload documents, generate summaries with multiple models, evaluate with comprehensive metrics, fuse outputs via ensemble, and visualize results  —  all from a single web interface.

---

## Table of Contents

- [Research Question](#research-question)
- [Key Features](#key-features)
- [Novel Research Contributions](#novel-research-contributions)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage Workflow](#usage-workflow)
- [Summarization Models](#summarization-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Frontend Pages](#frontend-pages)
- [API Reference](#api-reference)
- [Running Experiments](#running-experiments)
- [Research Guidelines](#research-guidelines)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Research Question

> **Does domain-specific NLP significantly improve summarization quality over generic models?**

The system tests this hypothesis by comparing six models across two specialized domains:

| Generic Models | Domain-Specific Models | LLM Baselines |
|---|---|---|
| BART (facebook/bart-large-cnn) | Legal-BERT + PEGASUS | Google Gemini 1.5 Flash |
| PEGASUS (google/pegasus-cnn_dailymail) | Clinical-BERT + PEGASUS | GPT-4 Turbo (optional) |

**Domains:** Legal (contracts, court opinions, statutes) and Medical (clinical notes, research papers, patient records)

Additionally, the system introduces **three novel contributions** that go beyond standard model comparison — see [Novel Research Contributions](#novel-research-contributions).

---

## Key Features

- **Multi-Model Summarization** — Generate summaries from 6 different models in one click
- **Automatic Domain Detection** — Zero-shot classification using BART-MNLI with keyword boosting
- **Comprehensive Evaluation** — ROUGE-1/2/L, BERTScore, factuality, semantic similarity, compression ratio
- **Interactive PDF Viewer** — View uploaded PDFs with color-coded highlights showing which source text each model used
- **Model Comparison** — Side-by-side metrics with radar charts and per-metric rankings
- **Sentence-Level Ensemble** — Fuse outputs from all models into a single superior summary via clustering
- **Cross-Model Consensus** — Measure inter-model agreement with a novel consensus metric and heatmap visualization
- **Adaptive Model Recommender** — ML-powered prediction of the best model for any document before summarization
- **Research Dashboard** — Aggregate statistics, domain distribution, model usage trends
- **Data Export** — Download results as CSV or JSON for external analysis
- **Experiment Management** — Create, track, and reproduce research experiments

---

## Novel Research Contributions

Beyond standard model comparison, this system introduces **three novel features** that represent original research contributions:

### 1. Sentence-Level Ensemble Summarization (CGSSE)

> *Cluster-Guided Sentence-Selection Ensemble* — fuses outputs from all models into a single, superior summary.

**Algorithm:**

```
Input:  Summaries S₁, S₂, ..., Sₙ from N models + source document D
Output: Ensemble summary E

1. Extract all sentences from S₁ ∪ S₂ ∪ ... ∪ Sₙ
2. Compute sentence embeddings using SentenceTransformer (all-MiniLM-L6-v2)
3. Build cosine similarity matrix; greedily cluster sentences with similarity > 0.7
4. Score each cluster:
       score = agreement_count × avg_cosine_similarity_to_source
   where agreement_count = number of distinct models contributing to the cluster
5. Select representative sentence per cluster (highest source similarity)
6. Pick top-k clusters by score; order representatives by document position
7. Join into final ensemble summary E
```

**Why it's novel:** Unlike traditional ensemble methods that operate at the model-weight level, CGSSE works at the *sentence level* across heterogeneous architectures (BART, PEGASUS, BERT-pipelines, LLMs). It rewards cross-model agreement while maintaining source relevance.

### 2. Cross-Model Consensus Metric (IMSA)

> *Inter-Model Semantic Agreement* — a new evaluation metric measuring how much different models agree on content.

**Algorithm:**

```
Input:  Summaries S₁, S₂, ..., Sₙ from N models
Output: Consensus score, agreement matrix, unique content ratios

1. Split each summary into sentences; compute embeddings
2. Agreement Matrix (N×N):
   For each model pair (A, B):
       score(A,B) = avg over sentences a ∈ A of max cosine_sim(a, b) for b ∈ B
3. Per-sentence consensus:
   For each sentence s in any model, count how many models have
   a semantically similar sentence (cosine > 0.7)
4. Consensus Score = fraction of sentences with agreement count ≥ 2
5. Unique Content Ratio = per-model fraction of sentences unique to that model
6. High-Agreement Sentences = top sentences sorted by consensus count
```

**Why it's novel:** Existing metrics (ROUGE, BERTScore) compare summaries to a *reference*. IMSA compares summaries to *each other*, measuring inter-model agreement without requiring ground-truth references. A high consensus score indicates robust, model-independent content selection.

### 3. Adaptive Model Recommender

> Predicts the best-performing model for a given document *before* summarization.

**Feature Vector (7 dimensions):**

| Feature | Description |
|---|---|
| `word_count` | Total words in document |
| `sentence_count` | Total sentences |
| `avg_sentence_length` | Average words per sentence |
| `domain_confidence` | Classifier confidence score |
| `type_token_ratio` | Vocabulary richness (unique/total words) |
| `is_legal` | Binary domain indicator |
| `is_medical` | Binary domain indicator |

**Training:** RandomForest classifier (n_estimators=50, max_depth=5) trained on past evaluation results. The target label for each document is the model that achieved the highest weighted score:

```
weighted_score = 0.3 × ROUGE-L + 0.3 × BERTScore + 0.2 × Semantic + 0.2 × Factuality
```

**Fallback:** When insufficient training data exists, a rule-based fallback is used:

- Legal documents → Legal-BERT + PEGASUS
- Medical documents → Clinical-BERT + PEGASUS
- Long documents (>2000 words) → Gemini
- Default → BART

**Why it's novel:** Instead of running all models and comparing after the fact, the recommender predicts the optimal model *a priori*, saving computational resources and providing actionable guidance.

---

## Architecture

```
                          ┌──────────────────────────────────────────┐
                          │           Next.js 16 Frontend            │
                          │                                          │
                          │  Documents | PDF Viewer | Compare        │
                          │  Results   | Statistics | Dashboard      │
                          │  Consensus Heatmap | Ensemble | Recommender
                          └──────────────────┬───────────────────────┘
                                             │ REST API
                          ┌──────────────────▼───────────────────────┐
                          │            FastAPI Backend                │
                          │                                          │
                          │  ┌──────────┐  ┌──────────────────────┐  │
                          │  │ Document  │  │ Domain Classifier    │  │
                          │  │ Processor │  │ (BART-MNLI)          │  │
                          │  └──────────┘  └──────────────────────┘  │
                          │  ┌──────────┐  ┌──────────────────────┐  │
                          │  │ Summary  │  │ Evaluation           │  │
                          │  │ Engine   │  │ (ROUGE/BERT/NLI)     │  │
                          │  └──────────┘  └──────────────────────┘  │
                          │  ┌─────────────────────────────────────┐  │
                          │  │      Novel Research Modules         │  │
                          │  │  Ensemble | Consensus | Recommender │  │
                          │  └─────────────────────────────────────┘  │
                          │  ┌─────────────────────────────────────┐  │
                          │  │   SQLite (async) + File Storage     │  │
                          │  └─────────────────────────────────────┘  │
                          └──────────────────────────────────────────┘
```

---

## Tech Stack

### Backend
| Component | Technology |
|---|---|
| Framework | FastAPI with async SQLAlchemy |
| Database | SQLite (async via aiosqlite) |
| ML/NLP | PyTorch, Hugging Face Transformers, Sentence-Transformers |
| Evaluation | rouge-score, bert-score, NLI-based factuality |
| Domain Classification | facebook/bart-large-mnli (zero-shot) |
| LLM Integration | Google Gemini API (free tier), OpenAI GPT-4 (optional) |
| Ensemble & Consensus | SentenceTransformers (all-MiniLM-L6-v2), cosine clustering |
| Model Recommender | scikit-learn RandomForest (with numpy k-NN fallback) |
| Logging | Loguru |

### Frontend
| Component | Technology |
|---|---|
| Framework | Next.js 16.1.6 (Turbopack) |
| UI Library | React 19.2.4 |
| Styling | Tailwind CSS + shadcn/ui (Radix primitives) |
| Charts | Recharts |
| PDF Viewer | react-pdf 10.x (pdf.js) |
| HTTP Client | Axios |
| Icons | Lucide React |

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- Node.js 18+
- CUDA-compatible GPU (recommended, not required)
- Google Gemini API key (free) — get one at [Google AI Studio](https://makersuite.google.com/app/apikey)

### 1. Clone the Repository

```bash
git clone https://github.com/pallav110/Domain-Specific-Summarization-Research-System.git
cd "Domain-Specific Summarization Research System"
```

### 2. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env and add: GEMINI_API_KEY=your_key_here

# Initialize the database
python -c "import asyncio; from database import init_db; asyncio.run(init_db())"

# Start the server
python main.py
```

Backend runs at **http://localhost:8000** — API docs at **http://localhost:8000/docs**

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend runs at **http://localhost:3000**

---

## Usage Workflow

### 1. Upload a Document
Navigate to **Documents** and upload a PDF or TXT file. The system automatically extracts text and classifies the domain (legal/medical) with a confidence score.

### 2. Generate Summaries
Open a document, select which models to run (BART, PEGASUS, Gemini, Legal-BERT, Clinical-BERT, GPT-4), and click **Generate**. Each model produces an abstractive summary independently.

### 3. View the PDF with Highlights
Switch to the **PDF Viewer** tab to see the original document. Toggle model highlights to see which parts of the source each model drew from — color-coded per model:

| Color | Model |
|---|---|
| Orange | BART |
| Violet | PEGASUS |
| Blue | Gemini |
| Green | GPT-4 |
| Cyan | Legal-BERT + PEGASUS |
| Pink | Clinical-BERT + PEGASUS |
| Gold | Ensemble (fused from all models) |

### 4. Evaluate & Compare
Navigate to **Compare Models** for side-by-side metrics including ROUGE, BERTScore, factuality, and semantic similarity. View precision/recall breakdowns with the P/R toggle.

### 5. Use Novel Research Features

- **Model Recommendation** — On the document detail page, an ML-powered recommendation panel appears above the model selector, suggesting the best model with confidence scores. The recommended model is auto-selected.
- **Consensus Analysis** — On the compare page, click **Consensus** to view the N×N agreement heatmap, unique content ratios per model, and high-agreement sentences across all models.
- **Ensemble Summary** — On the compare page, click **Ensemble** to fuse all existing summaries into a single, superior summary using sentence-level clustering. The ensemble appears as a new row in the comparison table.

### 6. Analyze Results
Use **Results**, **Statistics**, and **Dashboard** pages for aggregate analysis. Export data as CSV/JSON for publication.

---

## Summarization Models

### Generic Models

| Model | Description | Speed |
|---|---|---|
| **BART** | `facebook/bart-large-cnn` — Denoising autoencoder fine-tuned on CNN/DailyMail | ~3s |
| **PEGASUS** | `google/pegasus-cnn_dailymail` — Pre-trained with gap-sentence generation | ~3s |

### Domain-Specific Models (Two-Stage Pipeline)

| Model | Stage 1: Extraction | Stage 2: Abstraction | Speed |
|---|---|---|---|
| **Legal-BERT + PEGASUS** | `nlpaueb/legal-bert-base-uncased` extracts key sentences | PEGASUS abstracts the extraction | ~38s |
| **Clinical-BERT + PEGASUS** | `emilyalsentzer/Bio_ClinicalBERT` extracts key sentences | PEGASUS abstracts the extraction | ~38s |

### LLM Baselines

| Model | Description | Speed |
|---|---|---|
| **Gemini** | Google Gemini 1.5 Flash with domain-aware prompting (free API) | ~5s |
| **GPT-4** | OpenAI GPT-4 Turbo (requires paid API key) | ~8s |

---

## Evaluation Metrics

| Metric | What It Measures | Range | Ideal |
|---|---|---|---|
| **ROUGE-1** | Unigram overlap between summary and source | 0–1 | Higher |
| **ROUGE-2** | Bigram overlap | 0–1 | Higher |
| **ROUGE-L** | Longest common subsequence | 0–1 | Higher |
| **BERTScore F1** | Semantic similarity via contextual embeddings | 0–1 | Higher |
| **Factuality** | NLI-based factual consistency with source | 0–1 | Higher |
| **Semantic Similarity** | Cosine similarity of sentence embeddings | 0–1 | Higher |
| **Compression Ratio** | Summary length / source length | 0–1 | Lower |

Each metric also reports **precision** and **recall** variants (visible via the P/R toggle on the comparison page).

---

## Frontend Pages

| Page | Route | Description |
|---|---|---|
| **Home** | `/` | Quick upload, recent documents, overview |
| **Documents** | `/documents` | Browse all documents, filter by domain |
| **Document Detail** | `/documents/[id]` | View info, ML model recommendation, generate summaries, PDF viewer with highlights |
| **Summary Detail** | `/summaries/[id]` | Full summary text and evaluation metrics |
| **Compare Models** | `/compare/[id]` | Side-by-side metrics, radar chart, consensus heatmap, ensemble generation |
| **Experiments** | `/experiments` | Create and manage research experiments |
| **Results** | `/results` | Filterable table of all evaluation results |
| **Statistics** | `/statistics` | Per-model and per-domain statistical analysis |
| **Dashboard** | `/dashboard` | Aggregate stats, charts, domain distribution |

---

## API Reference

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/documents/upload` | Upload PDF/TXT file |
| `GET` | `/api/v1/documents` | List all documents |
| `GET` | `/api/v1/documents/{id}` | Get document details |
| `GET` | `/api/v1/documents/{id}/file` | Download original file |
| `DELETE` | `/api/v1/documents/{id}` | Delete document |

### Summarization

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/summarize` | Generate summaries for selected models |
| `GET` | `/api/v1/summaries/{id}` | Get summary by ID |
| `GET` | `/api/v1/documents/{id}/summaries` | List summaries for a document |

### Evaluation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/evaluate/{summary_id}` | Run evaluation metrics on a summary |
| `GET` | `/api/v1/evaluations/summary/{id}` | Get evaluation results |

### Experiments

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/experiments` | Create an experiment |
| `GET` | `/api/v1/experiments` | List all experiments |
| `GET` | `/api/v1/experiments/{id}` | Get experiment details |

### Novel Research Features

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/ensemble/{document_id}` | Generate ensemble summary from all model outputs |
| `GET` | `/api/v1/consensus/{document_id}` | Compute cross-model consensus metrics and agreement matrix |
| `GET` | `/api/v1/recommend/{document_id}` | Get ML-powered model recommendation for a document |
| `POST` | `/api/v1/recommender/train` | Train the recommender on existing evaluation data |

### Analysis & Export

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/compare/{document_id}` | Compare all models on a document |
| `GET` | `/api/v1/dashboard/stats` | Aggregate dashboard statistics |
| `GET` | `/api/v1/research/results` | Detailed results for analysis |
| `GET` | `/api/v1/statistics/analysis` | Per-model statistical analysis |
| `GET` | `/api/v1/export/csv` | Export results as CSV |
| `GET` | `/api/v1/export/json` | Export results as JSON |

Full interactive docs available at **http://localhost:8000/docs** when the backend is running.

---

## Running Experiments

### Via the Web Interface

1. Go to **Experiments** page
2. Select a document and models to test
3. Click **Create Experiment**
4. Results appear automatically with evaluations

### Via the API

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

---

## Research Guidelines

### Recommended Experiment Design

1. **Collect Data** — Upload at least 10 documents per domain (20+ total)
2. **Run All Models** — Generate summaries with all 6 models for each document
3. **Evaluate** — Run evaluations to get ROUGE, BERTScore, factuality, and semantic similarity
4. **Compare** — Use the Statistics page for per-model averages and standard deviations
5. **Export** — Download CSV/JSON for statistical analysis in Python/R

### Suggested Experiments

| Experiment | Goal |
|---|---|
| Generic Model Baseline | Compare BART vs PEGASUS across both domains |
| Domain-Specific Impact | Legal-BERT vs generic models on legal docs |
| LLM Benchmark | Gemini/GPT-4 vs all other models |
| Cross-Domain Analysis | Test legal models on medical docs and vice versa |
| Ensemble vs Individual | Compare ensemble summary quality against best individual model |
| Consensus Correlation | Analyze whether high consensus score correlates with higher quality |
| Recommender Accuracy | Train recommender, then evaluate if its predictions match actual best models |

### Statistical Analysis

```python
import pandas as pd

# Load exported results
df = pd.read_csv("research_results.csv")

# Per-model averages
print(df.groupby("model_type")[["rouge_l_f", "bertscore_f1", "factuality_score"]].mean())

# Paired t-test between models
from scipy.stats import ttest_rel
bart = df[df.model_type == "bart"]["rouge_l_f"]
legal = df[df.model_type == "legal_bert_pegasus"]["rouge_l_f"]
t_stat, p_val = ttest_rel(bart, legal)
print(f"t={t_stat:.3f}, p={p_val:.4f}")
```

---

## Troubleshooting

### Backend

| Problem | Solution |
|---|---|
| Models slow to load | First run downloads ~5GB of models. Subsequent runs use cache in `./models_cache` |
| Out of memory (OOM) | Set `DEVICE = "cpu"` in `backend/config.py` |
| Database errors | Delete `research_system.db` and re-initialize: `python -c "import asyncio; from database import init_db; asyncio.run(init_db())"` |
| Gemini API errors | Verify `GEMINI_API_KEY` in `.env`. Free tier allows 15 RPM |

### Frontend

| Problem | Solution |
|---|---|
| API connection refused | Ensure backend is running on port 8000. Check `NEXT_PUBLIC_API_URL` in `frontend/.env.local` |
| Build errors | Delete `.next` and `node_modules`, then `npm install && npm run dev` |
| PDF viewer not loading | Requires the document to be a PDF (`.txt` files show the viewer tab as disabled) |

---

## Citation

```bibtex
@software{domain_summarization_2026,
  title   = {Domain-Specific Abstractive Summarization Research System},
  author  = {Pallav},
  year    = {2026},
  url     = {https://github.com/pallav110/Domain-Specific-Summarization-Research-System}
}
```

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformer models and the Transformers library
- [Google AI](https://ai.google.dev/) for Gemini API free tier
- [FastAPI](https://fastapi.tiangolo.com/) and [Next.js](https://nextjs.org/) teams
- [shadcn/ui](https://ui.shadcn.com/) for the component library
