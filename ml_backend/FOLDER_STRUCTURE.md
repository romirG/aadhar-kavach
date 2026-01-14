# UIDAI ML Backend - Folder Structure

```
ml_backend/
├── main.py                          # FastAPI application entry point
├── config.py                        # Configuration and settings
├── requirements.txt                 # Python dependencies
├── API_REFERENCE.md                 # API documentation
├── FOLDER_STRUCTURE.md              # This file
│
├── api/                             # API Layer
│   ├── __init__.py
│   ├── schemas.py                   # Pydantic request/response models
│   ├── schemas_xai.py               # XAI-specific schemas
│   │
│   └── routes/                      # API Endpoints
│       ├── __init__.py
│       ├── datasets.py              # GET /datasets endpoints
│       ├── selection.py             # POST /select-dataset endpoint
│       ├── analysis.py              # POST /analyze, GET /results
│       ├── visualizations.py        # GET /visualizations endpoints
│       └── xai_endpoints.py         # Explainability endpoints
│
├── data/                            # Data Processing Layer
│   ├── __init__.py
│   ├── ingestion.py                 # Data fetching from data.gov.in
│   ├── preprocessing.py             # Data cleaning and normalization
│   ├── feature_engineering.py       # Feature creation
│   └── advanced_preprocessing.py    # ML-ready pipeline
│
├── models/                          # ML Models Layer
│   ├── __init__.py
│   ├── ensemble_detector.py         # Ensemble anomaly detection
│   ├── isolation_forest.py          # Isolation Forest model
│   ├── autoencoder.py               # PyTorch Autoencoder
│   ├── clustering.py                # HDBSCAN clustering
│   └── ensemble.py                  # Model ensemble orchestration
│
├── explainability/                  # XAI Layer
│   ├── __init__.py
│   ├── feature_attribution.py       # Feature importance & reasons
│   ├── reason_generator.py          # Human-readable explanations
│   ├── deviation_analyzer.py        # Deviation analysis
│   └── fraud_explainer.py           # Unified fraud explainer
│
├── visualization/                   # Visualization Layer
│   ├── __init__.py
│   ├── time_series.py               # Time series plots
│   ├── geo_heatmap.py               # Geographic heatmaps
│   ├── distribution.py              # Score distributions & dashboards
│   ├── cluster_viz.py               # Cluster visualizations
│   └── interactive_plots.py         # Interactive plotly charts
│
└── examples/                        # Demo Scripts & Outputs
    ├── xai_examples.py              # XAI demonstration
    ├── visualization_demo.py        # Visualization demonstration
    ├── sample_outputs/              # XAI sample JSON outputs
    └── viz_outputs/                 # Generated visualization files
```

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                      │
│                         (main.py)                             │
├─────────────────────────────────────────────────────────────┤
│                        API Layer                              │
│  ┌─────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────────┐  │
│  │Datasets │ │ Selection   │ │ Analysis │ │Visualizations│  │
│  └─────────┘ └─────────────┘ └──────────┘ └──────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     Service Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Data Processing │  │   ML Inference  │  │    XAI       │ │
│  │   (data/)       │  │   (models/)     │  │(explainability)│
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Visualization Layer                         │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────────┐ │
│  │Time Series│ │Geo Heatmap│ │Clusters │ │Risk Dashboards  │ │
│  └──────────┘ └──────────┘ └─────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app initialization, middleware, routers |
| `config.py` | Settings, environment variables, dataset configs |
| `api/schemas.py` | Pydantic models for requests/responses |
| `models/ensemble_detector.py` | Core ML ensemble orchestration |
| `explainability/fraud_explainer.py` | Unified XAI engine |
| `visualization/distribution.py` | Risk dashboard generation |

## Running the Server

```bash
cd ml_backend
pip install -r requirements.txt
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
