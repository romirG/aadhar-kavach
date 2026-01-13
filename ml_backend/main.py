"""
Biometric Re-enrollment Risk Predictor - FastAPI Application

UIDAI Data Hackathon Project
Predicts Aadhaar biometric authentication failure risk using government data.

PRIVACY SAFEGUARDS:
- Uses ONLY aggregated data from public government APIs
- No individual Aadhaar numbers processed
- All operations on state/district/age-group level
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from routers import datasets, analysis, visualizations
from config import OUTPUT_DIR, VISUALIZATION_DIR, MODEL_DIR, REPORT_DIR

# Create output directories
for directory in [OUTPUT_DIR, VISUALIZATION_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(directory, exist_ok=True)

app = FastAPI(
    title="Biometric Re-enrollment Risk Predictor",
    description="""
    ML-powered system to predict Aadhaar biometric authentication failure risk.
    
    **Features:**
    - Dataset selection and fusion from data.gov.in
    - Automated feature engineering
    - Auto model selection (supervised/unsupervised)
    - Risk scoring and categorization
    - Policy-ready visualizations
    - SHAP-based explainability
    
    **Privacy:** Uses only aggregated, anonymized government data.
    """,
    version="1.0.0",
    contact={
        "name": "UIDAI Hackathon Team"
    }
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount static files for visualizations
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# Include routers
app.include_router(datasets.router, prefix="/api", tags=["Datasets"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(visualizations.router, prefix="/api", tags=["Visualizations"])

@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "status": "healthy",
        "project": "Biometric Re-enrollment Risk Predictor",
        "version": "1.0.0",
        "endpoints": {
            "datasets": "/api/datasets",
            "select_dataset": "/api/select-dataset",
            "analyze": "/api/analyze",
            "train_model": "/api/train-model",
            "risk_summary": "/api/risk-summary",
            "visualizations": "/api/visualizations",
            "explain_model": "/api/explain-model"
        }
    }

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "ok"}
