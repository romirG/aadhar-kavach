"""
UIDAI ML Fraud Detection Backend - FastAPI Application
Main entry point for the ML backend server.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from api.routes import datasets, analysis, visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    logger.info("🚀 UIDAI ML Fraud Detection Backend starting...")
    settings = get_settings()
    logger.info(f"📊 Configured for {len(datasets.get_available_datasets())} datasets")
    yield
    # Shutdown
    logger.info("👋 UIDAI ML Backend shutting down...")


# Create FastAPI application
app = FastAPI(
    title="UIDAI ML Fraud Detection API",
    description="""
    AI-powered fraud detection system for Aadhaar enrolment and update data.
    
    ## Features
    - **Dataset Selection**: Choose from 3 official data.gov.in APIs
    - **ML Analysis**: Automatic model selection and ensemble scoring
    - **Explainability**: Human-readable fraud reasons
    - **Visualizations**: Interactive charts and heatmaps
    
    ## Models Used
    - Isolation Forest (baseline anomaly detection)
    - PyTorch Autoencoder (deep pattern recognition)
    - HDBSCAN (spatial clustering)
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Check if the ML backend is running."""
    return {
        "status": "healthy",
        "service": "UIDAI ML Fraud Detection Backend",
        "version": "1.0.0"
    }


# Include routers
app.include_router(datasets.router, prefix="/api/ml", tags=["Datasets"])
app.include_router(analysis.router, prefix="/api/ml", tags=["Analysis"])
app.include_router(visualizations.router, prefix="/api/ml", tags=["Visualizations"])


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
