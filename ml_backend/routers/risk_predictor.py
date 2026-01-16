"""
Biometric Re-enrollment Risk Predictor API Router

Provides endpoints for running biometric risk analysis and retrieving results.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from services.biometric_risk_service import biometric_risk_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/risk", tags=["Biometric Risk Predictor"])


# ==================== Request/Response Models ====================

class AnalysisRequest(BaseModel):
    """Request model for running risk analysis"""
    state: Optional[str] = Field(None, description="Filter by state name")
    district: Optional[str] = Field(None, description="Filter by district name")
    age_range: str = Field("all", description="Age range filter: 'elderly', 'adult', or 'all'")
    risk_threshold: float = Field(0.5, ge=0.3, le=0.9, description="Risk threshold for classification")
    records_limit: int = Field(5000, ge=1000, le=10000, description="Maximum records to analyze")


class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    success: bool
    timestamp: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    model_metrics: Optional[Dict[str, Any]] = None
    high_risk_regions: Optional[List[Dict[str, Any]]] = None
    cluster_analysis: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    visualizations: Optional[Dict[str, str]] = None
    feature_importance: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    model_loaded: bool


# ==================== Endpoints ====================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the risk prediction service is healthy"""
    return HealthResponse(
        status="healthy",
        service="Biometric Re-enrollment Risk Predictor",
        model_loaded=biometric_risk_service.trained_model is not None
    )


@router.post("/analyze", response_model=AnalysisResponse)
async def run_analysis(request: AnalysisRequest):
    """
    Run comprehensive biometric re-enrollment risk analysis.
    
    This endpoint:
    1. Fetches data from government APIs (enrolment, demographic, biometric)
    2. Engineers features for risk prediction
    3. Trains ML model (XGBoost/Random Forest)
    4. Identifies high-risk regions
    5. Performs cluster analysis
    6. Generates visualizations
    7. Provides actionable recommendations
    
    Returns complete analysis results including charts as base64 images.
    """
    logger.info(f"Starting risk analysis: state={request.state}, age={request.age_range}")
    
    try:
        results = await biometric_risk_service.run_full_analysis(
            state_filter=request.state,
            district_filter=request.district,
            age_range=request.age_range,
            risk_threshold=request.risk_threshold,
            records_limit=request.records_limit
        )
        
        return AnalysisResponse(**results)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/high-risk-regions")
async def get_high_risk_regions(
    threshold: float = Query(0.5, ge=0.3, le=0.9, description="Risk threshold"),
    limit: int = Query(20, ge=1, le=50, description="Max regions to return")
):
    """
    Get list of high-risk regions from the last analysis.
    
    Returns regions sorted by risk score with contributing factors.
    """
    if not biometric_risk_service.analysis_results:
        raise HTTPException(
            status_code=404, 
            detail="No analysis results available. Run /analyze first."
        )
    
    high_risk = biometric_risk_service.analysis_results.get('high_risk_regions', [])
    
    # Filter by threshold
    filtered = [r for r in high_risk if r.get('risk_score', 0) >= threshold]
    
    return {
        "success": True,
        "threshold": threshold,
        "total_high_risk": len(filtered),
        "regions": filtered[:limit]
    }


@router.get("/recommendations")
async def get_recommendations():
    """
    Get actionable recommendations from the last analysis.
    
    Returns prioritized recommendations for reducing biometric failure risk.
    """
    if not biometric_risk_service.analysis_results:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Run /analyze first."
        )
    
    recommendations = biometric_risk_service.analysis_results.get('recommendations', [])
    
    return {
        "success": True,
        "total_recommendations": len(recommendations),
        "recommendations": recommendations
    }


@router.get("/visualizations")
async def get_visualizations():
    """
    Get visualization charts from the last analysis.
    
    Returns charts as base64-encoded PNG images.
    """
    if not biometric_risk_service.analysis_results:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Run /analyze first."
        )
    
    visualizations = biometric_risk_service.analysis_results.get('visualizations', {})
    
    return {
        "success": True,
        "charts_available": list(visualizations.keys()),
        "visualizations": visualizations
    }


@router.get("/model-info")
async def get_model_info():
    """
    Get information about the trained ML model.
    
    Returns model type, accuracy, and feature importance.
    """
    if not biometric_risk_service.trained_model:
        return {
            "success": True,
            "model_loaded": False,
            "message": "No model trained yet. Run /analyze first."
        }
    
    model_metrics = biometric_risk_service.analysis_results.get('model_metrics', {})
    feature_importance = biometric_risk_service.analysis_results.get('feature_importance', {})
    
    return {
        "success": True,
        "model_loaded": True,
        "model_type": biometric_risk_service.model_type,
        "accuracy": model_metrics.get('accuracy', 0),
        "feature_importance": dict(list(feature_importance.items())[:10])  # Top 10
    }


@router.get("/summary")
async def get_analysis_summary():
    """
    Get summary of the last analysis.
    
    Returns key statistics and overview metrics.
    """
    if not biometric_risk_service.analysis_results:
        return {
            "success": True,
            "analysis_available": False,
            "message": "No analysis results available. Run /analyze first."
        }
    
    results = biometric_risk_service.analysis_results
    
    return {
        "success": True,
        "analysis_available": True,
        "timestamp": results.get('timestamp'),
        "parameters": results.get('parameters'),
        "summary": results.get('summary'),
        "high_risk_count": len(results.get('high_risk_regions', [])),
        "recommendations_count": len(results.get('recommendations', []))
    }


@router.get("/states")
async def get_available_states():
    """
    Get list of states available in the data.
    
    Used for populating the state filter dropdown.
    """
    from services.data_gov_client import data_client
    
    try:
        # Fetch a sample to get unique states
        result = data_client.fetch_data('enrolment', limit=1000)
        
        if result.get('success') and result.get('records'):
            import pandas as pd
            df = pd.DataFrame(result['records'])
            states = sorted(df['state'].unique().tolist()) if 'state' in df.columns else []
            
            return {
                "success": True,
                "states": states,
                "total": len(states)
            }
        
        return {
            "success": False,
            "error": "Could not fetch states",
            "states": []
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch states: {e}")
        return {
            "success": False,
            "error": str(e),
            "states": []
        }
