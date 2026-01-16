"""
Biometric Re-enrollment Risk Predictor API Router

Provides endpoints for running biometric risk analysis and retrieving results.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# ==================== NEW ENDPOINTS (ChatGPT Specifications) ====================

@router.get("/survival-curve")
async def get_survival_curve():
    """
    Get survival curve data for template aging analysis.
    
    Returns Kaplan-Meier style time-to-failure data with:
    - Overall population survival curve
    - Risk-stratified curves (low/high risk)
    - Key statistics (median survival, 1-year/3-year rates)
    """
    if not biometric_risk_service.analysis_results:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Run /analyze first."
        )
    
    survival_data = biometric_risk_service.analysis_results.get('survival_data', {})
    
    return {
        "success": True,
        "survival_data": survival_data
    }


@router.get("/shap-explanation")
async def get_shap_explanations():
    """
    Get SHAP-based model explanations.
    
    Returns:
    - Global feature importance (mean absolute SHAP values)
    - Local explanations (top-3 factors for high-risk regions)
    """
    if not biometric_risk_service.analysis_results:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Run /analyze first."
        )
    
    shap_analysis = biometric_risk_service.analysis_results.get('shap_analysis', {})
    
    return {
        "success": True,
        "shap_available": shap_analysis.get('available', False),
        "message": shap_analysis.get('message', ''),
        "global_importance": shap_analysis.get('global_importance', {}),
        "local_explanations": shap_analysis.get('local_explanations', [])
    }


@router.get("/centre-performance")
async def get_centre_performance():
    """
    Get capture centre performance metrics.
    
    Returns per-centre/state metrics including:
    - Quality score
    - Biometric update ratio
    - High-risk count
    - Performance grade (A-D)
    """
    if not biometric_risk_service.analysis_results:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Run /analyze first."
        )
    
    centre_performance = biometric_risk_service.analysis_results.get('centre_performance', [])
    
    return {
        "success": True,
        "total_centres": len(centre_performance),
        "centres": centre_performance
    }


@router.get("/age-bucket-analysis")
async def get_age_bucket_analysis():
    """
    Get risk analysis by 5 age buckets.
    
    Buckets: 0-17, 18-34, 35-49, 50-64, 65+
    Returns risk estimates and recommendations per bucket.
    """
    if not biometric_risk_service.analysis_results:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Run /analyze first."
        )
    
    age_bucket_analysis = biometric_risk_service.analysis_results.get('age_bucket_analysis', {})
    
    return {
        "success": True,
        "age_bucket_analysis": age_bucket_analysis
    }


@router.get("/threshold-sensitivity")
async def get_threshold_sensitivity(
    thresholds: str = Query("0.3,0.5,0.6,0.7,0.8", description="Comma-separated thresholds to evaluate")
):
    """
    Get operational impact at different risk thresholds.
    
    Shows how many residents would be flagged at each threshold level.
    Helps operations team pick optimal threshold balancing cost vs impact.
    """
    if not biometric_risk_service.features_df is None and biometric_risk_service.features_df.empty:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Run /analyze first."
        )
    
    try:
        threshold_list = [float(t.strip()) for t in thresholds.split(',')]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid threshold format")
    
    df = biometric_risk_service.features_df
    if df is None or 'risk_score' not in df.columns:
        return {
            "success": False,
            "error": "No risk scores available. Run analysis first."
        }
    
    total_regions = len(df)
    sensitivity = []
    
    for threshold in threshold_list:
        flagged = int((df['risk_score'] >= threshold).sum())
        sensitivity.append({
            "threshold": threshold,
            "flagged_count": flagged,
            "flagged_percentage": round(flagged / total_regions * 100, 1) if total_regions > 0 else 0,
            "action_level": "Critical" if threshold >= 0.8 else "High" if threshold >= 0.6 else "Medium" if threshold >= 0.5 else "Low"
        })
    
    return {
        "success": True,
        "total_regions": total_regions,
        "sensitivity": sensitivity
    }


@router.get("/export-csv")
async def export_high_risk_csv(
    threshold: float = Query(0.5, ge=0.3, le=0.9, description="Risk threshold for export"),
    limit: int = Query(100, ge=10, le=1000, description="Max records to export")
):
    """
    Export high-risk regions as CSV with masked IDs.
    
    PRIVACY: Uses masked/hashed identifiers only. No raw Aadhaar numbers.
    """
    import hashlib
    from fastapi.responses import StreamingResponse
    import io
    
    if biometric_risk_service.features_df is None:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Run /analyze first."
        )
    
    df = biometric_risk_service.features_df
    
    if 'risk_score' not in df.columns:
        raise HTTPException(status_code=400, detail="Risk scores not available")
    
    # Filter high-risk and sort
    high_risk = df[df['risk_score'] >= threshold].nlargest(limit, 'risk_score').copy()
    
    # Create masked IDs
    def mask_id(row_idx, state):
        raw = f"{row_idx}_{state}_{hash(str(row_idx))}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12].upper()
    
    export_data = []
    for idx, row in high_risk.iterrows():
        state = row.get('state', 'Unknown')
        export_data.append({
            'id_mask': mask_id(idx, state),
            'state': state,
            'risk_score': round(row.get('risk_score', 0), 3),
            'risk_bucket': 'Critical' if row.get('risk_score', 0) >= 0.75 else 'High' if row.get('risk_score', 0) >= 0.6 else 'Moderate',
            'elderly_ratio': round(row.get('elderly_ratio', 0), 3),
            'biometric_update_ratio': round(row.get('biometric_update_ratio', 0), 3),
            'centre_quality_score': round(row.get('centre_quality_score', 0), 3),
            'time_since_update_days': int(row.get('time_since_update_days', 0)),
            'recommended_action': 'Immediate Outreach' if row.get('risk_score', 0) >= 0.75 else 'Priority Scheduling' if row.get('risk_score', 0) >= 0.6 else 'Monitor'
        })
    
    # Create CSV
    import pandas as pd
    export_df = pd.DataFrame(export_data)
    
    output = io.StringIO()
    export_df.to_csv(output, index=False)
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=high_risk_regions_threshold_{threshold}.csv"}
    )


# ==================== AI-POWERED ANALYSIS ENDPOINTS (Groq) ====================

@router.get("/ai/deep-analysis")
async def get_ai_deep_analysis():
    """
    Get AI-powered deep analysis of the risk assessment data.
    
    Uses Groq's Llama 3.3 70B to provide:
    - Key findings and insights
    - Root cause analysis
    - Trend patterns
    - Risk factor ranking
    - Operational gaps identification
    """
    if not biometric_risk_service.analysis_results:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Run /analyze first."
        )
    
    try:
        from services.groq_ai_service import generate_deep_analysis
        
        result = generate_deep_analysis(biometric_risk_service.analysis_results)
        
        return {
            "success": result.get("success", False),
            "analysis": result.get("analysis"),
            "model": result.get("model", "llama-3.3-70b-versatile"),
            "type": "deep_analysis",
            "error": result.get("error")
        }
        
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"AI service not available: {str(e)}")
    except Exception as e:
        logger.error(f"AI deep analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/policy-recommendations")
async def get_ai_policy_recommendations():
    """
    Get AI-powered policy change recommendations.
    
    Uses Groq's Llama 3.3 70B to provide:
    - Immediate actions (0-30 days)
    - Short-term policies (1-6 months)
    - Long-term strategic changes
    - Budget allocation priorities
    - Monitoring & evaluation metrics
    """
    if not biometric_risk_service.analysis_results:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Run /analyze first."
        )
    
    try:
        from services.groq_ai_service import generate_policy_recommendations
        
        result = generate_policy_recommendations(biometric_risk_service.analysis_results)
        
        return {
            "success": result.get("success", False),
            "recommendations": result.get("recommendations"),
            "model": result.get("model", "llama-3.3-70b-versatile"),
            "type": "policy_recommendations",
            "error": result.get("error")
        }
        
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"AI service not available: {str(e)}")
    except Exception as e:
        logger.error(f"AI policy recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/region-analysis/{region_name}")
async def get_ai_region_analysis(region_name: str):
    """
    Get AI-powered analysis for a specific region/state.
    
    Provides targeted recommendations and risk assessment
    for the specified region.
    """
    if not biometric_risk_service.analysis_results:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Run /analyze first."
        )
    
    # Find the region data
    high_risk_regions = biometric_risk_service.analysis_results.get('high_risk_regions', [])
    region_data = None
    
    for region in high_risk_regions:
        if region.get('state', '').lower() == region_name.lower():
            region_data = region
            break
    
    if not region_data:
        # Try to find in features_df
        if biometric_risk_service.features_df is not None:
            df = biometric_risk_service.features_df
            if 'state' in df.columns:
                state_data = df[df['state'].str.lower() == region_name.lower()]
                if len(state_data) > 0:
                    row = state_data.iloc[0]
                    region_data = {
                        "state": region_name,
                        "risk_score": float(row.get('risk_score', 0)),
                        "elderly_ratio": float(row.get('elderly_ratio', 0)),
                        "biometric_update_ratio": float(row.get('biometric_update_ratio', 0)),
                        "centre_quality_score": float(row.get('centre_quality_score', 0))
                    }
    
    if not region_data:
        raise HTTPException(
            status_code=404,
            detail=f"Region '{region_name}' not found in analysis results"
        )
    
    try:
        from services.groq_ai_service import generate_region_specific_analysis
        
        result = generate_region_specific_analysis(region_name, region_data)
        
        return {
            "success": result.get("success", False),
            "region": region_name,
            "analysis": result.get("analysis"),
            "model": result.get("model", "llama-3.3-70b-versatile"),
            "error": result.get("error")
        }
        
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"AI service not available: {str(e)}")
    except Exception as e:
        logger.error(f"AI region analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
