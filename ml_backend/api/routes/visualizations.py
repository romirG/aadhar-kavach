"""
Visualization routes - Generate charts and dashboards.
"""
import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

import sys
sys.path.insert(0, '..')

from api.schemas import VisualizationResponse, ChartData
from api.routes.analysis import analysis_jobs, AnalysisStatus
from visualization.time_series import (
    create_time_series_plot,
    create_score_distribution_plot,
    create_feature_importance_plot
)
from visualization.geo_heatmap import create_geo_heatmap, create_cluster_visualization
from visualization.distribution import create_risk_dashboard, create_comparison_plot
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/analysis/{job_id}/visualizations")
async def get_visualizations(job_id: str) -> Dict[str, Any]:
    """
    Generate all visualizations for a completed analysis.
    
    Returns:
        - Time series plot with anomaly overlay
        - Anomaly score distribution
        - Geographic heatmap
        - Feature importance chart
        - Risk dashboard
        - Cluster visualization
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed. Current status: {job['status']}"
        )
    
    logger.info(f"Generating visualizations for job {job_id}")
    
    df = job.get("df")
    X = job.get("X")
    scores = job.get("scores")
    predictions = job.get("predictions")
    feature_names = job.get("feature_names", [])
    feature_importance = job.get("feature_importance", {})
    ensemble = job.get("ensemble")
    
    charts = []
    
    try:
        # 1. Time Series Plot
        logger.info("Creating time series plot...")
        time_series = create_time_series_plot(df, scores)
        if "error" not in time_series:
            charts.append(ChartData(
                chart_type="time_series",
                title="Event Volume and Anomaly Timeline",
                x_label="Date",
                y_label="Value / Anomaly Score",
                data=time_series["data"]
            ))
    except Exception as e:
        logger.warning(f"Error creating time series plot: {e}")
    
    try:
        # 2. Score Distribution
        logger.info("Creating score distribution...")
        distribution = create_score_distribution_plot(scores, predictions)
        if "error" not in distribution:
            charts.append(ChartData(
                chart_type="distribution",
                title="Anomaly Score Distribution",
                x_label="Anomaly Score",
                y_label="Frequency",
                data=distribution["data"]
            ))
    except Exception as e:
        logger.warning(f"Error creating distribution plot: {e}")
    
    try:
        # 3. Feature Importance
        logger.info("Creating feature importance plot...")
        importance_chart = create_feature_importance_plot(feature_importance)
        if "error" not in importance_chart:
            charts.append(ChartData(
                chart_type="feature_importance",
                title="Feature Importance for Anomaly Detection",
                x_label="Importance Score",
                y_label="Feature",
                data=importance_chart["data"]
            ))
    except Exception as e:
        logger.warning(f"Error creating feature importance plot: {e}")
    
    try:
        # 4. Geographic Heatmap
        logger.info("Creating geographic heatmap...")
        geo_chart = create_geo_heatmap(df, scores)
        if "error" not in geo_chart:
            charts.append(ChartData(
                chart_type="geo_heatmap",
                title="Geographic Anomaly Distribution",
                data=geo_chart["data"]
            ))
    except Exception as e:
        logger.warning(f"Error creating geo heatmap: {e}")
    
    try:
        # 5. Risk Dashboard
        logger.info("Creating risk dashboard...")
        model_results = ensemble.get_model_results() if ensemble else []
        dashboard = create_risk_dashboard(scores, predictions, df, model_results)
        if "error" not in dashboard:
            charts.append(ChartData(
                chart_type="risk_dashboard",
                title="UIDAI Fraud Detection Dashboard",
                data=dashboard["data"]
            ))
    except Exception as e:
        logger.warning(f"Error creating risk dashboard: {e}")
    
    try:
        # 6. Cluster Visualization
        if ensemble and "hdbscan" in ensemble.models:
            logger.info("Creating cluster visualization...")
            cluster_labels = ensemble.models["hdbscan"].get_cluster_labels()
            cluster_chart = create_cluster_visualization(df, cluster_labels, scores)
            if "error" not in cluster_chart:
                charts.append(ChartData(
                    chart_type="cluster",
                    title="Cluster Analysis",
                    data=cluster_chart["data"]
                ))
    except Exception as e:
        logger.warning(f"Error creating cluster visualization: {e}")
    
    try:
        # 7. Normal vs Anomaly Comparison
        logger.info("Creating comparison plot...")
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        
        if normal_mask.sum() > 0 and anomaly_mask.sum() > 0:
            comparison = create_comparison_plot(
                X[normal_mask],
                X[anomaly_mask],
                feature_names
            )
            if "error" not in comparison:
                charts.append(ChartData(
                    chart_type="comparison",
                    title="Normal vs Anomalous Records",
                    data=comparison["data"]
                ))
    except Exception as e:
        logger.warning(f"Error creating comparison plot: {e}")
    
    logger.info(f"Generated {len(charts)} visualizations for job {job_id}")
    
    return VisualizationResponse(
        job_id=job_id,
        charts=charts,
        generated_at=datetime.now()
    ).model_dump()


@router.get("/analysis/{job_id}/visualizations/{chart_type}")
async def get_specific_visualization(job_id: str, chart_type: str) -> Dict[str, Any]:
    """
    Get a specific visualization by type.
    
    Available chart types:
    - time_series
    - distribution
    - feature_importance
    - geo_heatmap
    - risk_dashboard
    - cluster
    - comparison
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed. Current status: {job['status']}"
        )
    
    df = job.get("df")
    X = job.get("X")
    scores = job.get("scores")
    predictions = job.get("predictions")
    feature_names = job.get("feature_names", [])
    feature_importance = job.get("feature_importance", {})
    ensemble = job.get("ensemble")
    
    if chart_type == "time_series":
        result = create_time_series_plot(df, scores)
    elif chart_type == "distribution":
        result = create_score_distribution_plot(scores, predictions)
    elif chart_type == "feature_importance":
        result = create_feature_importance_plot(feature_importance)
    elif chart_type == "geo_heatmap":
        result = create_geo_heatmap(df, scores)
    elif chart_type == "risk_dashboard":
        model_results = ensemble.get_model_results() if ensemble else []
        result = create_risk_dashboard(scores, predictions, df, model_results)
    elif chart_type == "cluster":
        if ensemble and "hdbscan" in ensemble.models:
            cluster_labels = ensemble.models["hdbscan"].get_cluster_labels()
            result = create_cluster_visualization(df, cluster_labels, scores)
        else:
            raise HTTPException(status_code=400, detail="Cluster visualization not available (HDBSCAN not used)")
    elif chart_type == "comparison":
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        result = create_comparison_plot(X[normal_mask], X[anomaly_mask], feature_names)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown chart type: {chart_type}. Available: time_series, distribution, feature_importance, geo_heatmap, risk_dashboard, cluster, comparison"
        )
    
    return result


@router.get("/analysis/{job_id}/charts/images/{chart_type}")
async def get_chart_image(job_id: str, chart_type: str) -> Dict[str, str]:
    """
    Get base64 encoded image for a specific chart type.
    
    Returns the image_base64 field from the visualization.
    """
    result = await get_specific_visualization(job_id, chart_type)
    
    if "image_base64" in result:
        return {
            "chart_type": chart_type,
            "image_base64": result["image_base64"],
            "content_type": "image/png"
        }
    else:
        raise HTTPException(status_code=500, detail="Image not available for this chart type")
