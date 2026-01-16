"""
Report generation API endpoints.
"""
import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, '..')

from api.routes.analysis import analysis_jobs, AnalysisStatus
from reporting.fraud_report_generator import FraudReportGenerator

logger = logging.getLogger(__name__)
router = APIRouter()


class ReportRequest(BaseModel):
    """Request for report generation."""
    include_policy_analysis: bool = Field(default=True, description="Include policy impact analysis")
    include_societal_notes: bool = Field(default=True, description="Include societal considerations")
    top_suspicious_count: int = Field(default=10, ge=1, le=50, description="Number of top suspicious entities")


class ReportResponse(BaseModel):
    """Response containing the report."""
    report_id: str
    generated_at: datetime
    executive_summary: dict
    anomaly_statistics: dict
    suspicious_entities: dict
    temporal_trends: list
    system_confidence: dict
    policy_impact: Optional[dict] = None
    human_readable_report: str


@router.post("/analysis/{job_id}/report", response_model=ReportResponse)
async def generate_fraud_report(job_id: str, request: ReportRequest = ReportRequest()):
    """
    Generate comprehensive fraud report for completed analysis.
    
    Returns:
        - Executive summary with key findings
        - Anomaly statistics and percentages
        - Top suspicious regions, centers, operators
        - Temporal fraud trends
        - System confidence and limitations
        - Policy and societal impact notes
        - Human-readable report text
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not yet completed")
    
    # Get stored data
    df = job.get("df")
    scores = job.get("scores")
    predictions = job.get("predictions")
    feature_names = job.get("feature_names", [])
    ensemble = job.get("ensemble")
    
    if df is None or scores is None:
        raise HTTPException(status_code=500, detail="Analysis data not available")
    
    # Get model results
    model_results = []
    if ensemble:
        for result in ensemble.get_model_results():
            model_results.append({
                'model_name': result.get('model_name'),
                'anomaly_count': result.get('anomaly_count', 0)
            })
    
    # Generate report
    generator = FraudReportGenerator()
    report = generator.generate_report(
        df=df,
        anomaly_scores=scores,
        predictions=predictions,
        feature_names=feature_names,
        model_results=model_results,
        dataset_name=job.get("dataset_id", "UIDAI Dataset")
    )
    
    # Build response
    response = ReportResponse(
        report_id=report['report_metadata']['report_id'],
        generated_at=datetime.now(),
        executive_summary=report['executive_summary'],
        anomaly_statistics=report['anomaly_statistics'],
        suspicious_entities=report['suspicious_entities'],
        temporal_trends=report['temporal_trends'],
        system_confidence=report['system_confidence'],
        policy_impact=report['policy_impact'] if request.include_policy_analysis else None,
        human_readable_report=report['human_readable_report']
    )
    
    logger.info(f"Generated fraud report for job {job_id}")
    return response


@router.get("/analysis/{job_id}/report/json")
async def get_json_report(job_id: str):
    """
    Get complete fraud report as JSON.
    
    Returns the full report structure suitable for programmatic processing.
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not yet completed")
    
    df = job.get("df")
    scores = job.get("scores")
    predictions = job.get("predictions")
    feature_names = job.get("feature_names", [])
    
    if df is None or scores is None:
        raise HTTPException(status_code=500, detail="Analysis data not available")
    
    generator = FraudReportGenerator()
    report = generator.generate_report(
        df=df,
        anomaly_scores=scores,
        predictions=predictions,
        feature_names=feature_names,
        dataset_name=job.get("dataset_id", "UIDAI Dataset")
    )
    
    # Remove human readable to reduce size
    del report['human_readable_report']
    
    return report


@router.get("/analysis/{job_id}/report/text")
async def get_text_report(job_id: str):
    """
    Get human-readable text report.
    
    Returns the report as formatted plain text suitable for printing/sharing.
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not yet completed")
    
    df = job.get("df")
    scores = job.get("scores")
    predictions = job.get("predictions")
    feature_names = job.get("feature_names", [])
    
    if df is None or scores is None:
        raise HTTPException(status_code=500, detail="Analysis data not available")
    
    generator = FraudReportGenerator()
    report = generator.generate_report(
        df=df,
        anomaly_scores=scores,
        predictions=predictions,
        feature_names=feature_names,
        dataset_name=job.get("dataset_id", "UIDAI Dataset")
    )
    
    return {
        "report_id": report['report_metadata']['report_id'],
        "content_type": "text/plain",
        "report": report['human_readable_report']
    }
