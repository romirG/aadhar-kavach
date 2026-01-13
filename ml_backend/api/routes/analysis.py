"""
Analysis routes - Trigger and manage ML analysis jobs.
"""
import logging
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '..')

from config import DATASETS, get_settings
from api.schemas import (
    AnalysisRequest, AnalysisJobResponse, AnalysisResults,
    AnalysisStatusResponse, AnalysisStatus, AnomalyRecord,
    RiskLevel, ModelResult, AuditorSummary, FraudPattern,
    HighRiskEntity, TemporalTrend
)
from data.ingestion import get_data_ingestion
from data.preprocessing import DataPreprocessor
from data.feature_engineering import FeatureEngineer
from models.ensemble import EnsembleScorer
from explainability.feature_attribution import FeatureAttribution, ReasonGenerator

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job storage (use Redis/DB in production)
analysis_jobs: Dict[str, Dict[str, Any]] = {}


def get_risk_level(score: float) -> RiskLevel:
    """Determine risk level from anomaly score."""
    settings = get_settings()
    if score >= settings.high_risk_threshold:
        return RiskLevel.HIGH
    elif score >= settings.medium_risk_threshold:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


async def run_analysis(job_id: str, dataset_id: str, limit: int):
    """
    Run the full analysis pipeline in background.
    
    This is the core ML pipeline that:
    1. Fetches data from data.gov.in
    2. Preprocesses and engineers features
    3. Runs ensemble anomaly detection
    4. Generates explanations
    """
    start_time = time.time()
    
    try:
        # Update status
        analysis_jobs[job_id]["status"] = AnalysisStatus.PROCESSING
        analysis_jobs[job_id]["started_at"] = datetime.now()
        
        logger.info(f"Starting analysis job {job_id} for dataset {dataset_id}")
        
        # Step 1: Data Ingestion
        logger.info("Step 1: Fetching data...")
        ingestion = get_data_ingestion()
        df = await ingestion.fetch_all_pages(dataset_id, max_records=limit)
        
        if df.empty:
            raise ValueError("No data fetched from API")
        
        analysis_jobs[job_id]["progress"] = 20
        logger.info(f"Fetched {len(df)} records")
        
        # Step 2: Feature Engineering
        logger.info("Step 2: Engineering features...")
        feature_engineer = FeatureEngineer()
        df_engineered = feature_engineer.engineer_features(df, dataset_id)
        
        analysis_jobs[job_id]["progress"] = 35
        
        # Step 3: Preprocessing
        logger.info("Step 3: Preprocessing data...")
        preprocessor = DataPreprocessor()
        X, feature_names = preprocessor.fit_transform(df_engineered)
        
        if X.size == 0:
            raise ValueError("No features after preprocessing")
        
        analysis_jobs[job_id]["progress"] = 50
        logger.info(f"Processed {X.shape[0]} samples with {X.shape[1]} features")
        
        # Step 4: ML Ensemble
        logger.info("Step 4: Running ML ensemble...")
        ensemble = EnsembleScorer()
        predictions, scores = ensemble.fit_predict(X, feature_names)
        
        analysis_jobs[job_id]["progress"] = 75
        
        # Step 5: Explainability
        logger.info("Step 5: Generating explanations...")
        attribution = FeatureAttribution()
        attribution.fit(X, feature_names)
        
        reason_generator = ReasonGenerator()
        
        # Get feature importance
        feature_importance = ensemble.get_feature_importance(X)
        
        analysis_jobs[job_id]["progress"] = 90
        
        # Build anomaly records
        anomaly_records = []
        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]
        
        for idx in anomaly_indices[:100]:  # Limit to top 100
            contributors = attribution.get_top_contributors(X[idx:idx+1], top_k=5)
            reasons = reason_generator.generate_reasons(
                contributors,
                float(scores[idx]),
                ensemble.get_model_results()
            )
            
            record = AnomalyRecord(
                record_id=str(idx),
                anomaly_score=float(scores[idx]),
                risk_level=get_risk_level(scores[idx]),
                confidence=float(min(0.95, scores[idx] + 0.1)),
                reasons=reasons,
                features={name: float(X[idx, i]) for i, name in enumerate(feature_names[:10])}
            )
            anomaly_records.append(record)
        
        # Sort by score descending
        anomaly_records.sort(key=lambda x: x.anomaly_score, reverse=True)
        
        # Build model results
        model_results = []
        for result in ensemble.get_model_results():
            model_results.append(ModelResult(
                model_name=result["model_name"],
                anomaly_scores=list(ensemble.model_results[result["model_name"]].scores[:100]),
                threshold=result["threshold"],
                anomaly_count=result["anomaly_count"],
                execution_time_ms=result["execution_time_ms"]
            ))
        
        execution_time = (time.time() - start_time) * 1000
        
        # Build final results
        results = AnalysisResults(
            job_id=job_id,
            dataset_id=dataset_id,
            status=AnalysisStatus.COMPLETED,
            total_records=len(df),
            anomaly_count=int(anomaly_mask.sum()),
            anomaly_percentage=float(100 * anomaly_mask.sum() / len(df)),
            models_used=list(ensemble.models.keys()),
            model_results=model_results,
            anomalies=anomaly_records,
            execution_time_ms=execution_time,
            created_at=analysis_jobs[job_id]["created_at"],
            completed_at=datetime.now()
        )
        
        # Store results
        analysis_jobs[job_id]["status"] = AnalysisStatus.COMPLETED
        analysis_jobs[job_id]["progress"] = 100
        analysis_jobs[job_id]["results"] = results
        analysis_jobs[job_id]["df"] = df_engineered
        analysis_jobs[job_id]["X"] = X
        analysis_jobs[job_id]["feature_names"] = feature_names
        analysis_jobs[job_id]["predictions"] = predictions
        analysis_jobs[job_id]["scores"] = scores
        analysis_jobs[job_id]["ensemble"] = ensemble
        analysis_jobs[job_id]["feature_importance"] = feature_importance
        
        logger.info(f"Analysis job {job_id} completed in {execution_time:.2f}ms")
        
    except Exception as e:
        logger.error(f"Analysis job {job_id} failed: {e}")
        analysis_jobs[job_id]["status"] = AnalysisStatus.FAILED
        analysis_jobs[job_id]["error"] = str(e)


@router.post("/datasets/{dataset_id}/analyze", response_model=AnalysisJobResponse)
async def start_analysis(
    dataset_id: str,
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new analysis job for the selected dataset.
    
    This triggers the ML pipeline in the background and returns a job ID
    for tracking progress.
    """
    if dataset_id not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found. Available: {list(DATASETS.keys())}"
        )
    
    # Create job
    job_id = str(uuid.uuid4())
    analysis_jobs[job_id] = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "status": AnalysisStatus.PENDING,
        "progress": 0,
        "created_at": datetime.now(),
        "started_at": None,
        "results": None,
        "error": None
    }
    
    # Start background analysis
    background_tasks.add_task(run_analysis, job_id, dataset_id, request.limit)
    
    logger.info(f"Created analysis job {job_id} for dataset {dataset_id}")
    
    return AnalysisJobResponse(
        job_id=job_id,
        dataset_id=dataset_id,
        status=AnalysisStatus.PENDING,
        created_at=analysis_jobs[job_id]["created_at"],
        message=f"Analysis started for {DATASETS[dataset_id]['name']}. Use job_id to check status."
    )


@router.get("/analysis/{job_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(job_id: str):
    """
    Check the status of an analysis job.
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = analysis_jobs[job_id]
    
    message = {
        AnalysisStatus.PENDING: "Job is queued and waiting to start",
        AnalysisStatus.PROCESSING: f"Analysis in progress ({job.get('progress', 0)}% complete)",
        AnalysisStatus.COMPLETED: "Analysis completed successfully",
        AnalysisStatus.FAILED: f"Analysis failed: {job.get('error', 'Unknown error')}"
    }.get(job["status"], "Unknown status")
    
    return AnalysisStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=float(job.get("progress", 0)),
        message=message,
        started_at=job.get("started_at")
    )


@router.get("/analysis/{job_id}/results", response_model=AnalysisResults)
async def get_analysis_results(job_id: str):
    """
    Get the results of a completed analysis job.
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] == AnalysisStatus.PENDING:
        raise HTTPException(status_code=202, detail="Analysis not yet started")
    
    if job["status"] == AnalysisStatus.PROCESSING:
        raise HTTPException(status_code=202, detail=f"Analysis in progress ({job.get('progress', 0)}% complete)")
    
    if job["status"] == AnalysisStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {job.get('error', 'Unknown error')}")
    
    return job["results"]


@router.get("/analysis/{job_id}/summary", response_model=AuditorSummary)
async def get_auditor_summary(job_id: str):
    """
    Get auditor-friendly summary of the analysis.
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not yet completed")
    
    results = job["results"]
    df = job.get("df", pd.DataFrame())
    scores = job.get("scores", np.array([]))
    predictions = job.get("predictions", np.array([]))
    feature_importance = job.get("feature_importance", {})
    
    # Calculate risk distribution
    risk_distribution = {
        "High": int((scores > 0.8).sum()),
        "Medium": int(((scores > 0.5) & (scores <= 0.8)).sum()),
        "Low": int((scores <= 0.5).sum())
    }
    
    # Generate fraud patterns
    fraud_patterns = []
    reason_generator = ReasonGenerator()
    
    top_features = list(feature_importance.keys())[:5]
    for feature in top_features:
        pattern = FraudPattern(
            pattern_type=reason_generator.generate_fraud_pattern([{"feature": feature}]),
            description=f"Significant anomalies detected in {feature.replace('_', ' ')}",
            affected_records=int(results.anomaly_count * feature_importance.get(feature, 0.2)),
            severity=RiskLevel.MEDIUM if feature_importance.get(feature, 0) > 0.3 else RiskLevel.LOW,
            examples=[f"Records with unusual {feature} values"]
        )
        fraud_patterns.append(pattern)
    
    # High risk entities
    high_risk_entities = []
    if 'state' in df.columns and len(scores) == len(df):
        df_temp = df.copy()
        df_temp['score'] = scores
        state_stats = df_temp.groupby('state').agg({'score': ['mean', 'count']}).reset_index()
        state_stats.columns = ['state', 'mean_score', 'count']
        high_risk_states = state_stats[state_stats['mean_score'] > 0.5].nlargest(5, 'mean_score')
        
        for _, row in high_risk_states.iterrows():
            entity = HighRiskEntity(
                entity_type="State",
                entity_id=row['state'],
                entity_name=row['state'],
                anomaly_count=int(row['count'] * row['mean_score']),
                risk_score=float(row['mean_score'])
            )
            high_risk_entities.append(entity)
    
    # Temporal trends
    temporal_trends = []
    if 'month' in df.columns and len(scores) == len(df):
        df_temp = df.copy()
        df_temp['score'] = scores
        df_temp['is_anomaly'] = predictions == -1
        
        monthly = df_temp.groupby('month').agg({
            'score': 'mean',
            'is_anomaly': 'sum'
        }).reset_index()
        
        for _, row in monthly.iterrows():
            trend = TemporalTrend(
                period=f"Month {int(row['month'])}",
                normal_avg=float(1 - row['score']),
                anomaly_avg=float(row['score']),
                spike_detected=row['is_anomaly'] > monthly['is_anomaly'].mean() * 1.5,
                spike_magnitude=float(row['is_anomaly'] / max(1, monthly['is_anomaly'].mean()))
            )
            temporal_trends.append(trend)
    
    # Recommendations
    recommendations = [
        f"Focus investigation on {risk_distribution['High']} high-risk records with scores above 0.8",
        f"Review {len(high_risk_entities)} high-risk geographic areas identified",
        f"Investigate unusual patterns in: {', '.join(top_features[:3])}",
        "Consider additional verification for records flagged by multiple models",
        "Set up monitoring for temporal spikes in anomaly rates"
    ]
    
    summary = AuditorSummary(
        job_id=job_id,
        generated_at=datetime.now(),
        total_records_analyzed=results.total_records,
        total_anomalies=results.anomaly_count,
        anomaly_percentage=results.anomaly_percentage,
        risk_distribution=risk_distribution,
        top_fraud_patterns=fraud_patterns[:5],
        high_risk_entities=high_risk_entities[:10],
        temporal_trends=temporal_trends,
        recommendations=recommendations
    )
    
    return summary
