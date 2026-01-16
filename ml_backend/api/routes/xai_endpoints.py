
# New API endpoints for XAI explanations - append to analysis.py

@router.post("/analysis/{job_id}/explain")
async def explain_records(job_id: str, record_indices: List[int] = None):
    """
    Generate detailed fraud explanations for specific records.
    
    Args:
        job_id: Analysis job ID
        record_indices: Optional list of record indices to explain (default: top 10 anomalies)
    
    Returns:
        Detailed fraud explanations with reasons, deviations, and recommendations
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not yet completed")
    
    # Get stored data
    X = job.get("X")
    df = job.get("df")
    scores = job.get("scores")
    predictions = job.get("predictions")
    feature_names = job.get("feature_names")
    ensemble = job.get("ensemble")
    
    if X is None or df is None:
        raise HTTPException(status_code=500, detail="Analysis data not available")
    
    # Default to top 10 anomalies if no indices specified
    if record_indices is None:
        anomaly_indices = np.where(predictions == -1)[0]
        top_indices = anomaly_indices[np.argsort(scores[anomaly_indices])[-10:]][::-1]
        record_indices = top_indices.tolist()
    
    # Initialize fraud explainer
    explainer = FraudExplainer()
    explainer.fit(X, df, feature_names)
    
    # Get model scores for each record
    model_scores_batch = []
    if ensemble:
        individual_scores = ensemble.get_model_scores(X)
        for idx in record_indices:
            model_scores_batch.append({
                name: float(scores_arr[idx])
                for name, scores_arr in individual_scores.items()
            })
    else:
        model_scores_batch = [None] * len(record_indices)
    
    # Generate explanations
    explanations = []
    for i, idx in enumerate(record_indices):
        if idx >= len(X):
            continue
        
        sample = X[idx]
        sample_dict = df.iloc[idx].to_dict() if idx < len(df) else {}
        score = float(scores[idx])
        
        # Determine label
        if score >= 0.7:
            label = "Highly Suspicious"
        elif score >= 0.3:
            label = "Suspicious"
        else:
            label = "Normal"
        
        explanation = explainer.explain(
            sample,
            sample_dict,
            score,
            label,
            model_scores_batch[i],
            record_id=f"record_{idx}"
        )
        
        # Convert to dict
        explanations.append({
            "record_id": explanation.record_id,
            "record_index": int(idx),
            "anomaly_score": explanation.anomaly_score,
            "anomaly_label": explanation.anomaly_label,
            "confidence": explanation.confidence,
            "severity": explanation.severity,
            "fraud_pattern": explanation.fraud_pattern,
            "primary_reasons": explanation.primary_reasons,
            "top_features": explanation.top_features,
            "deviations": explanation.deviations,
            "model_contributions": explanation.model_contributions,
            "temporal_flags": explanation.temporal_flags,
            "geographic_flags": explanation.geographic_flags,
            "recommendation": explanation.recommendation,
            "timestamp": explanation.timestamp
        })
    
    # Generate summary
    summary = explainer.generate_summary_report([
        FraudExplanation(**exp) for exp in explanations
    ])
    
    return {
        "job_id": job_id,
        "explanations": explanations,
        "summary": summary,
        "generated_at": datetime.now().isoformat()
    }


@router.get("/analysis/{job_id}/fraud-patterns")
async def get_fraud_patterns(job_id: str):
    """
    Get summary of common fraud patterns detected in the analysis.
    
    Args:
        job_id: Analysis job ID
    
    Returns:
        Summary of fraud patterns, top features, and severity distribution
    """
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not yet completed")
    
    # Get stored data
    X = job.get("X")
    df = job.get("df")
    scores = job.get("scores")
    predictions = job.get("predictions")
    feature_names = job.get("feature_names")
    ensemble = job.get("ensemble")
    
    if X is None or df is None:
        raise HTTPException(status_code=500, detail="Analysis data not available")
    
    # Get all anomalies
    anomaly_indices = np.where(predictions == -1)[0]
    
    # Initialize fraud explainer
    explainer = FraudExplainer()
    explainer.fit(X, df, feature_names)
    
    # Get model scores
    model_scores_batch = []
    if ensemble:
        individual_scores = ensemble.get_model_scores(X)
        for idx in anomaly_indices[:100]:  # Limit to 100 for performance
            model_scores_batch.append({
                name: float(scores_arr[idx])
                for name, scores_arr in individual_scores.items()
            })
    else:
        model_scores_batch = [None] * min(len(anomaly_indices), 100)
    
    # Generate explanations for sample of anomalies
    sample_size = min(100, len(anomaly_indices))
    sample_indices = anomaly_indices[:sample_size]
    
    explanations = []
    for i, idx in enumerate(sample_indices):
        sample = X[idx]
        sample_dict = df.iloc[idx].to_dict() if idx < len(df) else {}
        score = float(scores[idx])
        
        if score >= 0.7:
            label = "Highly Suspicious"
        elif score >= 0.3:
            label = "Suspicious"
        else:
            label = "Normal"
        
        explanation = explainer.explain(
            sample,
            sample_dict,
            score,
            label,
            model_scores_batch[i] if i < len(model_scores_batch) else None,
            record_id=f"record_{idx}"
        )
        explanations.append(explanation)
    
    # Generate comprehensive summary
    summary = explainer.generate_summary_report(explanations)
    
    # Extract fraud patterns
    fraud_patterns = []
    for pattern_info in summary.get('top_fraud_patterns', []):
        fraud_patterns.append({
            "pattern": pattern_info['pattern'],
            "count": pattern_info['count'],
            "percentage": round(100 * pattern_info['count'] / len(explanations), 2)
        })
    
    # Extract top features
    top_features = []
    for feature_info in summary.get('most_anomalous_features', []):
        top_features.append({
            "feature": feature_info['feature'],
            "mentions": feature_info['mentions'],
            "percentage": round(100 * feature_info['mentions'] / len(explanations), 2)
        })
    
    # Severity distribution
    severity_dist = summary.get('severity_distribution', {})
    
    return {
        "job_id": job_id,
        "total_records": len(df),
        "total_anomalies": len(anomaly_indices),
        "analyzed_sample": len(explanations),
        "fraud_patterns": fraud_patterns,
        "top_features": top_features,
        "severity_distribution": severity_dist,
        "severity_details": summary.get('severity_details', {}),
        "avg_anomaly_score": summary.get('avg_anomaly_score', 0),
        "avg_confidence": summary.get('avg_confidence', 0),
        "generated_at": datetime.now().isoformat()
    }
