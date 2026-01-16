"""
Unified Fraud Explainer for UIDAI Anomaly Detection.

Aggregates explanations from all models and generates comprehensive fraud reports.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from explainability.feature_attribution import FeatureAttribution, ReasonGenerator
from explainability.deviation_analyzer import DeviationAnalyzer, DeviationResult

logger = logging.getLogger(__name__)


@dataclass
class FraudExplanation:
    """Complete fraud explanation for a single record."""
    record_id: Optional[str]
    anomaly_score: float
    anomaly_label: str
    confidence: float
    fraud_pattern: str
    primary_reasons: List[str]
    top_features: List[Dict[str, Any]]
    deviations: List[Dict[str, Any]]
    model_contributions: Dict[str, Any]
    temporal_flags: List[Dict[str, Any]]
    geographic_flags: List[Dict[str, Any]]
    severity: str
    recommendation: str
    timestamp: str


class FraudExplainer:
    """
    Unified fraud explanation engine.
    
    Combines insights from:
    - Isolation Forest (statistical outliers)
    - HDBSCAN (cluster deviations)
    - Autoencoder (reconstruction errors)
    - Feature attribution analysis
    - Deviation analysis
    """
    
    def __init__(self):
        self.feature_attributor = FeatureAttribution()
        self.reason_generator = ReasonGenerator()
        self.deviation_analyzer = DeviationAnalyzer()
        self.is_fitted = False
        
    def fit(
        self,
        X: np.ndarray,
        df: pd.DataFrame,
        feature_names: List[str]
    ) -> 'FraudExplainer':
        """
        Fit the explainer on training data.
        
        Args:
            X: Training data array
            df: Training DataFrame (for contextual analysis)
            feature_names: List of feature names
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting fraud explainer...")
        
        # Fit feature attributor
        self.feature_attributor.fit(X, feature_names)
        
        # Fit deviation analyzer
        self.deviation_analyzer.fit(df, feature_names)
        
        self.is_fitted = True
        logger.info("Fraud explainer fitted successfully")
        
        return self
    
    def explain(
        self,
        sample: np.ndarray,
        sample_dict: Dict[str, Any],
        anomaly_score: float,
        anomaly_label: str,
        model_scores: Optional[Dict[str, float]] = None,
        record_id: Optional[str] = None
    ) -> FraudExplanation:
        """
        Generate comprehensive fraud explanation for a single record.
        
        Args:
            sample: Feature array
            sample_dict: Feature dictionary (for contextual analysis)
            anomaly_score: Overall anomaly score from ensemble
            anomaly_label: Anomaly label (Normal, Suspicious, Highly Suspicious)
            model_scores: Optional individual model scores
            record_id: Optional record identifier
            
        Returns:
            FraudExplanation object
        """
        if not self.is_fitted:
            raise ValueError("Explainer must be fitted first")
        
        # Get feature attributions
        top_contributors = self.feature_attributor.get_top_contributors(sample, top_k=10)
        
        # Prepare model results for reason generation
        model_results = []
        if model_scores:
            for model_name, score in model_scores.items():
                model_results.append({
                    'model_name': model_name,
                    'score': score,
                    'metadata': self._get_model_metadata(model_name, score)
                })
        
        # Generate primary reasons
        primary_reasons = self.reason_generator.generate_reasons(
            top_contributors,
            anomaly_score,
            model_results
        )
        
        # Identify fraud pattern
        fraud_pattern = self.reason_generator.generate_fraud_pattern(
            top_contributors,
            model_results
        )
        
        # Analyze deviations
        deviations = self.deviation_analyzer.analyze_deviations(sample_dict, top_k=10)
        
        # Detect temporal deviations
        temporal_flags = self.deviation_analyzer.detect_temporal_deviations(sample_dict)
        
        # Detect geographic deviations
        geographic_flags = self.deviation_analyzer.detect_geographic_deviations(sample_dict)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            anomaly_score,
            model_scores,
            len(deviations),
            len(temporal_flags),
            len(geographic_flags)
        )
        
        # Determine severity
        severity = self._determine_severity(anomaly_score, confidence, deviations)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(severity, fraud_pattern, anomaly_label)
        
        # Build model contributions summary
        model_contributions = self._build_model_contributions(model_scores, model_results)
        
        return FraudExplanation(
            record_id=record_id,
            anomaly_score=float(anomaly_score),
            anomaly_label=anomaly_label,
            confidence=confidence,
            fraud_pattern=fraud_pattern,
            primary_reasons=primary_reasons,
            top_features=[
                {
                    'feature': c['feature'],
                    'value': c['value'],
                    'mean': c['mean'],
                    'direction': c['direction'],
                    'z_score': c['z_score'],
                    'attribution': c['attribution']
                }
                for c in top_contributors[:5]
            ],
            deviations=[
                {
                    'feature': d.feature_name,
                    'value': d.value,
                    'expected': d.expected_value,
                    'deviation_type': d.deviation_type,
                    'severity': d.severity,
                    'z_score': d.z_score,
                    'explanation': d.explanation
                }
                for d in deviations[:5]
            ],
            model_contributions=model_contributions,
            temporal_flags=temporal_flags,
            geographic_flags=geographic_flags,
            severity=severity,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
    
    def explain_batch(
        self,
        X: np.ndarray,
        df: pd.DataFrame,
        anomaly_scores: np.ndarray,
        anomaly_labels: List[str],
        model_scores_batch: Optional[List[Dict[str, float]]] = None,
        record_ids: Optional[List[str]] = None
    ) -> List[FraudExplanation]:
        """
        Generate explanations for multiple records.
        
        Args:
            X: Feature array (n_samples, n_features)
            df: DataFrame with feature names
            anomaly_scores: Array of anomaly scores
            anomaly_labels: List of anomaly labels
            model_scores_batch: Optional list of model score dicts
            record_ids: Optional list of record IDs
            
        Returns:
            List of FraudExplanation objects
        """
        explanations = []
        
        for i in range(len(X)):
            sample = X[i]
            sample_dict = df.iloc[i].to_dict() if i < len(df) else {}
            score = anomaly_scores[i]
            label = anomaly_labels[i]
            model_scores = model_scores_batch[i] if model_scores_batch else None
            record_id = record_ids[i] if record_ids else f"record_{i}"
            
            explanation = self.explain(
                sample,
                sample_dict,
                score,
                label,
                model_scores,
                record_id
            )
            explanations.append(explanation)
        
        return explanations
    
    def _get_model_metadata(self, model_name: str, score: float) -> Dict[str, Any]:
        """Get model-specific metadata for explanation."""
        metadata = {}
        
        if model_name == 'hdbscan':
            metadata['type'] = 'Density-based clustering'
            metadata['interpretation'] = 'Measures how well record fits into behavioral clusters'
            if score > 0.7:
                metadata['finding'] = 'Does not belong to any normal behavior cluster'
        
        elif model_name == 'isolation_forest':
            metadata['type'] = 'Tree-based isolation'
            metadata['interpretation'] = 'Measures how easily record can be isolated from others'
            if score > 0.7:
                metadata['finding'] = 'Easily isolated from normal records (statistical outlier)'
        
        elif model_name == 'autoencoder':
            metadata['type'] = 'Deep learning reconstruction'
            metadata['interpretation'] = 'Measures reconstruction error from learned normal patterns'
            if score > 0.7:
                metadata['finding'] = 'High reconstruction error (complex pattern deviation)'
        
        elif model_name == 'one_class_svm':
            metadata['type'] = 'Support vector boundary'
            metadata['interpretation'] = 'Measures distance from normal data boundary'
            if score > 0.7:
                metadata['finding'] = 'Falls outside normal data boundary'
        
        return metadata
    
    def _calculate_confidence(
        self,
        anomaly_score: float,
        model_scores: Optional[Dict[str, float]],
        num_deviations: int,
        num_temporal_flags: int,
        num_geo_flags: int
    ) -> float:
        """
        Calculate confidence in the anomaly detection.
        
        Higher confidence when:
        - Multiple models agree
        - High anomaly score
        - Multiple deviations detected
        - Temporal/geographic flags present
        """
        confidence = 0.0
        
        # Base confidence from anomaly score
        confidence += anomaly_score * 0.4
        
        # Model agreement
        if model_scores:
            high_scores = sum(1 for score in model_scores.values() if score > 0.6)
            model_agreement = high_scores / len(model_scores)
            confidence += model_agreement * 0.3
        
        # Deviation evidence
        deviation_score = min(num_deviations / 5.0, 1.0)  # Normalize to 0-1
        confidence += deviation_score * 0.2
        
        # Contextual flags
        flag_score = min((num_temporal_flags + num_geo_flags) / 3.0, 1.0)
        confidence += flag_score * 0.1
        
        return min(confidence, 1.0)
    
    def _determine_severity(
        self,
        anomaly_score: float,
        confidence: float,
        deviations: List[DeviationResult]
    ) -> str:
        """Determine overall severity level."""
        # Count critical/high severity deviations
        critical_count = sum(1 for d in deviations if d.severity == 'critical')
        high_count = sum(1 for d in deviations if d.severity == 'high')
        
        if anomaly_score > 0.9 or (confidence > 0.8 and critical_count >= 2):
            return 'CRITICAL'
        elif anomaly_score > 0.7 or (confidence > 0.6 and (critical_count >= 1 or high_count >= 2)):
            return 'HIGH'
        elif anomaly_score > 0.5 or confidence > 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_recommendation(
        self,
        severity: str,
        fraud_pattern: str,
        anomaly_label: str
    ) -> str:
        """Generate actionable recommendation for auditors."""
        if severity == 'CRITICAL':
            return f"IMMEDIATE ACTION REQUIRED: {fraud_pattern}. Conduct thorough investigation and consider suspension pending review."
        elif severity == 'HIGH':
            return f"PRIORITY INVESTIGATION: {fraud_pattern}. Review all related transactions and operator activity."
        elif severity == 'MEDIUM':
            return f"REVIEW RECOMMENDED: {fraud_pattern}. Flag for manual verification in next audit cycle."
        else:
            return f"LOW PRIORITY: {fraud_pattern}. Monitor for recurring patterns."
    
    def _build_model_contributions(
        self,
        model_scores: Optional[Dict[str, float]],
        model_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build summary of model contributions."""
        if not model_scores:
            return {}
        
        contributions = {}
        for model_name, score in model_scores.items():
            metadata = next(
                (r['metadata'] for r in model_results if r['model_name'] == model_name),
                {}
            )
            
            contributions[model_name] = {
                'score': float(score),
                'weight': self._get_model_weight(model_name),
                'type': metadata.get('type', 'Unknown'),
                'finding': metadata.get('finding', 'No significant finding')
            }
        
        return contributions
    
    def _get_model_weight(self, model_name: str) -> float:
        """Get default model weight (should match ensemble config)."""
        weights = {
            'isolation_forest': 0.35,
            'hdbscan': 0.25,
            'autoencoder': 0.30,
            'one_class_svm': 0.10
        }
        return weights.get(model_name, 0.25)
    
    def generate_summary_report(
        self,
        explanations: List[FraudExplanation]
    ) -> Dict[str, Any]:
        """
        Generate summary report for multiple explanations.
        
        Args:
            explanations: List of fraud explanations
            
        Returns:
            Summary statistics and insights
        """
        if not explanations:
            return {}
        
        # Count by severity
        severity_counts = {}
        for exp in explanations:
            severity_counts[exp.severity] = severity_counts.get(exp.severity, 0) + 1
        
        # Count by fraud pattern
        pattern_counts = {}
        for exp in explanations:
            pattern_counts[exp.fraud_pattern] = pattern_counts.get(exp.fraud_pattern, 0) + 1
        
        # Top fraud patterns
        top_patterns = sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Average scores by severity
        severity_scores = {}
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            scores = [e.anomaly_score for e in explanations if e.severity == severity]
            if scores:
                severity_scores[severity] = {
                    'count': len(scores),
                    'avg_score': float(np.mean(scores)),
                    'avg_confidence': float(np.mean([e.confidence for e in explanations if e.severity == severity]))
                }
        
        # Most common features in anomalies
        feature_mentions = {}
        for exp in explanations:
            for feature in exp.top_features:
                fname = feature['feature']
                feature_mentions[fname] = feature_mentions.get(fname, 0) + 1
        
        top_features = sorted(
            feature_mentions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_records': len(explanations),
            'severity_distribution': severity_counts,
            'severity_details': severity_scores,
            'top_fraud_patterns': [{'pattern': p, 'count': c} for p, c in top_patterns],
            'most_anomalous_features': [{'feature': f, 'mentions': c} for f, c in top_features],
            'avg_anomaly_score': float(np.mean([e.anomaly_score for e in explanations])),
            'avg_confidence': float(np.mean([e.confidence for e in explanations])),
            'timestamp': datetime.now().isoformat()
        }


# Singleton instance
_explainer_instance: Optional[FraudExplainer] = None


def get_fraud_explainer() -> FraudExplainer:
    """Get or create FraudExplainer singleton."""
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = FraudExplainer()
    return _explainer_instance
