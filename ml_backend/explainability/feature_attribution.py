"""
Feature attribution module for explainability.
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureAttribution:
    """
    Analyze feature contributions to anomaly scores.
    
    Provides SHAP-like explanations for why a record is anomalous.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
        self.baseline_values: Optional[np.ndarray] = None
        self.feature_stats: Dict[str, Dict[str, float]] = {}
    
    def fit(self, X: np.ndarray, feature_names: List[str]) -> 'FeatureAttribution':
        """
        Compute baseline statistics for attribution.
        
        Args:
            X: Training data
            feature_names: Names of features
            
        Returns:
            Self for method chaining
        """
        self.feature_names = feature_names
        self.baseline_values = np.mean(X, axis=0)
        
        # Compute feature statistics
        for i, name in enumerate(feature_names):
            self.feature_stats[name] = {
                "mean": float(np.mean(X[:, i])),
                "std": float(np.std(X[:, i])),
                "min": float(np.min(X[:, i])),
                "max": float(np.max(X[:, i])),
                "q25": float(np.percentile(X[:, i], 25)),
                "q75": float(np.percentile(X[:, i], 75))
            }
        
        logger.info(f"Feature attribution fitted on {len(feature_names)} features")
        return self
    
    def get_attributions(self, sample: np.ndarray) -> Dict[str, float]:
        """
        Get feature attributions for a single sample.
        
        Attribution is based on deviation from baseline (population mean).
        
        Args:
            sample: Single sample to explain
            
        Returns:
            Dict mapping feature names to attribution scores
        """
        if self.baseline_values is None:
            raise ValueError("Must fit before getting attributions")
        
        sample = sample.flatten()
        attributions = {}
        
        for i, name in enumerate(self.feature_names):
            if i >= len(sample):
                break
                
            stats = self.feature_stats.get(name, {})
            std = stats.get("std", 1.0)
            
            if std > 0:
                # Z-score based attribution
                z_score = (sample[i] - stats.get("mean", 0)) / std
                attributions[name] = float(abs(z_score))
            else:
                attributions[name] = 0.0
        
        # Normalize to sum to 1
        total = sum(attributions.values())
        if total > 0:
            attributions = {k: v / total for k, v in attributions.items()}
        
        return dict(sorted(attributions.items(), key=lambda x: x[1], reverse=True))
    
    def get_batch_attributions(self, X: np.ndarray) -> List[Dict[str, float]]:
        """Get attributions for multiple samples."""
        return [self.get_attributions(X[i]) for i in range(len(X))]
    
    def get_top_contributors(self, sample: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top contributing features for a sample.
        
        Args:
            sample: Sample to explain
            top_k: Number of top features to return
            
        Returns:
            List of dicts with feature info
        """
        attributions = self.get_attributions(sample)
        sample = sample.flatten()
        
        contributors = []
        for i, (name, score) in enumerate(attributions.items()):
            if i >= top_k:
                break
            
            feature_idx = self.feature_names.index(name) if name in self.feature_names else None
            
            if feature_idx is not None and feature_idx < len(sample):
                stats = self.feature_stats.get(name, {})
                value = sample[feature_idx]
                
                # Determine deviation direction
                mean = stats.get("mean", 0)
                if value > mean:
                    direction = "above"
                elif value < mean:
                    direction = "below"
                else:
                    direction = "at"
                
                contributors.append({
                    "feature": name,
                    "attribution": score,
                    "value": float(value),
                    "mean": mean,
                    "direction": direction,
                    "z_score": float((value - mean) / stats.get("std", 1)) if stats.get("std", 0) > 0 else 0
                })
        
        return contributors
    
    def compare_to_normal(self, anomaly_sample: np.ndarray, normal_samples: np.ndarray) -> Dict[str, Any]:
        """
        Compare an anomalous sample to normal samples.
        
        Args:
            anomaly_sample: The anomalous sample
            normal_samples: Array of normal samples for comparison
            
        Returns:
            Dict with comparison details
        """
        anomaly_sample = anomaly_sample.flatten()
        
        comparison = {}
        for i, name in enumerate(self.feature_names):
            if i >= len(anomaly_sample):
                break
                
            anomaly_value = anomaly_sample[i]
            normal_mean = np.mean(normal_samples[:, i])
            normal_std = np.std(normal_samples[:, i])
            
            comparison[name] = {
                "anomaly_value": float(anomaly_value),
                "normal_mean": float(normal_mean),
                "normal_std": float(normal_std),
                "deviation": float(anomaly_value - normal_mean),
                "percentile": float(np.mean(normal_samples[:, i] <= anomaly_value) * 100)
            }
        
        return comparison


class ReasonGenerator:
    """
    Generate human-readable explanations for anomalies.
    """
    
    def __init__(self):
        self.feature_descriptions = {
            # Temporal features
            "month": "time of year",
            "year": "year",
            "quarter": "quarter",
            "day_of_week": "day of week",
            "is_weekend": "weekend activity",
            "is_month_end": "month-end timing",
            
            # Geographic features
            "state_event_count": "state activity level",
            "district_event_count": "district activity level",
            "pincode_event_count": "pincode activity level",
            "state_event_pct": "state's share of events",
            "district_event_pct": "district's share of events",
            
            # Enrolment features
            "total_enrolments": "total enrolment count",
            "age_0_5_ratio": "infant enrolment ratio",
            "age_5_17_ratio": "child/teen enrolment ratio",
            "age_18_greater_ratio": "adult enrolment ratio",
            
            # Update features
            "total_demo_updates": "demographic update count",
            "demo_update_zscore": "demographic update intensity",
            "total_bio_updates": "biometric update count",
            "bio_update_zscore": "biometric update intensity",
            
            # Statistical features
            "row_mean": "average activity level",
            "row_std": "activity variability",
            "row_max": "peak activity",
            "row_range": "activity range"
        }
    
    def generate_reasons(
        self,
        contributors: List[Dict[str, Any]],
        anomaly_score: float,
        model_results: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Generate human-readable reasons for anomaly.
        
        Args:
            contributors: Top contributing features from FeatureAttribution
            anomaly_score: Overall anomaly score
            model_results: Results from individual models
            
        Returns:
            List of explanation strings
        """
        reasons = []
        
        for contrib in contributors[:5]:  # Top 5 reasons
            feature = contrib["feature"]
            direction = contrib["direction"]
            z_score = abs(contrib.get("z_score", 0))
            
            # Get human-readable feature description
            feature_desc = self.feature_descriptions.get(feature, feature.replace("_", " "))
            
            # Generate severity word
            if z_score > 3:
                severity = "extremely"
            elif z_score > 2:
                severity = "significantly"
            elif z_score > 1:
                severity = "moderately"
            else:
                severity = "slightly"
            
            # Generate reason
            if direction == "above":
                reason = f"The {feature_desc} is {severity} higher than expected"
            elif direction == "below":
                reason = f"The {feature_desc} is {severity} lower than expected"
            else:
                continue
            
            # Add context for specific features
            if "spike" in feature.lower() or "count" in feature.lower():
                reason += " (possible volume spike)"
            elif "ratio" in feature.lower():
                reason += " (unusual distribution)"
            elif "zscore" in feature.lower():
                reason += " (statistical outlier)"
            
            reasons.append(reason)
        
        # Add model-specific insights
        if model_results:
            for result in model_results:
                if result.get("model_name") == "hdbscan":
                    n_clusters = result.get("metadata", {}).get("n_clusters", 0)
                    if n_clusters > 0:
                        reasons.append(f"Record does not fit into any of the {n_clusters} normal behavioral clusters")
                
                if result.get("model_name") == "autoencoder":
                    reasons.append("Complex pattern deviates from learned normal behavior")
        
        # Add overall severity
        if anomaly_score > 0.9:
            reasons.insert(0, "⚠️ CRITICAL: Multiple strong anomaly indicators detected")
        elif anomaly_score > 0.7:
            reasons.insert(0, "⚠️ HIGH RISK: Significant deviations from normal patterns")
        elif anomaly_score > 0.5:
            reasons.insert(0, "⚡ MEDIUM RISK: Notable anomalies requiring review")
        
        return reasons if reasons else ["No specific anomaly patterns identified"]
    
    def generate_fraud_pattern(self, contributors: List[Dict[str, Any]]) -> str:
        """
        Identify the likely fraud pattern based on features.
        
        Args:
            contributors: Top contributing features
            
        Returns:
            String describing the likely fraud pattern
        """
        feature_names = [c["feature"] for c in contributors[:3]]
        
        # Pattern detection logic
        if any("enrolment" in f or "age" in f for f in feature_names):
            if any("spike" in f or "count" in f for f in feature_names):
                return "Volume Spike Fraud: Unusual surge in enrolments"
            else:
                return "Demographic Anomaly: Unusual age distribution in enrolments"
        
        if any("geo" in f or "state" in f or "district" in f for f in feature_names):
            return "Geographic Anomaly: Activity pattern inconsistent with location norms"
        
        if any("bio" in f for f in feature_names):
            return "Biometric Update Anomaly: Unusual biometric update patterns"
        
        if any("demo" in f for f in feature_names):
            return "Demographic Update Anomaly: Suspicious demographic changes"
        
        if any("operator" in f or "center" in f for f in feature_names):
            return "Operator/Center Fraud: Unusual activity from specific operators or centers"
        
        if any("time" in f or "month" in f or "weekend" in f for f in feature_names):
            return "Temporal Anomaly: Activity occurring at unusual times"
        
        return "General Statistical Anomaly: Multiple deviations from normal patterns"
