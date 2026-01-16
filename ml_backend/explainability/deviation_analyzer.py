"""
Feature Deviation Analyzer for Explainability.

Analyzes statistical and contextual deviations in anomalous records.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DeviationResult:
    """Result of deviation analysis for a single feature."""
    feature_name: str
    value: float
    expected_value: float
    deviation: float
    deviation_type: str  # 'statistical', 'temporal', 'geographic', 'behavioral'
    severity: str  # 'low', 'medium', 'high', 'critical'
    z_score: float
    percentile: float
    explanation: str


class DeviationAnalyzer:
    """
    Analyzes feature deviations to explain anomalies.
    
    Provides statistical, temporal, geographic, and behavioral deviation analysis.
    """
    
    def __init__(self):
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.temporal_baselines: Dict[str, Dict] = {}
        self.geographic_baselines: Dict[str, Dict] = {}
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, feature_names: List[str]) -> 'DeviationAnalyzer':
        """
        Compute baseline statistics for deviation analysis.
        
        Args:
            df: Training DataFrame
            feature_names: List of feature names to analyze
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting deviation analyzer on {len(df)} samples...")
        
        # Compute statistical baselines
        for feature in feature_names:
            if feature in df.columns:
                values = df[feature].dropna()
                if len(values) > 0:
                    self.feature_stats[feature] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'median': float(values.median()),
                        'q25': float(values.quantile(0.25)),
                        'q75': float(values.quantile(0.75)),
                        'q95': float(values.quantile(0.95)),
                        'q99': float(values.quantile(0.99)),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'iqr': float(values.quantile(0.75) - values.quantile(0.25))
                    }
        
        # Compute temporal baselines (if date column exists)
        if 'date' in df.columns or 'month' in df.columns:
            self._compute_temporal_baselines(df, feature_names)
        
        # Compute geographic baselines (if location columns exist)
        if any(col in df.columns for col in ['state', 'district', 'pincode']):
            self._compute_geographic_baselines(df, feature_names)
        
        self.is_fitted = True
        logger.info(f"Deviation analyzer fitted on {len(self.feature_stats)} features")
        
        return self
    
    def _compute_temporal_baselines(self, df: pd.DataFrame, feature_names: List[str]):
        """Compute temporal baselines for time-based deviation detection."""
        if 'month' in df.columns:
            for feature in feature_names:
                if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                    monthly_stats = df.groupby('month')[feature].agg(['mean', 'std']).to_dict('index')
                    self.temporal_baselines[feature] = {
                        'monthly': monthly_stats
                    }
        
        if 'day_of_week' in df.columns:
            for feature in feature_names:
                if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                    dow_stats = df.groupby('day_of_week')[feature].agg(['mean', 'std']).to_dict('index')
                    if feature not in self.temporal_baselines:
                        self.temporal_baselines[feature] = {}
                    self.temporal_baselines[feature]['day_of_week'] = dow_stats
    
    def _compute_geographic_baselines(self, df: pd.DataFrame, feature_names: List[str]):
        """Compute geographic baselines for location-based deviation detection."""
        if 'state' in df.columns:
            for feature in feature_names:
                if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                    state_stats = df.groupby('state')[feature].agg(['mean', 'std', 'count']).to_dict('index')
                    self.geographic_baselines[feature] = {
                        'state': state_stats
                    }
        
        if 'district' in df.columns:
            for feature in feature_names:
                if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                    district_stats = df.groupby('district')[feature].agg(['mean', 'std', 'count']).to_dict('index')
                    if feature not in self.geographic_baselines:
                        self.geographic_baselines[feature] = {}
                    self.geographic_baselines[feature]['district'] = district_stats
    
    def analyze_deviations(
        self,
        sample: Dict[str, Any],
        top_k: int = 10
    ) -> List[DeviationResult]:
        """
        Analyze deviations in a sample record.
        
        Args:
            sample: Dictionary of feature values
            top_k: Number of top deviations to return
            
        Returns:
            List of DeviationResult objects, sorted by severity
        """
        if not self.is_fitted:
            raise ValueError("Analyzer must be fitted first")
        
        deviations = []
        
        for feature, value in sample.items():
            if feature not in self.feature_stats:
                continue
            
            if not isinstance(value, (int, float, np.number)):
                continue
            
            stats = self.feature_stats[feature]
            
            # Calculate statistical deviation
            mean = stats['mean']
            std = stats['std']
            
            if std > 0:
                z_score = (value - mean) / std
            else:
                z_score = 0.0
            
            # Calculate percentile
            if value <= stats['min']:
                percentile = 0.0
            elif value >= stats['max']:
                percentile = 100.0
            else:
                # Approximate percentile
                if value < mean:
                    percentile = 50 * (value - stats['min']) / (mean - stats['min'])
                else:
                    percentile = 50 + 50 * (value - mean) / (stats['max'] - mean)
            
            # Determine deviation type and severity
            deviation_type = self._determine_deviation_type(feature, sample)
            severity = self._calculate_severity(z_score, percentile)
            explanation = self._generate_deviation_explanation(
                feature, value, mean, z_score, deviation_type, severity
            )
            
            deviations.append(DeviationResult(
                feature_name=feature,
                value=float(value),
                expected_value=mean,
                deviation=float(value - mean),
                deviation_type=deviation_type,
                severity=severity,
                z_score=float(abs(z_score)),
                percentile=float(percentile),
                explanation=explanation
            ))
        
        # Sort by z_score (severity)
        deviations.sort(key=lambda x: x.z_score, reverse=True)
        
        return deviations[:top_k]
    
    def _determine_deviation_type(self, feature: str, sample: Dict[str, Any]) -> str:
        """Determine the type of deviation based on feature name and context."""
        feature_lower = feature.lower()
        
        if any(x in feature_lower for x in ['state', 'district', 'pincode', 'geo']):
            return 'geographic'
        elif any(x in feature_lower for x in ['month', 'day', 'weekend', 'quarter', 'time']):
            return 'temporal'
        elif any(x in feature_lower for x in ['operator', 'center', 'update', 'enrolment']):
            return 'behavioral'
        else:
            return 'statistical'
    
    def _calculate_severity(self, z_score: float, percentile: float) -> str:
        """Calculate severity level based on z-score and percentile."""
        abs_z = abs(z_score)
        
        if abs_z > 4 or percentile > 99.5 or percentile < 0.5:
            return 'critical'
        elif abs_z > 3 or percentile > 99 or percentile < 1:
            return 'high'
        elif abs_z > 2 or percentile > 95 or percentile < 5:
            return 'medium'
        else:
            return 'low'
    
    def _generate_deviation_explanation(
        self,
        feature: str,
        value: float,
        expected: float,
        z_score: float,
        deviation_type: str,
        severity: str
    ) -> str:
        """Generate human-readable explanation for a deviation."""
        direction = "higher" if value > expected else "lower"
        
        # Severity descriptor
        if severity == 'critical':
            intensity = "extremely"
        elif severity == 'high':
            intensity = "significantly"
        elif severity == 'medium':
            intensity = "moderately"
        else:
            intensity = "slightly"
        
        # Feature-friendly name
        feature_display = feature.replace('_', ' ')
        
        # Type-specific context
        if deviation_type == 'geographic':
            context = " (unusual for this location)"
        elif deviation_type == 'temporal':
            context = " (unusual for this time period)"
        elif deviation_type == 'behavioral':
            context = " (deviates from typical behavior)"
        else:
            context = ""
        
        return f"{feature_display.title()} is {intensity} {direction} than expected{context}"
    
    def compare_to_similar_records(
        self,
        sample: Dict[str, Any],
        similar_samples: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare anomalous sample to similar normal records.
        
        Args:
            sample: Anomalous record
            similar_samples: DataFrame of similar normal records
            features: Optional list of features to compare
            
        Returns:
            Dictionary with comparison results
        """
        if features is None:
            features = list(self.feature_stats.keys())
        
        comparisons = {}
        
        for feature in features:
            if feature not in sample or feature not in similar_samples.columns:
                continue
            
            value = sample[feature]
            if not isinstance(value, (int, float, np.number)):
                continue
            
            similar_values = similar_samples[feature].dropna()
            if len(similar_values) == 0:
                continue
            
            comparisons[feature] = {
                'anomaly_value': float(value),
                'similar_mean': float(similar_values.mean()),
                'similar_std': float(similar_values.std()),
                'similar_median': float(similar_values.median()),
                'percentile_in_similar': float(np.mean(similar_values <= value) * 100),
                'is_outlier': value > similar_values.quantile(0.95) or value < similar_values.quantile(0.05)
            }
        
        return comparisons
    
    def detect_temporal_deviations(
        self,
        sample: Dict[str, Any],
        features: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect temporal deviations (unusual activity timing).
        
        Args:
            sample: Record to analyze
            features: Optional list of features to check
            
        Returns:
            List of temporal deviation findings
        """
        temporal_deviations = []
        
        # Check weekend activity
        if sample.get('is_weekend') == 1:
            temporal_deviations.append({
                'type': 'weekend_activity',
                'severity': 'medium',
                'explanation': 'Activity occurred during weekend (unusual for government operations)'
            })
        
        # Check month-end/start activity
        if sample.get('is_month_end') == 1:
            temporal_deviations.append({
                'type': 'month_end_activity',
                'severity': 'low',
                'explanation': 'Activity occurred at month-end (possible target-driven behavior)'
            })
        
        if sample.get('is_month_start') == 1:
            temporal_deviations.append({
                'type': 'month_start_activity',
                'severity': 'low',
                'explanation': 'Activity occurred at month-start'
            })
        
        # Check if features deviate from temporal baselines
        if features is None:
            features = list(self.temporal_baselines.keys())
        
        for feature in features:
            if feature not in self.temporal_baselines or feature not in sample:
                continue
            
            value = sample[feature]
            if not isinstance(value, (int, float, np.number)):
                continue
            
            # Check monthly baseline
            if 'monthly' in self.temporal_baselines[feature] and 'month' in sample:
                month = sample['month']
                if month in self.temporal_baselines[feature]['monthly']:
                    baseline = self.temporal_baselines[feature]['monthly'][month]
                    expected = baseline['mean']
                    std = baseline['std']
                    
                    if std > 0:
                        z_score = abs((value - expected) / std)
                        if z_score > 2:
                            temporal_deviations.append({
                                'type': 'monthly_deviation',
                                'feature': feature,
                                'severity': 'high' if z_score > 3 else 'medium',
                                'explanation': f'{feature} deviates from typical pattern for this month'
                            })
        
        return temporal_deviations
    
    def detect_geographic_deviations(
        self,
        sample: Dict[str, Any],
        features: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect geographic deviations (unusual location patterns).
        
        Args:
            sample: Record to analyze
            features: Optional list of features to check
            
        Returns:
            List of geographic deviation findings
        """
        geo_deviations = []
        
        if features is None:
            features = list(self.geographic_baselines.keys())
        
        for feature in features:
            if feature not in self.geographic_baselines or feature not in sample:
                continue
            
            value = sample[feature]
            if not isinstance(value, (int, float, np.number)):
                continue
            
            # Check state baseline
            if 'state' in self.geographic_baselines[feature] and 'state' in sample:
                state = sample['state']
                if state in self.geographic_baselines[feature]['state']:
                    baseline = self.geographic_baselines[feature]['state'][state]
                    expected = baseline['mean']
                    std = baseline['std']
                    
                    if std > 0:
                        z_score = abs((value - expected) / std)
                        if z_score > 2:
                            geo_deviations.append({
                                'type': 'state_deviation',
                                'feature': feature,
                                'location': state,
                                'severity': 'high' if z_score > 3 else 'medium',
                                'explanation': f'{feature} is unusual for state {state}'
                            })
            
            # Check district baseline
            if 'district' in self.geographic_baselines[feature] and 'district' in sample:
                district = sample['district']
                if district in self.geographic_baselines[feature]['district']:
                    baseline = self.geographic_baselines[feature]['district'][district]
                    expected = baseline['mean']
                    std = baseline['std']
                    
                    if std > 0:
                        z_score = abs((value - expected) / std)
                        if z_score > 2:
                            geo_deviations.append({
                                'type': 'district_deviation',
                                'feature': feature,
                                'location': district,
                                'severity': 'high' if z_score > 3 else 'medium',
                                'explanation': f'{feature} is unusual for district {district}'
                            })
        
        return geo_deviations


# Singleton instance
_analyzer_instance: Optional[DeviationAnalyzer] = None


def get_deviation_analyzer() -> DeviationAnalyzer:
    """Get or create DeviationAnalyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = DeviationAnalyzer()
    return _analyzer_instance
