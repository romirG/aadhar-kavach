"""
Isolation Forest model for anomaly detection.
"""
import logging
import numpy as np
from typing import Tuple, Optional
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
import joblib

import sys
sys.path.insert(0, '..')

from config import get_settings

logger = logging.getLogger(__name__)


class IsolationForestModel:
    """
    Isolation Forest for unsupervised anomaly detection.
    
    Effective for detecting:
    - Volume spikes in enrolments
    - Unusual activity patterns
    - Outliers in aggregate statistics
    """
    
    def __init__(self, contamination: Optional[float] = None, random_state: int = 42):
        """
        Initialize Isolation Forest model.
        
        Args:
            contamination: Expected proportion of outliers (default from config)
            random_state: Random seed for reproducibility
        """
        settings = get_settings()
        self.contamination = contamination or settings.isolation_forest_contamination
        self.random_state = random_state
        self.model: Optional[SklearnIsolationForest] = None
        self.is_fitted = False
        self.feature_names: list = []
    
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None) -> 'IsolationForestModel':
        """
        Fit the Isolation Forest model.
        
        Args:
            X: Training data (n_samples, n_features)
            feature_names: Names of features for explainability
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Isolation Forest on {X.shape[0]} samples with {X.shape[1]} features")
        
        self.model = SklearnIsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
            bootstrap=True
        )
        
        self.model.fit(X)
        self.is_fitted = True
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info("Isolation Forest training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Data to predict on
            
        Returns:
            Array of -1 (anomaly) or 1 (normal) for each sample
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for samples.
        
        Args:
            X: Data to score
            
        Returns:
            Array of anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        # Isolation Forest returns negative scores, more negative = more anomalous
        raw_scores = self.model.score_samples(X)
        
        # Normalize to 0-1 range where higher = more anomalous
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        
        if max_score - min_score > 0:
            normalized_scores = 1 - (raw_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(raw_scores)
        
        return normalized_scores
    
    def fit_predict(self, X: np.ndarray, feature_names: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit model and return predictions and scores.
        
        Args:
            X: Training data
            feature_names: Names of features
            
        Returns:
            Tuple of (predictions, normalized_scores)
        """
        self.fit(X, feature_names)
        predictions = self.predict(X)
        scores = self.score_samples(X)
        
        anomaly_count = (predictions == -1).sum()
        logger.info(f"Isolation Forest detected {anomaly_count} anomalies ({100*anomaly_count/len(predictions):.2f}%)")
        
        return predictions, scores
    
    def get_feature_importance(self, X: np.ndarray) -> dict:
        """
        Estimate feature importance based on anomaly score variance.
        
        Args:
            X: Data used for importance estimation
            
        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        base_scores = self.score_samples(X)
        importances = {}
        
        # Permutation importance
        for i, feature_name in enumerate(self.feature_names):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_scores = self.score_samples(X_permuted)
            importance = np.abs(base_scores - permuted_scores).mean()
            importances[feature_name] = float(importance)
        
        # Normalize
        max_imp = max(importances.values()) if importances else 1
        importances = {k: v/max_imp for k, v in importances.items()}
        
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'contamination': self.contamination
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.contamination = data['contamination']
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
