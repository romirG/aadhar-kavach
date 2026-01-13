"""
HDBSCAN clustering model for spatial anomaly detection.
"""
import logging
import numpy as np
from typing import Tuple, Optional, List
import hdbscan

import sys
sys.path.insert(0, '..')

from config import get_settings

logger = logging.getLogger(__name__)


class ClusteringModel:
    """
    HDBSCAN for density-based anomaly detection.
    
    Effective for:
    - Geographic anomalies (unusual location patterns)
    - Cluster-based outliers
    - Points that don't belong to any natural cluster
    """
    
    def __init__(
        self,
        min_cluster_size: Optional[int] = None,
        min_samples: int = 5,
        metric: str = 'euclidean'
    ):
        """
        Initialize HDBSCAN clustering model.
        
        Args:
            min_cluster_size: Minimum number of points in a cluster
            min_samples: Core point threshold
            metric: Distance metric to use
        """
        settings = get_settings()
        self.min_cluster_size = min_cluster_size or settings.hdbscan_min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        
        self.model: Optional[hdbscan.HDBSCAN] = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.n_clusters: int = 0
        self.noise_ratio: float = 0.0
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'ClusteringModel':
        """
        Fit HDBSCAN clustering.
        
        Args:
            X: Training data (n_samples, n_features)
            feature_names: Names of features
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting HDBSCAN on {X.shape[0]} samples with {X.shape[1]} features")
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Initialize and fit HDBSCAN
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        self.model.fit(X)
        self.is_fitted = True
        
        # Calculate statistics
        labels = self.model.labels_
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.noise_ratio = (labels == -1).sum() / len(labels)
        
        logger.info(f"HDBSCAN found {self.n_clusters} clusters, {self.noise_ratio*100:.2f}% noise points")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels and identify anomalies.
        
        Points labeled as -1 (noise) are considered anomalies.
        
        Args:
            X: Data to predict on
            
        Returns:
            Array of -1 (anomaly/noise) or 1 (normal/clustered)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use approximate prediction for new data
        labels, _ = hdbscan.approximate_predict(self.model, X)
        
        # Convert: noise (-1) stays -1, clustered becomes 1
        predictions = np.where(labels == -1, -1, 1)
        return predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores based on outlier probability.
        
        Args:
            X: Data to score
            
        Returns:
            Array of anomaly scores (0-1, higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        # Get outlier scores from HDBSCAN
        # outlier_scores_ represents probability of being an outlier
        if hasattr(self.model, 'outlier_scores_') and self.model.outlier_scores_ is not None:
            # For training data, use pre-computed scores
            if len(X) == len(self.model.outlier_scores_):
                scores = self.model.outlier_scores_
            else:
                # For new data, approximate based on cluster membership
                labels, strengths = hdbscan.approximate_predict(self.model, X)
                # Invert strength: high strength = low anomaly score
                scores = 1 - strengths
                # Noise points get high anomaly score
                scores[labels == -1] = 0.9
        else:
            # Fallback: use cluster membership
            labels, strengths = hdbscan.approximate_predict(self.model, X)
            scores = 1 - strengths
            scores[labels == -1] = 0.9
        
        # Normalize to 0-1
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score > 0:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(scores)
        
        return normalized_scores
    
    def fit_predict(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit model and return predictions and scores.
        
        Args:
            X: Training data
            feature_names: Names of features
            
        Returns:
            Tuple of (predictions, normalized_scores)
        """
        self.fit(X, feature_names)
        
        # Use fitted labels for predictions
        labels = self.model.labels_
        predictions = np.where(labels == -1, -1, 1)
        scores = self.score_samples(X)
        
        anomaly_count = (predictions == -1).sum()
        logger.info(f"HDBSCAN detected {anomaly_count} anomalies ({100*anomaly_count/len(predictions):.2f}%)")
        
        return predictions, scores
    
    def get_cluster_info(self) -> dict:
        """
        Get information about discovered clusters.
        
        Returns:
            Dict with cluster statistics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        labels = self.model.labels_
        
        cluster_info = {
            "n_clusters": self.n_clusters,
            "noise_count": int((labels == -1).sum()),
            "noise_ratio": self.noise_ratio,
            "cluster_sizes": {}
        }
        
        for cluster_id in range(self.n_clusters):
            cluster_info["cluster_sizes"][f"cluster_{cluster_id}"] = int((labels == cluster_id).sum())
        
        return cluster_info
    
    def get_cluster_labels(self) -> np.ndarray:
        """Get cluster labels from fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.labels_
    
    def get_cluster_probabilities(self) -> np.ndarray:
        """Get probability of cluster membership for each point."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.probabilities_
