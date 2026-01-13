"""
Ensemble model combining multiple anomaly detection methods.
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

import sys
sys.path.insert(0, '..')

from config import get_settings
from models.isolation_forest import IsolationForestModel
from models.autoencoder import AutoencoderModel
from models.clustering import ClusteringModel

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result from a single model."""
    model_name: str
    predictions: np.ndarray
    scores: np.ndarray
    threshold: float
    anomaly_count: int
    execution_time_ms: float
    metadata: Dict[str, Any]


class EnsembleScorer:
    """
    Ensemble anomaly detector combining multiple models.
    
    Automatically selects appropriate models based on data profile
    and combines their scores using weighted voting.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble scorer.
        
        Args:
            weights: Dict mapping model names to weights (default from config)
        """
        settings = get_settings()
        self.weights = weights or settings.ensemble_weights
        
        self.models: Dict[str, Any] = {}
        self.model_results: Dict[str, ModelResult] = {}
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.data_profile: Dict[str, Any] = {}
    
    def _profile_data(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Profile the dataset to select appropriate models.
        
        Args:
            X: Data to profile
            feature_names: Names of features
            
        Returns:
            Dict with data profile information
        """
        profile = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "has_temporal_features": any('month' in f or 'year' in f or 'day' in f for f in feature_names),
            "has_geo_features": any('state' in f or 'district' in f or 'pincode' in f for f in feature_names),
            "variance": float(np.var(X)),
            "sparsity": float((X == 0).sum() / X.size),
            "feature_names": feature_names
        }
        
        # Determine complexity
        if profile["n_features"] > 20 or profile["variance"] > 100:
            profile["complexity"] = "high"
        elif profile["n_features"] > 10:
            profile["complexity"] = "medium"
        else:
            profile["complexity"] = "low"
        
        logger.info(f"Data profile: {profile['n_samples']} samples, {profile['n_features']} features, complexity: {profile['complexity']}")
        
        return profile
    
    def _select_models(self, profile: Dict[str, Any]) -> List[str]:
        """
        Select models based on data profile.
        
        Args:
            profile: Data profile from _profile_data
            
        Returns:
            List of model names to use
        """
        models = ["isolation_forest"]  # Always use as baseline
        
        # Use autoencoder for complex patterns
        if profile["complexity"] in ["medium", "high"] or profile["has_temporal_features"]:
            models.append("autoencoder")
        
        # Use HDBSCAN for spatial/clustering patterns
        if profile["has_geo_features"] or profile["n_samples"] > 100:
            models.append("hdbscan")
        
        logger.info(f"Selected models: {models}")
        return models
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'EnsembleScorer':
        """
        Fit all selected models on the data.
        
        Args:
            X: Training data
            feature_names: Names of features
            
        Returns:
            Self for method chaining
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Profile data and select models
        self.data_profile = self._profile_data(X, self.feature_names)
        selected_models = self._select_models(self.data_profile)
        
        logger.info(f"Training ensemble with models: {selected_models}")
        
        # Initialize and train selected models
        for model_name in selected_models:
            start_time = time.time()
            
            try:
                if model_name == "isolation_forest":
                    model = IsolationForestModel()
                    predictions, scores = model.fit_predict(X, self.feature_names)
                    threshold = 0.5  # Default threshold
                    metadata = {"contamination": model.contamination}
                    
                elif model_name == "autoencoder":
                    model = AutoencoderModel()
                    predictions, scores = model.fit_predict(X, self.feature_names)
                    threshold = model.threshold
                    metadata = {"latent_dim": model.latent_dim, "epochs": model.epochs}
                    
                elif model_name == "hdbscan":
                    model = ClusteringModel()
                    predictions, scores = model.fit_predict(X, self.feature_names)
                    threshold = 0.5
                    cluster_info = model.get_cluster_info()
                    metadata = cluster_info
                
                else:
                    continue
                
                execution_time = (time.time() - start_time) * 1000
                
                self.models[model_name] = model
                self.model_results[model_name] = ModelResult(
                    model_name=model_name,
                    predictions=predictions,
                    scores=scores,
                    threshold=threshold,
                    anomaly_count=int((predictions == -1).sum()),
                    execution_time_ms=execution_time,
                    metadata=metadata
                )
                
                logger.info(f"{model_name} training complete in {execution_time:.2f}ms")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble predictions using weighted voting.
        
        Args:
            X: Data to predict on
            
        Returns:
            Array of -1 (anomaly) or 1 (normal)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        scores = self.score_samples(X)
        settings = get_settings()
        
        # Use medium risk threshold for binary classification
        threshold = settings.medium_risk_threshold
        predictions = np.where(scores > threshold, -1, 1)
        
        return predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble anomaly scores using weighted average.
        
        Args:
            X: Data to score
            
        Returns:
            Array of ensemble anomaly scores (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before scoring")
        
        all_scores = []
        all_weights = []
        
        for model_name, model in self.models.items():
            try:
                scores = model.score_samples(X)
                weight = self.weights.get(model_name, 0.33)
                all_scores.append(scores)
                all_weights.append(weight)
            except Exception as e:
                logger.warning(f"Error scoring with {model_name}: {e}")
                continue
        
        if not all_scores:
            return np.zeros(len(X))
        
        # Weighted average of scores
        all_scores = np.array(all_scores)
        all_weights = np.array(all_weights)
        all_weights = all_weights / all_weights.sum()  # Normalize weights
        
        ensemble_scores = np.average(all_scores, axis=0, weights=all_weights)
        
        return ensemble_scores
    
    def fit_predict(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit ensemble and return predictions and scores.
        
        Args:
            X: Training data
            feature_names: Names of features
            
        Returns:
            Tuple of (predictions, scores)
        """
        self.fit(X, feature_names)
        
        # Use cached scores from fitting
        predictions = self.predict(X)
        scores = self.score_samples(X)
        
        anomaly_count = (predictions == -1).sum()
        logger.info(f"Ensemble detected {anomaly_count} anomalies ({100*anomaly_count/len(predictions):.2f}%)")
        
        return predictions, scores
    
    def get_model_results(self) -> List[Dict[str, Any]]:
        """Get results from all individual models."""
        results = []
        for name, result in self.model_results.items():
            results.append({
                "model_name": result.model_name,
                "anomaly_count": result.anomaly_count,
                "threshold": result.threshold,
                "execution_time_ms": result.execution_time_ms,
                "metadata": result.metadata
            })
        return results
    
    def get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """
        Get aggregated feature importance from all models.
        
        Args:
            X: Data for importance calculation
            
        Returns:
            Dict mapping feature names to importance scores
        """
        aggregated = {}
        count = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'get_feature_importance'):
                    importance = model.get_feature_importance(X)
                    for feature, value in importance.items():
                        aggregated[feature] = aggregated.get(feature, 0) + value
                        count[feature] = count.get(feature, 0) + 1
                elif hasattr(model, 'get_feature_reconstruction_errors'):
                    errors = model.get_feature_reconstruction_errors(X)
                    max_error = max(errors.values()) if errors else 1
                    for feature, value in errors.items():
                        aggregated[feature] = aggregated.get(feature, 0) + value / max_error
                        count[feature] = count.get(feature, 0) + 1
            except Exception as e:
                logger.warning(f"Error getting importance from {model_name}: {e}")
        
        # Average across models
        for feature in aggregated:
            if count.get(feature, 0) > 0:
                aggregated[feature] /= count[feature]
        
        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))
