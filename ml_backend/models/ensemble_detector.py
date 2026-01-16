"""
Ensemble Anomaly Detection Pipeline for UIDAI Fraud Detection

This module implements a production-grade ensemble anomaly detection system combining:
1. Isolation Forest - Fast, scalable, tree-based isolation
2. HDBSCAN - Density-based clustering for spatial anomalies
3. Autoencoder - Deep learning for complex pattern recognition
4. One-Class SVM (Optional) - Maximum margin outlier detection

Each model captures different types of anomalies, and their scores are combined
using weighted voting to produce a robust, unified anomaly score.

Author: UIDAI Fraud Detection Team
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings
from abc import ABC, abstractmethod

# ML Libraries
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
import joblib

# Deep Learning
import torch
import torch.nn as nn

# Clustering
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not available. Install with: pip install hdbscan")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class AnomalyLabel(Enum):
    """Classification labels for detected anomalies."""
    NORMAL = "Normal"
    SUSPICIOUS = "Suspicious"
    HIGHLY_SUSPICIOUS = "Highly Suspicious"


@dataclass
class ModelConfig:
    """Configuration for individual models in the ensemble."""
    enabled: bool = True
    weight: float = 0.25
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    """Configuration for the entire ensemble."""
    # Model weights (must sum to 1.0 for enabled models)
    isolation_forest: ModelConfig = field(default_factory=lambda: ModelConfig(
        weight=0.35,
        params={'contamination': 0.05, 'n_estimators': 100, 'random_state': 42}
    ))
    hdbscan: ModelConfig = field(default_factory=lambda: ModelConfig(
        weight=0.25,
        params={'min_cluster_size': 15, 'min_samples': 5}
    ))
    autoencoder: ModelConfig = field(default_factory=lambda: ModelConfig(
        weight=0.30,
        params={'latent_dim': 8, 'epochs': 50, 'batch_size': 64}
    ))
    one_class_svm: ModelConfig = field(default_factory=lambda: ModelConfig(
        enabled=False,  # Optional, disabled by default (slow)
        weight=0.10,
        params={'nu': 0.05, 'kernel': 'rbf', 'gamma': 'scale'}
    ))
    
    # Threshold selection
    normal_threshold: float = 0.3       # Below this = Normal
    suspicious_threshold: float = 0.7    # Above this = Highly Suspicious
    # Between thresholds = Suspicious
    
    # Concept drift detection
    drift_window_size: int = 1000
    drift_threshold: float = 0.1


# =============================================================================
# MODEL EXPLANATIONS
# =============================================================================

MODEL_EXPLANATIONS = """
================================================================================
WHY EACH MODEL IS USED IN THE ENSEMBLE
================================================================================

1. ISOLATION FOREST
   ----------------
   WHY: Based on the principle that anomalies are "few and different."
        Trees can isolate anomalies with fewer splits than normal points.
   
   FRAUD PATTERNS DETECTED:
   - Volume spikes (operators processing too many transactions)
   - Statistical outliers in any feature
   - Records that don't conform to typical patterns
   
   PROS:
   + Fast training (O(n log n))
   + Handles high-dimensional data well
   + No distance metric required
   + Works without assuming data distribution
   
   CONS:
   - May miss local anomalies in dense regions
   - Requires tuning contamination parameter
   
   WEIGHT: 35% (Primary detector, fast and reliable)

--------------------------------------------------------------------------------

2. HDBSCAN (Hierarchical Density-Based Spatial Clustering)
   -------------------------------------------------------
   WHY: Identifies noise points that don't belong to any cluster.
        Uses varying density, perfect for geographic/behavioral clusters.
   
   FRAUD PATTERNS DETECTED:
   - Geographic anomalies (activity in unusual locations)
   - Behavioral outliers (operators not fitting patterns)
   - Sparse region activities (isolated fraudulent clusters)
   
   PROS:
   + No need to specify number of clusters
   + Handles varying density clusters
   + Robust to noise
   + Provides outlier scores
   
   CONS:
   - Slower on large datasets
   - Sensitive to min_cluster_size parameter
   
   WEIGHT: 25% (Spatial/density patterns)

--------------------------------------------------------------------------------

3. AUTOENCODER (Deep Learning)
   ---------------------------
   WHY: Learns compressed representation of "normal" data.
        Anomalies have high reconstruction error.
   
   FRAUD PATTERNS DETECTED:
   - Complex, non-linear patterns
   - Correlated feature anomalies
   - Subtle deviations across multiple features
   - Novel fraud patterns not seen before
   
   ARCHITECTURE:
   Input → Encoder → Latent Space → Decoder → Reconstruction
   
   PROS:
   + Captures complex, non-linear relationships
   + Learns representation automatically
   + Can detect novel anomalies
   + No assumptions about data distribution
   
   CONS:
   - Requires more data for training
   - Computationally expensive
   - May overfit on small datasets
   
   WEIGHT: 30% (Complex pattern detection)

--------------------------------------------------------------------------------

4. ONE-CLASS SVM (Optional)
   ------------------------
   WHY: Finds the maximum-margin hyperplane that separates
        normal data from the origin in feature space.
   
   FRAUD PATTERNS DETECTED:
   - Boundary violations
   - Margin-based outliers
   - Support vector defined anomalies
   
   PROS:
   + Strong theoretical foundation
   + Works well with clear boundaries
   + Kernel trick for non-linear boundaries
   
   CONS:
   - Very slow on large datasets (O(n²) to O(n³))
   - Memory intensive
   - Sensitive to kernel/gamma parameters
   
   WEIGHT: 10% (Optional, disabled by default due to computational cost)

================================================================================
THRESHOLD SELECTION STRATEGY
================================================================================

We use a THREE-TIER classification system:

1. NORMAL (score < 0.3):
   - Low anomaly probability
   - No further investigation needed
   - ~90-95% of legitimate records
   
2. SUSPICIOUS (0.3 ≤ score < 0.7):
   - Moderate anomaly signals
   - Flagged for review
   - May require additional verification
   - ~3-8% of records
   
3. HIGHLY SUSPICIOUS (score ≥ 0.7):
   - Strong anomaly signals
   - Multiple models agree on anomaly
   - Immediate investigation required
   - ~1-3% of records

THRESHOLD TUNING:
- Use validation set with known fraud cases
- Optimize for business requirements (precision vs recall)
- Consider cost of false positives vs false negatives
- Use percentile-based thresholds for adaptive selection

================================================================================
HANDLING CONCEPT DRIFT
================================================================================

Concept drift occurs when the statistical properties of the data change
over time. In fraud detection, this is CRITICAL because:
- Legitimate behavior evolves (new enrollment patterns)
- Fraudsters adapt their techniques
- Seasonal variations affect patterns

OUR APPROACH:

1. STATISTICAL DRIFT DETECTION:
   - Monitor mean/variance of anomaly scores over sliding windows
   - Alert when distribution shifts significantly (KL divergence)
   - Track feature importance changes

2. MODEL RETRAINING TRIGGERS:
   - Periodic retraining (weekly/monthly)
   - Drift-triggered retraining when threshold exceeded
   - Incremental learning where possible

3. ENSEMBLE ADAPTATION:
   - Adjust model weights based on recent performance
   - Increase weight of models performing better on recent data
   - Add new models for emerging patterns

4. IMPLEMENTATION:
   ```python
   class DriftDetector:
       def detect_drift(self, recent_scores, historical_scores):
           # Page-Hinkley test or ADWIN algorithm
           drift_magnitude = abs(np.mean(recent_scores) - np.mean(historical_scores))
           return drift_magnitude > self.threshold
   ```

================================================================================
COMPUTATIONAL TRADE-OFFS
================================================================================

| Model          | Training Time | Inference Time | Memory    | Scalability |
|----------------|---------------|----------------|-----------|-------------|
| Isolation Forest| O(n log n)   | O(log n)       | Low       | Excellent   |
| HDBSCAN        | O(n log n)    | O(n)           | Medium    | Good        |
| Autoencoder    | O(n * epochs) | O(n)           | High      | Good (GPU)  |
| One-Class SVM  | O(n² - n³)    | O(n)           | Very High | Poor        |

RECOMMENDATIONS:
- For real-time: Use Isolation Forest only (< 1ms inference)
- For batch: Full ensemble (seconds per batch)
- For large data (>1M): Skip One-Class SVM, sample for Autoencoder
- GPU available: Prioritize Autoencoder

================================================================================
"""


# =============================================================================
# BASE MODEL INTERFACE
# =============================================================================

class BaseAnomalyModel(ABC):
    """Abstract base class for all anomaly detection models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseAnomalyModel':
        """Fit the model on training data."""
        pass
    
    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (0-1, higher = more anomalous)."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions: -1 for anomaly, 1 for normal."""
        pass


# =============================================================================
# ISOLATION FOREST MODEL
# =============================================================================

class IsolationForestDetector(BaseAnomalyModel):
    """
    Isolation Forest for anomaly detection.
    
    Key Insight: Anomalies are easier to isolate (fewer tree splits)
    because they are "few and different."
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit isolation forest on data."""
        logger.info(f"Training Isolation Forest with {self.n_estimators} trees...")
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            warm_start=False
        )
        self.model.fit(X)
        self.is_fitted = True
        
        logger.info("Isolation Forest training complete")
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores normalized to [0, 1].
        
        Original scores are negative (more negative = more anomalous).
        We convert to [0, 1] where 1 = highly anomalous.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        # Original scores: lower (more negative) = more anomalous
        raw_scores = self.model.score_samples(X)
        
        # Convert to [0, 1]: higher = more anomalous
        # Typical range is [-0.5, 0.5], we shift and scale
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        
        if max_score - min_score > 0:
            normalized = (max_score - raw_scores) / (max_score - min_score)
        else:
            normalized = np.zeros_like(raw_scores)
            
        return np.clip(normalized, 0, 1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict: -1 for anomaly, 1 for normal."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.predict(X)


# =============================================================================
# HDBSCAN MODEL
# =============================================================================

class HDBSCANDetector(BaseAnomalyModel):
    """
    HDBSCAN for density-based anomaly detection.
    
    Key Insight: Points that don't belong to any cluster (noise points)
    are potential anomalies. Outlier scores indicate how far from clusters.
    """
    
    def __init__(
        self,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        cluster_selection_epsilon: float = 0.0
    ):
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
            
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'HDBSCANDetector':
        """Fit HDBSCAN clustering."""
        logger.info(f"Training HDBSCAN with min_cluster_size={self.min_cluster_size}...")
        
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=True,
            gen_min_span_tree=True
        )
        self.model.fit(X)
        self.is_fitted = True
        
        n_clusters = len(set(self.model.labels_)) - (1 if -1 in self.model.labels_ else 0)
        n_noise = (self.model.labels_ == -1).sum()
        logger.info(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points")
        
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get outlier scores from HDBSCAN.
        
        Uses the outlier_scores_ attribute which is in [0, 1].
        Higher scores = more likely to be outlier.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        # HDBSCAN provides outlier_scores_ in [0, 1]
        if hasattr(self.model, 'outlier_scores_') and self.model.outlier_scores_ is not None:
            scores = self.model.outlier_scores_
        else:
            # Fallback: use cluster membership probability
            scores = 1 - self.model.probabilities_
            
        return np.clip(scores, 0, 1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict: -1 for anomaly (noise), 1 for normal (in cluster)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        labels = self.model.labels_
        predictions = np.where(labels == -1, -1, 1)
        return predictions
    
    def get_cluster_labels(self) -> np.ndarray:
        """Get cluster assignments."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.labels_


# =============================================================================
# AUTOENCODER MODEL
# =============================================================================

class Autoencoder(nn.Module):
    """
    Deep Autoencoder for anomaly detection.
    
    Architecture:
    - Encoder: Compress input to latent space
    - Decoder: Reconstruct from latent space
    - Anomalies have high reconstruction error
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        
        # Calculate intermediate dimensions
        hidden1 = max(32, input_dim // 2)
        hidden2 = max(16, hidden1 // 2)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, input_dim),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class AutoencoderDetector(BaseAnomalyModel):
    """
    Autoencoder-based anomaly detection.
    
    Key Insight: Train to reconstruct normal data. Anomalies
    will have high reconstruction error because they differ
    from the learned normal patterns.
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        threshold_percentile: float = 95
    ):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = None
        self.is_fitted = False
        self.scaler = MinMaxScaler()
        
    def fit(self, X: np.ndarray) -> 'AutoencoderDetector':
        """Train autoencoder on data."""
        logger.info(f"Training Autoencoder ({self.epochs} epochs, device={self.device})...")
        
        input_dim = X.shape[1]
        
        # Initialize model
        self.model = Autoencoder(input_dim, self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                
                optimizer.zero_grad()
                reconstructed = self.model(x)
                loss = criterion(reconstructed, x)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.6f}")
        
        # Calculate threshold from training data reconstruction errors
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            self.threshold = np.percentile(errors, self.threshold_percentile)
        
        # Fit scaler for error normalization
        self.scaler.fit(errors.reshape(-1, 1))
        
        self.is_fitted = True
        logger.info(f"Autoencoder training complete (threshold={self.threshold:.6f})")
        
        return self
    
    def _get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Calculate reconstruction errors."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            
        return errors
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores from reconstruction error.
        
        Higher reconstruction error = higher anomaly score.
        Normalized to [0, 1].
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        errors = self._get_reconstruction_errors(X)
        
        # Normalize to [0, 1]
        scores = self.scaler.transform(errors.reshape(-1, 1)).flatten()
        return np.clip(scores, 0, 1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict: -1 for anomaly, 1 for normal."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        errors = self._get_reconstruction_errors(X)
        predictions = np.where(errors > self.threshold, -1, 1)
        return predictions


# =============================================================================
# ONE-CLASS SVM MODEL (Optional)
# =============================================================================

class OneClassSVMDetector(BaseAnomalyModel):
    """
    One-Class SVM for anomaly detection.
    
    Key Insight: Finds a maximum-margin hyperplane that separates
    normal data from the origin. Points on the wrong side are anomalies.
    
    WARNING: Very slow on large datasets (O(n²) - O(n³)).
    Use only for small datasets or critical applications.
    """
    
    def __init__(
        self,
        nu: float = 0.05,
        kernel: str = 'rbf',
        gamma: str = 'scale'
    ):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'OneClassSVMDetector':
        """Fit One-Class SVM."""
        logger.info(f"Training One-Class SVM (kernel={self.kernel}, nu={self.nu})...")
        logger.warning("One-Class SVM is slow on large datasets. Consider disabling for large data.")
        
        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma
        )
        self.model.fit(X)
        self.is_fitted = True
        
        logger.info("One-Class SVM training complete")
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores from SVM decision function.
        
        Negative decision values = anomalies.
        Normalized to [0, 1] where higher = more anomalous.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        # Decision function: negative = anomaly
        decision = self.model.decision_function(X)
        
        # Convert to [0, 1]: higher = more anomalous
        min_dec = decision.min()
        max_dec = decision.max()
        
        if max_dec - min_dec > 0:
            normalized = (max_dec - decision) / (max_dec - min_dec)
        else:
            normalized = np.zeros_like(decision)
            
        return np.clip(normalized, 0, 1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict: -1 for anomaly, 1 for normal."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.predict(X)


# =============================================================================
# CONCEPT DRIFT DETECTOR
# =============================================================================

class ConceptDriftDetector:
    """
    Detects concept drift in anomaly scores over time.
    
    Uses sliding window comparison to detect when the
    distribution of anomaly scores changes significantly.
    """
    
    def __init__(self, window_size: int = 1000, threshold: float = 0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.historical_scores: List[float] = []
        self.drift_history: List[Dict] = []
        
    def update(self, scores: np.ndarray) -> bool:
        """
        Update with new scores and detect drift.
        
        Returns True if drift is detected.
        """
        self.historical_scores.extend(scores.tolist())
        
        # Keep only recent history
        if len(self.historical_scores) > self.window_size * 2:
            self.historical_scores = self.historical_scores[-self.window_size * 2:]
        
        # Need enough historical data
        if len(self.historical_scores) < self.window_size * 2:
            return False
        
        # Compare recent window to historical window
        historical = np.array(self.historical_scores[:self.window_size])
        recent = np.array(self.historical_scores[-self.window_size:])
        
        # Calculate drift magnitude using mean and std difference
        mean_diff = abs(np.mean(recent) - np.mean(historical))
        std_diff = abs(np.std(recent) - np.std(historical))
        
        drift_magnitude = (mean_diff + std_diff) / 2
        
        if drift_magnitude > self.threshold:
            self.drift_history.append({
                'timestamp': time.time(),
                'magnitude': drift_magnitude,
                'historical_mean': float(np.mean(historical)),
                'recent_mean': float(np.mean(recent))
            })
            logger.warning(f"Concept drift detected! Magnitude: {drift_magnitude:.4f}")
            return True
            
        return False
    
    def get_drift_history(self) -> List[Dict]:
        """Return history of detected drifts."""
        return self.drift_history


# =============================================================================
# ENSEMBLE ANOMALY DETECTOR
# =============================================================================

class EnsembleAnomalyDetector:
    """
    Production-grade ensemble anomaly detection system.
    
    Combines multiple models using weighted voting to produce
    robust, normalized anomaly scores with confidence labels.
    
    Usage:
        detector = EnsembleAnomalyDetector()
        detector.fit(X_train)
        scores, labels = detector.predict(X_test)
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        Initialize ensemble detector.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or EnsembleConfig()
        self.models: Dict[str, BaseAnomalyModel] = {}
        self.is_fitted = False
        self.training_time_ms: Dict[str, float] = {}
        self.drift_detector = ConceptDriftDetector(
            window_size=self.config.drift_window_size,
            threshold=self.config.drift_threshold
        )
        
    def _normalize_weights(self) -> Dict[str, float]:
        """Normalize weights of enabled models to sum to 1."""
        weights = {}
        
        if self.config.isolation_forest.enabled:
            weights['isolation_forest'] = self.config.isolation_forest.weight
        if self.config.hdbscan.enabled:
            weights['hdbscan'] = self.config.hdbscan.weight
        if self.config.autoencoder.enabled:
            weights['autoencoder'] = self.config.autoencoder.weight
        if self.config.one_class_svm.enabled:
            weights['one_class_svm'] = self.config.one_class_svm.weight
            
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
    
    def fit(self, X: np.ndarray) -> 'EnsembleAnomalyDetector':
        """
        Fit all enabled models in the ensemble.
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting ensemble on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Train Isolation Forest
        if self.config.isolation_forest.enabled:
            start = time.time()
            self.models['isolation_forest'] = IsolationForestDetector(
                **self.config.isolation_forest.params
            ).fit(X)
            self.training_time_ms['isolation_forest'] = (time.time() - start) * 1000
            
        # Train HDBSCAN
        if self.config.hdbscan.enabled and HDBSCAN_AVAILABLE:
            start = time.time()
            self.models['hdbscan'] = HDBSCANDetector(
                **self.config.hdbscan.params
            ).fit(X)
            self.training_time_ms['hdbscan'] = (time.time() - start) * 1000
            
        # Train Autoencoder
        if self.config.autoencoder.enabled:
            start = time.time()
            self.models['autoencoder'] = AutoencoderDetector(
                **self.config.autoencoder.params
            ).fit(X)
            self.training_time_ms['autoencoder'] = (time.time() - start) * 1000
            
        # Train One-Class SVM (optional, slow)
        if self.config.one_class_svm.enabled:
            # Only use SVM on smaller datasets
            if X.shape[0] > 10000:
                logger.warning("Skipping One-Class SVM due to large dataset size")
            else:
                start = time.time()
                self.models['one_class_svm'] = OneClassSVMDetector(
                    **self.config.one_class_svm.params
                ).fit(X)
                self.training_time_ms['one_class_svm'] = (time.time() - start) * 1000
        
        self.is_fitted = True
        total_time = sum(self.training_time_ms.values())
        logger.info(f"Ensemble training complete in {total_time:.2f}ms")
        
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get combined anomaly scores from all models.
        
        Scores are weighted average, normalized to [0, 1].
        Higher scores = more anomalous.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
            
        weights = self._normalize_weights()
        all_scores = []
        all_weights = []
        
        for name, model in self.models.items():
            if name in weights:
                try:
                    scores = model.score_samples(X)
                    all_scores.append(scores)
                    all_weights.append(weights[name])
                except Exception as e:
                    logger.warning(f"Error getting scores from {name}: {e}")
                    
        if not all_scores:
            return np.zeros(len(X))
            
        # Weighted average
        all_scores = np.array(all_scores)
        all_weights = np.array(all_weights)
        all_weights = all_weights / all_weights.sum()  # Renormalize
        
        ensemble_scores = np.average(all_scores, axis=0, weights=all_weights)
        
        # Update drift detector
        self.drift_detector.update(ensemble_scores)
        
        return ensemble_scores
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, List[AnomalyLabel]]:
        """
        Predict anomaly scores and labels.
        
        Returns:
            Tuple of (scores, labels) where:
            - scores: float array in [0, 1]
            - labels: list of AnomalyLabel enums
        """
        scores = self.score_samples(X)
        labels = self._assign_labels(scores)
        return scores, labels
    
    def _assign_labels(self, scores: np.ndarray) -> List[AnomalyLabel]:
        """
        Assign categorical labels based on scores.
        
        Uses configured thresholds:
        - score < normal_threshold → Normal
        - normal_threshold ≤ score < suspicious_threshold → Suspicious
        - score ≥ suspicious_threshold → Highly Suspicious
        """
        labels = []
        for score in scores:
            if score < self.config.normal_threshold:
                labels.append(AnomalyLabel.NORMAL)
            elif score < self.config.suspicious_threshold:
                labels.append(AnomalyLabel.SUSPICIOUS)
            else:
                labels.append(AnomalyLabel.HIGHLY_SUSPICIOUS)
        return labels
    
    def get_model_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get individual scores from each model for analysis."""
        scores = {}
        for name, model in self.models.items():
            try:
                scores[name] = model.score_samples(X)
            except Exception as e:
                logger.warning(f"Error getting scores from {name}: {e}")
        return scores
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics and model information."""
        return {
            'models_enabled': list(self.models.keys()),
            'model_weights': self._normalize_weights(),
            'training_time_ms': self.training_time_ms,
            'thresholds': {
                'normal': self.config.normal_threshold,
                'suspicious': self.config.suspicious_threshold
            },
            'drift_history': self.drift_detector.get_drift_history()
        }
    
    def save(self, path: str):
        """Save the ensemble to disk."""
        joblib.dump({
            'config': self.config,
            'models': self.models,
            'training_time_ms': self.training_time_ms
        }, path)
        logger.info(f"Ensemble saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'EnsembleAnomalyDetector':
        """Load an ensemble from disk."""
        data = joblib.load(path)
        ensemble = cls(config=data['config'])
        ensemble.models = data['models']
        ensemble.training_time_ms = data['training_time_ms']
        ensemble.is_fitted = True
        logger.info(f"Ensemble loaded from {path}")
        return ensemble


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_ensemble(
    enable_svm: bool = False,
    contamination: float = 0.05
) -> EnsembleAnomalyDetector:
    """
    Create an ensemble with sensible defaults.
    
    Args:
        enable_svm: Enable One-Class SVM (slow but accurate)
        contamination: Expected proportion of anomalies
        
    Returns:
        Configured EnsembleAnomalyDetector
    """
    config = EnsembleConfig()
    config.isolation_forest.params['contamination'] = contamination
    config.one_class_svm.enabled = enable_svm
    config.one_class_svm.params['nu'] = contamination
    
    return EnsembleAnomalyDetector(config)


def get_model_explanations() -> str:
    """Return detailed explanations for all models."""
    return MODEL_EXPLANATIONS


# =============================================================================
# MAIN / EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Print model explanations
    print(MODEL_EXPLANATIONS)
    
    # Example usage
    print("\n" + "="*80)
    print("EXAMPLE USAGE")
    print("="*80)
    
    # Generate synthetic data
    np.random.seed(42)
    n_normal = 1000
    n_anomaly = 50
    n_features = 10
    
    # Normal data
    X_normal = np.random.randn(n_normal, n_features)
    
    # Anomalous data (shifted mean)
    X_anomaly = np.random.randn(n_anomaly, n_features) + 3
    
    X = np.vstack([X_normal, X_anomaly])
    y_true = np.array([0] * n_normal + [1] * n_anomaly)  # 0=normal, 1=anomaly
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True anomalies: {n_anomaly} ({100*n_anomaly/len(X):.1f}%)")
    
    # Create and fit ensemble
    ensemble = create_ensemble(enable_svm=False)
    ensemble.fit(X)
    
    # Predict
    scores, labels = ensemble.predict(X)
    
    # Evaluate
    highly_suspicious = sum(1 for l in labels if l == AnomalyLabel.HIGHLY_SUSPICIOUS)
    suspicious = sum(1 for l in labels if l == AnomalyLabel.SUSPICIOUS)
    normal = sum(1 for l in labels if l == AnomalyLabel.NORMAL)
    
    print(f"\nResults:")
    print(f"  Normal: {normal}")
    print(f"  Suspicious: {suspicious}")
    print(f"  Highly Suspicious: {highly_suspicious}")
    print(f"\nStatistics: {ensemble.get_statistics()}")
