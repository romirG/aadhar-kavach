"""
PyTorch Autoencoder model for deep anomaly detection.
"""
import logging
import numpy as np
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, '..')

from config import get_settings

logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AutoencoderNetwork(nn.Module):
    """PyTorch Autoencoder architecture."""
    
    def __init__(self, input_dim: int, latent_dim: int = 16):
        """
        Initialize autoencoder network.
        
        Args:
            input_dim: Number of input features
            latent_dim: Dimension of latent space
        """
        super().__init__()
        
        # Calculate hidden dimensions
        hidden1 = max(64, input_dim * 2)
        hidden2 = max(32, input_dim)
        
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
            nn.ReLU()
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
            nn.Linear(hidden1, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and latent representation."""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)


class AutoencoderModel:
    """
    Autoencoder for deep anomaly detection.
    
    Detects anomalies based on reconstruction error - 
    anomalous patterns are harder to reconstruct.
    
    Effective for:
    - Complex non-linear patterns
    - Subtle demographic inconsistencies
    - Multi-dimensional anomalies
    """
    
    def __init__(
        self,
        latent_dim: Optional[int] = None,
        epochs: Optional[int] = None,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ):
        """
        Initialize Autoencoder model.
        
        Args:
            latent_dim: Dimension of latent space
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        settings = get_settings()
        self.latent_dim = latent_dim or settings.autoencoder_latent_dim
        self.epochs = epochs or settings.autoencoder_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.model: Optional[AutoencoderNetwork] = None
        self.threshold: float = 0.0
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.training_history: List[float] = []
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'AutoencoderModel':
        """
        Train the autoencoder.
        
        Args:
            X: Training data (n_samples, n_features)
            feature_names: Names of features
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Autoencoder on {X.shape[0]} samples with {X.shape[1]} features")
        
        input_dim = X.shape[1]
        self.feature_names = feature_names or [f"feature_{i}" for i in range(input_dim)]
        
        # Initialize network
        self.model = AutoencoderNetwork(input_dim, self.latent_dim).to(device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate,
                              weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        self.model.train()
        self.training_history = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                reconstruction, _ = self.model(batch_x)
                loss = criterion(reconstruction, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.training_history.append(avg_loss)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        # Calculate threshold based on training reconstruction errors
        self.model.eval()
        with torch.no_grad():
            reconstruction, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstruction) ** 2, dim=1).cpu().numpy()
        
        # Set threshold at 95th percentile of reconstruction errors
        self.threshold = float(np.percentile(errors, 95))
        self.is_fitted = True
        
        logger.info(f"Autoencoder training complete. Threshold: {self.threshold:.6f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels based on reconstruction error.
        
        Args:
            X: Data to predict on
            
        Returns:
            Array of -1 (anomaly) or 1 (normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        errors = self._get_reconstruction_errors(X)
        predictions = np.where(errors > self.threshold, -1, 1)
        return predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores based on reconstruction error.
        
        Args:
            X: Data to score
            
        Returns:
            Array of normalized anomaly scores (0-1, higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        errors = self._get_reconstruction_errors(X)
        
        # Normalize to 0-1 range
        min_error = errors.min()
        max_error = errors.max()
        
        if max_error - min_error > 0:
            normalized_scores = (errors - min_error) / (max_error - min_error)
        else:
            normalized_scores = np.zeros_like(errors)
        
        return normalized_scores
    
    def _get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Calculate reconstruction errors for samples."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            reconstruction, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstruction) ** 2, dim=1)
        
        return errors.cpu().numpy()
    
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
        predictions = self.predict(X)
        scores = self.score_samples(X)
        
        anomaly_count = (predictions == -1).sum()
        logger.info(f"Autoencoder detected {anomaly_count} anomalies ({100*anomaly_count/len(predictions):.2f}%)")
        
        return predictions, scores
    
    def get_feature_reconstruction_errors(self, X: np.ndarray) -> dict:
        """
        Get per-feature reconstruction errors for explainability.
        
        Args:
            X: Data to analyze
            
        Returns:
            Dict mapping feature names to average reconstruction errors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            reconstruction, _ = self.model(X_tensor)
            feature_errors = torch.mean((X_tensor - reconstruction) ** 2, dim=0)
        
        errors = feature_errors.cpu().numpy()
        
        return {
            name: float(error) 
            for name, error in zip(self.feature_names, errors)
        }
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """Get latent space representation of data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            latent = self.model.encode(X_tensor)
        
        return latent.cpu().numpy()
    
    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state': self.model.state_dict(),
            'threshold': self.threshold,
            'latent_dim': self.latent_dim,
            'feature_names': self.feature_names,
            'input_dim': len(self.feature_names)
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=device)
        self.model = AutoencoderNetwork(
            checkpoint['input_dim'], 
            checkpoint['latent_dim']
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.threshold = checkpoint['threshold']
        self.feature_names = checkpoint['feature_names']
        self.latent_dim = checkpoint['latent_dim']
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
