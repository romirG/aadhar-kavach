"""
Deep Autoencoder for Aadhaar Fraud Detection

This module implements a production-grade Autoencoder-based anomaly detection
system specifically designed for detecting fraudulent patterns in Aadhaar
enrolment and update data.

================================================================================
WHY AUTOENCODERS ARE SUITABLE FOR AADHAAR FRAUD DETECTION
================================================================================

1. UNSUPERVISED LEARNING
   - No labeled fraud data required (fraud is rare and evolving)
   - Learns from the vast majority of legitimate transactions
   - Can detect novel fraud patterns never seen before

2. RECONSTRUCTION-BASED DETECTION
   - Trains to reconstruct "normal" behavior patterns
   - Fraudulent records have high reconstruction error
   - Natural threshold: what can't be reconstructed is anomalous

3. COMPLEX PATTERN RECOGNITION
   - Captures non-linear relationships between features
   - Models correlations across temporal, geographic, behavioral data
   - Learns hierarchical feature representations

4. DIMENSIONALITY REDUCTION
   - Compresses high-dimensional Aadhaar data to latent space
   - Latent space captures essential patterns
   - Anomalies deviate from learned manifold

5. SPECIFIC AADHAAR USE CASES:
   - Unusual time-location combinations for operators
   - Abnormal age distribution patterns
   - Suspicious update frequencies per Aadhaar ID
   - Geographic concentration anomalies
   - Cross-feature correlation violations

================================================================================
ARCHITECTURE DESIGN
================================================================================

Input Layer (n_features)
       ↓
Encoder Block 1: Linear → BatchNorm → ReLU → Dropout
       ↓
Encoder Block 2: Linear → BatchNorm → ReLU → Dropout  
       ↓
Encoder Block 3: Linear → BatchNorm → ReLU
       ↓
Latent Space (latent_dim) ← Bottleneck: compressed representation
       ↓
Decoder Block 1: Linear → BatchNorm → ReLU → Dropout
       ↓
Decoder Block 2: Linear → BatchNorm → ReLU → Dropout
       ↓
Decoder Block 3: Linear → BatchNorm → ReLU  
       ↓
Output Layer (n_features) ← Reconstruction

Loss: Mean Squared Error (MSE) between input and reconstruction
Anomaly Score: Reconstruction error (higher = more anomalous)

================================================================================
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# HYPERPARAMETERS CONFIGURATION
# =============================================================================

@dataclass
class AutoencoderConfig:
    """
    Hyperparameter configuration for the Autoencoder.
    
    HYPERPARAMETER CHOICES EXPLAINED:
    
    latent_dim (8):
        - Compression target: ~10-20% of input features
        - Too small: loses important patterns
        - Too large: doesn't compress enough, overfits
        - 8 is good for 30-50 input features
    
    hidden_dims ([64, 32]):
        - Progressive compression in encoder
        - Mirror structure in decoder
        - Powers of 2 for GPU efficiency
    
    dropout (0.2):
        - Regularization to prevent overfitting
        - Higher for smaller datasets
        - 0.2 is a good default
    
    epochs (100):
        - More for complex patterns
        - Use early stopping to prevent overfitting
        - 50-100 usually sufficient
    
    batch_size (64):
        - Larger = faster, less noise
        - Smaller = better generalization
        - 32-128 typical range
    
    learning_rate (0.001):
        - Adam default, works well for autoencoders
        - Reduce if loss oscillates
        - Use scheduler for better convergence
    
    threshold_percentile (95):
        - Top 5% by reconstruction error = anomaly
        - Adjust based on expected fraud rate
        - Higher = more conservative (fewer false positives)
    """
    # Architecture
    latent_dim: int = 8
    hidden_dims: List[int] = None  # Will default to [64, 32]
    dropout: float = 0.2
    use_batch_norm: bool = True
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    min_delta: float = 1e-4
    
    # Anomaly detection
    threshold_percentile: float = 95.0
    
    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# AUTOENCODER ARCHITECTURE
# =============================================================================

class EncoderBlock(nn.Module):
    """Single encoder block with Linear → BatchNorm → ReLU → Dropout."""
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
            
        layers.append(nn.ReLU())
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Single decoder block with Linear → BatchNorm → ReLU → Dropout."""
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        final_layer: bool = False
    ):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        
        if not final_layer:
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)


class AadhaarAutoencoder(nn.Module):
    """
    Deep Autoencoder for Aadhaar Fraud Detection.
    
    Architecture:
    - Symmetric encoder-decoder structure
    - Bottleneck latent space for compression
    - BatchNorm for training stability
    - Dropout for regularization
    
    The encoder compresses input features into a low-dimensional
    latent representation. The decoder reconstructs the original
    input from this representation. Anomalies have high reconstruction
    error because they don't fit the learned normal patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        config: AutoencoderConfig
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.config = config
        self.latent_dim = config.latent_dim
        
        # Build encoder
        encoder_layers = []
        dims = [input_dim] + config.hidden_dims + [config.latent_dim]
        
        for i in range(len(dims) - 1):
            is_last = (i == len(dims) - 2)
            encoder_layers.append(EncoderBlock(
                dims[i], 
                dims[i + 1],
                dropout=0 if is_last else config.dropout,
                use_batch_norm=config.use_batch_norm
            ))
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        decoder_dims = [config.latent_dim] + config.hidden_dims[::-1] + [input_dim]
        
        for i in range(len(decoder_dims) - 1):
            is_last = (i == len(decoder_dims) - 2)
            decoder_layers.append(DecoderBlock(
                decoder_dims[i],
                decoder_dims[i + 1],
                dropout=0 if is_last else config.dropout,
                use_batch_norm=config.use_batch_norm,
                final_layer=is_last
            ))
            
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Created Autoencoder: {input_dim} → {config.hidden_dims} → "
                   f"{config.latent_dim} → {config.hidden_dims[::-1]} → {input_dim}")
        
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate per-sample reconstruction error (MSE)."""
        reconstruction = self.forward(x)
        error = torch.mean((x - reconstruction) ** 2, dim=1)
        return error


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


# =============================================================================
# ANOMALY DETECTOR CLASS
# =============================================================================

class AutoencoderAnomalyDetector:
    """
    Complete Autoencoder-based anomaly detection system.
    
    This class provides:
    - Data preprocessing and scaling
    - Model training with early stopping
    - Anomaly scoring based on reconstruction error
    - Threshold selection (percentile-based or manual)
    - Visualization of results
    - Model persistence (save/load)
    
    Usage:
        detector = AutoencoderAnomalyDetector()
        detector.fit(X_train)
        scores = detector.score_samples(X_test)
        predictions = detector.predict(X_test)
        detector.plot_reconstruction_error()
    """
    
    def __init__(self, config: Optional[AutoencoderConfig] = None):
        """
        Initialize the detector.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or AutoencoderConfig()
        self.model: Optional[AadhaarAutoencoder] = None
        self.scaler = StandardScaler()
        self.device = torch.device(self.config.device)
        self.threshold: Optional[float] = None
        self.training_errors: Optional[np.ndarray] = None
        self.history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}
        self.is_fitted = False
        
        logger.info(f"Initialized AutoencoderAnomalyDetector (device={self.device})")
        
    def fit(
        self,
        X: np.ndarray,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> 'AutoencoderAnomalyDetector':
        """
        Train the autoencoder on normal data.
        
        Args:
            X: Training data (n_samples, n_features)
            validation_split: Fraction for validation
            verbose: Print training progress
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into train/validation
        X_train, X_val = train_test_split(
            X_scaled, 
            test_size=validation_split, 
            random_state=42
        )
        
        logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)}")
        
        # Create model
        input_dim = X.shape[1]
        self.model = AadhaarAutoencoder(input_dim, self.config).to(self.device)
        
        # Prepare data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val))
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Optimizer and loss
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.min_delta
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                reconstruction = self.model(x)
                loss = criterion(reconstruction, x)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(x)
                
            train_loss /= len(X_train)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(self.device)
                    reconstruction = self.model(x)
                    loss = criterion(reconstruction, x)
                    val_loss += loss.item() * len(x)
                    
            val_loss /= len(X_val)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                           f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
            # Early stopping
            if early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        # Calculate threshold from training data
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            self.training_errors = self.model.get_reconstruction_error(X_tensor).cpu().numpy()
            
        self.threshold = np.percentile(
            self.training_errors, 
            self.config.threshold_percentile
        )
        
        self.is_fitted = True
        training_time = time.time() - start_time
        
        logger.info(f"Training complete in {training_time:.2f}s")
        logger.info(f"Threshold ({self.config.threshold_percentile}th percentile): {self.threshold:.6f}")
        
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (reconstruction errors) for samples.
        
        Args:
            X: Data to score (n_samples, n_features)
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            errors = self.model.get_reconstruction_error(X_tensor).cpu().numpy()
            
        return errors
    
    def score_samples_normalized(self, X: np.ndarray) -> np.ndarray:
        """
        Get normalized anomaly scores in [0, 1].
        
        Uses training error distribution for normalization.
        """
        errors = self.score_samples(X)
        
        # Normalize using training error statistics
        min_err = self.training_errors.min()
        max_err = self.training_errors.max()
        
        if max_err - min_err > 0:
            normalized = (errors - min_err) / (max_err - min_err)
        else:
            normalized = np.zeros_like(errors)
            
        return np.clip(normalized, 0, 1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Data to predict (n_samples, n_features)
            
        Returns:
            Labels: -1 for anomaly, 1 for normal
        """
        errors = self.score_samples(X)
        predictions = np.where(errors > self.threshold, -1, 1)
        return predictions
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent space representation of data.
        
        Useful for visualization and understanding learned patterns.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            latent = self.model.encode(X_tensor).cpu().numpy()
            
        return latent
    
    def get_reconstruction(self, X: np.ndarray) -> np.ndarray:
        """Get reconstructed data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstruction = self.model(X_tensor).cpu().numpy()
            
        # Inverse transform
        return self.scaler.inverse_transform(reconstruction)
    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def plot_training_history(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training and validation loss curves.
        
        Shows model convergence and potential overfitting.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Autoencoder Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")
            
        return fig
    
    def plot_reconstruction_error_distribution(
        self,
        X_test: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot reconstruction error distribution with threshold.
        
        Args:
            X_test: Optional test data to include
            y_true: Optional true labels (0=normal, 1=anomaly) for coloring
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Training error distribution
        ax1 = axes[0]
        ax1.hist(
            self.training_errors, 
            bins=50, 
            density=True, 
            alpha=0.7,
            color='steelblue',
            edgecolor='black',
            linewidth=0.5,
            label='Training Data'
        )
        ax1.axvline(
            self.threshold, 
            color='red', 
            linestyle='--', 
            linewidth=2,
            label=f'Threshold ({self.config.threshold_percentile}th %ile)'
        )
        ax1.set_xlabel('Reconstruction Error', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Training Data Reconstruction Errors', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Shade anomaly region
        xlim = ax1.get_xlim()
        ax1.fill_betweenx(
            [0, ax1.get_ylim()[1]], 
            self.threshold, 
            xlim[1],
            alpha=0.2, 
            color='red',
            label='Anomaly Region'
        )
        
        # Right plot: Test data (if provided)
        ax2 = axes[1]
        if X_test is not None:
            test_errors = self.score_samples(X_test)
            
            if y_true is not None:
                # Color by true label
                normal_mask = y_true == 0
                anomaly_mask = y_true == 1
                
                ax2.hist(
                    test_errors[normal_mask], 
                    bins=30, 
                    density=True,
                    alpha=0.6,
                    color='green',
                    edgecolor='black',
                    linewidth=0.5,
                    label='True Normal'
                )
                ax2.hist(
                    test_errors[anomaly_mask], 
                    bins=30, 
                    density=True,
                    alpha=0.6,
                    color='red',
                    edgecolor='black',
                    linewidth=0.5,
                    label='True Anomaly'
                )
            else:
                ax2.hist(
                    test_errors, 
                    bins=50, 
                    density=True,
                    alpha=0.7,
                    color='orange',
                    edgecolor='black',
                    linewidth=0.5,
                    label='Test Data'
                )
                
            ax2.axvline(
                self.threshold, 
                color='red', 
                linestyle='--', 
                linewidth=2,
                label='Threshold'
            )
            ax2.set_xlabel('Reconstruction Error', fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)
            ax2.set_title('Test Data Reconstruction Errors', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5, 0.5, 
                'No test data provided',
                ha='center', va='center',
                fontsize=14,
                transform=ax2.transAxes
            )
            ax2.set_title('Test Data', fontsize=14, fontweight='bold')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved reconstruction error plot to {save_path}")
            
        return fig
    
    def plot_latent_space(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the 2D projection of latent space.
        
        Uses first 2 dimensions of latent space or PCA if latent_dim > 2.
        """
        latent = self.get_latent_representation(X)
        
        if latent.shape[1] > 2:
            # Use PCA for visualization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent)
            title_suffix = " (PCA projection)"
        else:
            latent_2d = latent[:, :2]
            title_suffix = ""
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            scatter = ax.scatter(
                latent_2d[:, 0], 
                latent_2d[:, 1],
                c=labels,
                cmap='RdYlGn_r',
                alpha=0.6,
                s=20
            )
            plt.colorbar(scatter, ax=ax, label='Anomaly Score')
        else:
            ax.scatter(
                latent_2d[:, 0], 
                latent_2d[:, 1],
                alpha=0.6,
                s=20,
                color='steelblue'
            )
            
        ax.set_xlabel('Latent Dimension 1', fontsize=12)
        ax.set_ylabel('Latent Dimension 2', fontsize=12)
        ax.set_title(f'Latent Space Visualization{title_suffix}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved latent space plot to {save_path}")
            
        return fig
    
    def plot_feature_reconstruction(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 5,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare original vs reconstructed features for sample records.
        
        Useful for understanding what the model learns.
        """
        reconstruction = self.get_reconstruction(X[:n_samples])
        
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
            
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
        if n_samples == 1:
            axes = [axes]
            
        for i, ax in enumerate(axes):
            x = np.arange(n_features)
            width = 0.35
            
            ax.bar(x - width/2, X[i], width, label='Original', color='steelblue', alpha=0.8)
            ax.bar(x + width/2, reconstruction[i], width, label='Reconstructed', color='orange', alpha=0.8)
            
            ax.set_xlabel('Features', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(f'Sample {i+1}', fontsize=12)
            ax.legend(fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names[:n_features], rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
        plt.suptitle('Original vs Reconstructed Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved reconstruction comparison to {save_path}")
            
        return fig
    
    def plot_anomaly_scores(
        self,
        X: np.ndarray,
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot anomaly scores with threshold visualization.
        """
        scores = self.score_samples(X)
        sorted_indices = np.argsort(scores)[::-1]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['red' if s > self.threshold else 'green' for s in scores[sorted_indices[:top_n]]]
        
        ax.barh(
            range(top_n), 
            scores[sorted_indices[:top_n]],
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )
        ax.axvline(self.threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
        
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f'Record {i}' for i in sorted_indices[:top_n]])
        ax.set_xlabel('Reconstruction Error (Anomaly Score)', fontsize=12)
        ax.set_title(f'Top {top_n} Anomaly Scores', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved anomaly scores plot to {save_path}")
            
        return fig
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self, path: str):
        """Save the complete detector to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), path / 'model.pth')
        
        # Save config and metadata
        metadata = {
            'config': {
                'latent_dim': self.config.latent_dim,
                'hidden_dims': self.config.hidden_dims,
                'dropout': self.config.dropout,
                'threshold_percentile': self.config.threshold_percentile
            },
            'input_dim': self.model.input_dim,
            'threshold': float(self.threshold),
            'history': self.history
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save scaler
        import joblib
        joblib.dump(self.scaler, path / 'scaler.pkl')
        
        # Save training errors
        np.save(path / 'training_errors.npy', self.training_errors)
        
        logger.info(f"Saved detector to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'AutoencoderAnomalyDetector':
        """Load a saved detector."""
        path = Path(path)
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            
        # Create config
        config = AutoencoderConfig(**metadata['config'])
        
        # Create detector
        detector = cls(config)
        
        # Create and load model
        detector.model = AadhaarAutoencoder(metadata['input_dim'], config)
        detector.model.load_state_dict(torch.load(path / 'model.pth'))
        detector.model.to(detector.device)
        
        # Load scaler
        import joblib
        detector.scaler = joblib.load(path / 'scaler.pkl')
        
        # Load other attributes
        detector.threshold = metadata['threshold']
        detector.history = metadata['history']
        detector.training_errors = np.load(path / 'training_errors.npy')
        detector.is_fitted = True
        
        logger.info(f"Loaded detector from {path}")
        return detector
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = f"""
================================================================================
AUTOENCODER MODEL SUMMARY
================================================================================
Architecture:
  Input Dimension: {self.model.input_dim}
  Hidden Layers: {self.config.hidden_dims}
  Latent Dimension: {self.config.latent_dim}
  Dropout: {self.config.dropout}
  Batch Normalization: {self.config.use_batch_norm}

Parameters:
  Total: {total_params:,}
  Trainable: {trainable_params:,}

Training:
  Epochs: {len(self.history['train_loss'])}
  Final Train Loss: {self.history['train_loss'][-1]:.6f}
  Final Val Loss: {self.history['val_loss'][-1]:.6f}

Anomaly Detection:
  Threshold Percentile: {self.config.threshold_percentile}
  Threshold Value: {self.threshold:.6f}

Device: {self.device}
================================================================================
"""
        return summary


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def train_autoencoder(
    X: np.ndarray,
    latent_dim: int = 8,
    epochs: int = 100,
    threshold_percentile: float = 95.0
) -> AutoencoderAnomalyDetector:
    """
    Convenience function to train an autoencoder.
    
    Args:
        X: Training data
        latent_dim: Dimension of latent space
        epochs: Number of training epochs
        threshold_percentile: Percentile for anomaly threshold
        
    Returns:
        Trained AutoencoderAnomalyDetector
    """
    config = AutoencoderConfig(
        latent_dim=latent_dim,
        epochs=epochs,
        threshold_percentile=threshold_percentile
    )
    
    detector = AutoencoderAnomalyDetector(config)
    detector.fit(X)
    
    return detector


# =============================================================================
# MAIN - DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
    
    # Generate synthetic Aadhaar-like data
    print("\n" + "="*80)
    print("DEMONSTRATION: Autoencoder for Aadhaar Fraud Detection")
    print("="*80)
    
    np.random.seed(42)
    
    # Simulate features
    n_normal = 2000
    n_anomaly = 100
    n_features = 20
    
    # Normal patterns (realistic Aadhaar data)
    X_normal = np.random.randn(n_normal, n_features)
    
    # Anomalous patterns (shifted distributions)
    X_anomaly = np.random.randn(n_anomaly, n_features) * 2 + 3
    
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normal + [1] * n_anomaly)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Normal: {n_normal}, Anomalies: {n_anomaly}")
    
    # Train autoencoder
    print("\n" + "-"*40)
    print("Training Autoencoder...")
    print("-"*40)
    
    config = AutoencoderConfig(
        latent_dim=8,
        hidden_dims=[64, 32],
        epochs=50,
        threshold_percentile=95.0
    )
    
    detector = AutoencoderAnomalyDetector(config)
    detector.fit(X)
    
    print(detector.get_model_summary())
    
    # Evaluate
    predictions = detector.predict(X)
    scores = detector.score_samples(X)
    
    detected_anomalies = (predictions == -1).sum()
    true_positives = ((predictions == -1) & (y == 1)).sum()
    false_positives = ((predictions == -1) & (y == 0)).sum()
    
    print("\nResults:")
    print(f"  Detected Anomalies: {detected_anomalies}")
    print(f"  True Positives: {true_positives}/{n_anomaly}")
    print(f"  False Positives: {false_positives}")
    print(f"  Precision: {true_positives / max(1, detected_anomalies):.2%}")
    print(f"  Recall: {true_positives / n_anomaly:.2%}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    detector.plot_training_history('training_history.png')
    detector.plot_reconstruction_error_distribution(X, y, 'reconstruction_error.png')
    detector.plot_anomaly_scores(X, top_n=20, save_path='anomaly_scores.png')
    
    print("\nVisualization files saved!")
