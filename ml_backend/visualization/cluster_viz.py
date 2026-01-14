"""
Enhanced cluster visualization with dimensionality reduction.

Provides advanced cluster visualizations using t-SNE, UMAP, and PCA
for better understanding of high-dimensional anomaly patterns.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import io
import base64

logger = logging.getLogger(__name__)

# Optional imports for dimensionality reduction
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available for dimensionality reduction")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.info("UMAP not available (optional)")


def reduce_dimensions(
    X: np.ndarray,
    method: str = 'tsne',
    n_components: int = 2,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce dimensions of data for visualization.
    
    Args:
        X: High-dimensional data
        method: 'tsne', 'umap', or 'pca'
        n_components: Number of dimensions (2 or 3)
        random_state: Random seed
        
    Returns:
        Reduced dimensional data
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("Dimensionality reduction not available, returning first 2 columns")
        return X[:, :n_components] if X.shape[1] >= n_components else X
    
    try:
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=random_state, 
                          perplexity=min(30, len(X) - 1))
            return reducer.fit_transform(X)
        
        elif method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
            return reducer.fit_transform(X)
        
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=random_state)
            return reducer.fit_transform(X)
        
        else:
            logger.warning(f"Method {method} not available, using PCA")
            reducer = PCA(n_components=n_components, random_state=random_state)
            return reducer.fit_transform(X)
            
    except Exception as e:
        logger.error(f"Error in dimensionality reduction: {e}")
        return X[:, :n_components] if X.shape[1] >= n_components else X


def create_advanced_cluster_plot(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = 'tsne'
) -> Dict[str, Any]:
    """
    Create advanced cluster visualization with dimensionality reduction.
    
    Args:
        X: Feature matrix
        cluster_labels: Cluster labels from HDBSCAN/DBSCAN
        anomaly_scores: Anomaly scores
        feature_names: Optional feature names
        method: Dimensionality reduction method ('tsne', 'umap', 'pca')
        
    Returns:
        Dict with chart data and base64 encoded image
    """
    try:
        # Reduce to 2D
        X_reduced = reduce_dimensions(X, method=method, n_components=2)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot 1: Cluster membership
        ax1 = axes[0, 0]
        
        unique_labels = set(cluster_labels)
        noise_mask = cluster_labels == -1
        
        # Plot clusters
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                continue
            mask = cluster_labels == label
            ax1.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                       c=[color], label=f'Cluster {label}',
                       alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        
        # Plot noise points
        if noise_mask.any():
            ax1.scatter(X_reduced[noise_mask, 0], X_reduced[noise_mask, 1],
                       c='red', marker='x', label='Outliers',
                       alpha=0.8, s=100, linewidths=2)
        
        ax1.set_xlabel(f'{method.upper()} Component 1')
        ax1.set_ylabel(f'{method.upper()} Component 2')
        ax1.set_title(f'🔬 Cluster Visualization ({method.upper()})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly score heatmap
        ax2 = axes[0, 1]
        
        scatter = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1],
                             c=anomaly_scores, cmap='RdYlGn_r',
                             alpha=0.7, s=50, edgecolors='black', linewidths=0.5)
        
        plt.colorbar(scatter, ax=ax2, label='Anomaly Score')
        ax2.set_xlabel(f'{method.upper()} Component 1')
        ax2.set_ylabel(f'{method.upper()} Component 2')
        ax2.set_title('🎯 Anomaly Score Heatmap')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cluster density
        ax3 = axes[1, 0]
        
        # Create 2D histogram
        h = ax3.hist2d(X_reduced[:, 0], X_reduced[:, 1], bins=30, cmap='YlOrRd')
        plt.colorbar(h[3], ax=ax3, label='Density')
        
        # Overlay outliers
        if noise_mask.any():
            ax3.scatter(X_reduced[noise_mask, 0], X_reduced[noise_mask, 1],
                       c='blue', marker='x', s=100, linewidths=2, alpha=0.8)
        
        ax3.set_xlabel(f'{method.upper()} Component 1')
        ax3.set_ylabel(f'{method.upper()} Component 2')
        ax3.set_title('📊 Cluster Density Map')
        
        # Plot 4: Cluster statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate cluster statistics
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_outliers = noise_mask.sum()
        
        stats_text = f"""
        📈 CLUSTER ANALYSIS SUMMARY
        {'─' * 40}
        
        Dimensionality Reduction: {method.upper()}
        Total Samples: {len(X):,}
        Number of Clusters: {n_clusters}
        Outliers Detected: {n_outliers:,} ({100*n_outliers/len(X):.1f}%)
        
        🔍 CLUSTER DETAILS
        {'─' * 40}
        """
        
        for label in sorted(unique_labels):
            if label == -1:
                continue
            mask = cluster_labels == label
            count = mask.sum()
            mean_score = anomaly_scores[mask].mean()
            
            stats_text += f"\nCluster {label}:"
            stats_text += f"\n  • Size: {count:,} ({100*count/len(X):.1f}%)"
            stats_text += f"\n  • Mean Anomaly Score: {mean_score:.3f}"
        
        if n_outliers > 0:
            outlier_mean_score = anomaly_scores[noise_mask].mean()
            stats_text += f"\n\nOutliers:"
            stats_text += f"\n  • Count: {n_outliers:,}"
            stats_text += f"\n  • Mean Anomaly Score: {outlier_mean_score:.3f}"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle(f'Advanced Cluster Analysis - {method.upper()} Visualization',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        # Prepare data
        cluster_stats = {}
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_stats[f"cluster_{label}"] = {
                "count": int(mask.sum()),
                "mean_anomaly_score": float(anomaly_scores[mask].mean()),
                "is_noise": label == -1
            }
        
        chart_data = {
            "chart_type": "advanced_cluster",
            "title": f"Advanced Cluster Analysis ({method.upper()})",
            "method": method,
            "data": {
                "n_clusters": n_clusters,
                "n_outliers": int(n_outliers),
                "cluster_stats": cluster_stats,
                "reduced_coordinates": {
                    "x": X_reduced[:, 0].tolist(),
                    "y": X_reduced[:, 1].tolist()
                },
                "cluster_labels": cluster_labels.tolist(),
                "anomaly_scores": anomaly_scores.tolist()
            },
            "image_base64": image_base64
        }
        
        logger.info(f"Created advanced cluster plot with {method}")
        return chart_data
        
    except Exception as e:
        logger.error(f"Error creating advanced cluster plot: {e}")
        return {"error": str(e), "chart_type": "advanced_cluster"}


def create_3d_cluster_plot(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    method: str = 'pca'
) -> Dict[str, Any]:
    """
    Create 3D cluster visualization.
    
    Args:
        X: Feature matrix
        cluster_labels: Cluster labels
        anomaly_scores: Anomaly scores
        method: Dimensionality reduction method
        
    Returns:
        Dict with chart data and base64 encoded image
    """
    try:
        # Reduce to 3D
        X_reduced = reduce_dimensions(X, method=method, n_components=3)
        
        fig = plt.figure(figsize=(16, 6))
        
        # Plot 1: 3D Clusters
        ax1 = fig.add_subplot(131, projection='3d')
        
        unique_labels = set(cluster_labels)
        noise_mask = cluster_labels == -1
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                continue
            mask = cluster_labels == label
            ax1.scatter(X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                       c=[color], label=f'C{label}', alpha=0.6, s=30)
        
        if noise_mask.any():
            ax1.scatter(X_reduced[noise_mask, 0], X_reduced[noise_mask, 1], 
                       X_reduced[noise_mask, 2], c='red', marker='x',
                       label='Outliers', alpha=0.8, s=50)
        
        ax1.set_xlabel(f'{method.upper()} 1')
        ax1.set_ylabel(f'{method.upper()} 2')
        ax1.set_zlabel(f'{method.upper()} 3')
        ax1.set_title('3D Cluster View')
        ax1.legend(fontsize=8)
        
        # Plot 2: 3D Anomaly Scores
        ax2 = fig.add_subplot(132, projection='3d')
        
        scatter = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                             c=anomaly_scores, cmap='RdYlGn_r', alpha=0.6, s=30)
        
        fig.colorbar(scatter, ax=ax2, label='Anomaly Score', shrink=0.5)
        ax2.set_xlabel(f'{method.upper()} 1')
        ax2.set_ylabel(f'{method.upper()} 2')
        ax2.set_zlabel(f'{method.upper()} 3')
        ax2.set_title('3D Anomaly Heatmap')
        
        # Plot 3: Different angle
        ax3 = fig.add_subplot(133, projection='3d')
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                continue
            mask = cluster_labels == label
            ax3.scatter(X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                       c=[color], alpha=0.6, s=30)
        
        if noise_mask.any():
            ax3.scatter(X_reduced[noise_mask, 0], X_reduced[noise_mask, 1],
                       X_reduced[noise_mask, 2], c='red', marker='x',
                       alpha=0.8, s=50)
        
        ax3.set_xlabel(f'{method.upper()} 1')
        ax3.set_ylabel(f'{method.upper()} 2')
        ax3.set_zlabel(f'{method.upper()} 3')
        ax3.set_title('3D View (Rotated)')
        ax3.view_init(elev=20, azim=45)
        
        plt.suptitle(f'3D Cluster Visualization ({method.upper()})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        chart_data = {
            "chart_type": "3d_cluster",
            "title": f"3D Cluster Visualization ({method.upper()})",
            "method": method,
            "data": {
                "reduced_coordinates": {
                    "x": X_reduced[:, 0].tolist(),
                    "y": X_reduced[:, 1].tolist(),
                    "z": X_reduced[:, 2].tolist()
                },
                "cluster_labels": cluster_labels.tolist(),
                "anomaly_scores": anomaly_scores.tolist()
            },
            "image_base64": image_base64
        }
        
        logger.info(f"Created 3D cluster plot with {method}")
        return chart_data
        
    except Exception as e:
        logger.error(f"Error creating 3D cluster plot: {e}")
        return {"error": str(e), "chart_type": "3d_cluster"}


def create_cluster_comparison(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    anomaly_scores: np.ndarray
) -> Dict[str, Any]:
    """
    Create comparison of different dimensionality reduction methods.
    
    Args:
        X: Feature matrix
        cluster_labels: Cluster labels
        anomaly_scores: Anomaly scores
        
    Returns:
        Dict with chart data and base64 encoded image
    """
    try:
        methods = ['pca', 'tsne']
        if UMAP_AVAILABLE:
            methods.append('umap')
        
        fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
        if len(methods) == 1:
            axes = [axes]
        
        for ax, method in zip(axes, methods):
            X_reduced = reduce_dimensions(X, method=method, n_components=2)
            
            unique_labels = set(cluster_labels)
            noise_mask = cluster_labels == -1
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    continue
                mask = cluster_labels == label
                ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                          c=[color], alpha=0.6, s=30, edgecolors='black', linewidths=0.3)
            
            if noise_mask.any():
                ax.scatter(X_reduced[noise_mask, 0], X_reduced[noise_mask, 1],
                          c='red', marker='x', s=80, linewidths=2, alpha=0.8)
            
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel(f'{method.upper()} 2')
            ax.set_title(f'{method.upper()} Projection')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Dimensionality Reduction Method Comparison',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        chart_data = {
            "chart_type": "cluster_comparison",
            "title": "Dimensionality Reduction Comparison",
            "methods": methods,
            "image_base64": image_base64
        }
        
        logger.info("Created cluster comparison plot")
        return chart_data
        
    except Exception as e:
        logger.error(f"Error creating cluster comparison: {e}")
        return {"error": str(e), "chart_type": "cluster_comparison"}
