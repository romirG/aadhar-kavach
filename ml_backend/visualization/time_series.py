"""
Time series visualization module.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64

logger = logging.getLogger(__name__)


def create_time_series_plot(
    df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    date_column: str = 'date',
    value_columns: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Create time series plot with anomaly overlay.
    
    Args:
        df: DataFrame with time series data
        anomaly_scores: Array of anomaly scores
        date_column: Name of date column
        value_columns: Columns to plot (auto-detect if None)
        threshold: Threshold for highlighting anomalies
        
    Returns:
        Dict with chart data and base64 encoded image
    """
    try:
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Parse dates
        if date_column in df.columns:
            df = df.copy()
            df['date_parsed'] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.dropna(subset=['date_parsed'])
            df = df.sort_values('date_parsed')
        else:
            df['date_parsed'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # Auto-detect value columns if not specified
        if value_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_columns = [c for c in numeric_cols if 'encoded' not in c and 'parsed' not in str(c)][:3]
        
        # Plot 1: Time series with anomaly highlights
        ax1 = axes[0]
        
        for col in value_columns[:3]:
            if col in df.columns:
                ax1.plot(df['date_parsed'], df[col], label=col.replace('_', ' ').title(), alpha=0.7)
        
        # Highlight anomaly regions
        anomaly_mask = anomaly_scores > threshold
        if len(anomaly_mask) == len(df):
            anomaly_dates = df['date_parsed'].values[anomaly_mask]
            for date in anomaly_dates:
                ax1.axvline(x=date, color='red', alpha=0.3, linewidth=0.5)
        
        ax1.set_ylabel('Value')
        ax1.set_title('📈 Event Volume Over Time (with Anomaly Overlay)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores over time
        ax2 = axes[1]
        
        if len(anomaly_scores) == len(df):
            ax2.fill_between(df['date_parsed'], anomaly_scores, alpha=0.5, color='orange', label='Anomaly Score')
            ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
            ax2.axhline(y=0.8, color='darkred', linestyle=':', alpha=0.5, label='High Risk (0.8)')
        
        ax2.set_ylabel('Anomaly Score')
        ax2.set_xlabel('Date')
        ax2.set_title('🚨 Anomaly Score Timeline')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        # Prepare plotly-compatible data
        chart_data = {
            "chart_type": "time_series",
            "title": "Event Volume and Anomaly Scores Over Time",
            "x_label": "Date",
            "y_label": "Value / Anomaly Score",
            "data": {
                "dates": df['date_parsed'].dt.strftime('%Y-%m-%d').tolist(),
                "series": {},
                "anomaly_scores": anomaly_scores.tolist() if len(anomaly_scores) == len(df) else [],
                "threshold": threshold
            },
            "image_base64": image_base64
        }
        
        for col in value_columns[:3]:
            if col in df.columns:
                chart_data["data"]["series"][col] = df[col].tolist()
        
        logger.info("Created time series plot")
        return chart_data
        
    except Exception as e:
        logger.error(f"Error creating time series plot: {e}")
        return {"error": str(e), "chart_type": "time_series"}


def create_score_distribution_plot(
    anomaly_scores: np.ndarray,
    predictions: np.ndarray,
    thresholds: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create anomaly score distribution histogram.
    
    Args:
        anomaly_scores: Array of anomaly scores
        predictions: Array of predictions (-1 or 1)
        thresholds: Dict with risk thresholds
        
    Returns:
        Dict with chart data and base64 encoded image
    """
    try:
        thresholds = thresholds or {"medium": 0.5, "high": 0.8}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        bins = np.linspace(0, 1, 50)
        
        normal_scores = anomaly_scores[predictions == 1]
        anomaly_indices = predictions == -1
        high_risk_scores = anomaly_scores[(anomaly_scores >= thresholds["high"]) & anomaly_indices]
        medium_risk_scores = anomaly_scores[(anomaly_scores >= thresholds["medium"]) & (anomaly_scores < thresholds["high"]) & anomaly_indices]
        low_risk_scores = anomaly_scores[(anomaly_scores < thresholds["medium"]) | ~anomaly_indices]
        
        ax.hist(low_risk_scores, bins=bins, alpha=0.7, label='Normal / Low Risk', color='green')
        ax.hist(medium_risk_scores, bins=bins, alpha=0.7, label='Medium Risk', color='orange')
        ax.hist(high_risk_scores, bins=bins, alpha=0.7, label='High Risk', color='red')
        
        # Add threshold lines
        ax.axvline(x=thresholds["medium"], color='orange', linestyle='--', linewidth=2, label=f'Medium Threshold ({thresholds["medium"]})')
        ax.axvline(x=thresholds["high"], color='red', linestyle='--', linewidth=2, label=f'High Threshold ({thresholds["high"]})')
        
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title('📊 Anomaly Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        # Statistics
        stats = {
            "mean": float(np.mean(anomaly_scores)),
            "std": float(np.std(anomaly_scores)),
            "min": float(np.min(anomaly_scores)),
            "max": float(np.max(anomaly_scores)),
            "median": float(np.median(anomaly_scores)),
            "high_risk_count": int(len(high_risk_scores)),
            "medium_risk_count": int(len(medium_risk_scores)),
            "normal_count": int(len(low_risk_scores))
        }
        
        chart_data = {
            "chart_type": "distribution",
            "title": "Anomaly Score Distribution",
            "x_label": "Anomaly Score",
            "y_label": "Frequency",
            "data": {
                "scores": anomaly_scores.tolist(),
                "histogram": np.histogram(anomaly_scores, bins=bins)[0].tolist(),
                "bin_edges": bins.tolist(),
                "thresholds": thresholds,
                "statistics": stats
            },
            "image_base64": image_base64
        }
        
        logger.info("Created score distribution plot")
        return chart_data
        
    except Exception as e:
        logger.error(f"Error creating distribution plot: {e}")
        return {"error": str(e), "chart_type": "distribution"}


def create_feature_importance_plot(
    feature_importance: Dict[str, float],
    top_k: int = 15
) -> Dict[str, Any]:
    """
    Create horizontal bar chart of feature importance.
    
    Args:
        feature_importance: Dict mapping feature names to importance
        top_k: Number of top features to show
        
    Returns:
        Dict with chart data and base64 encoded image
    """
    try:
        # Sort and get top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        colors = plt.cm.RdYlGn_r(np.array(importances))
        
        bars = ax.barh(y_pos, importances, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title('🔍 Feature Importance for Anomaly Detection')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, importance in zip(bars, importances):
            ax.text(importance + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        chart_data = {
            "chart_type": "feature_importance",
            "title": "Feature Importance for Anomaly Detection",
            "x_label": "Importance Score",
            "y_label": "Feature",
            "data": {
                "features": features,
                "importances": importances
            },
            "image_base64": image_base64
        }
        
        logger.info("Created feature importance plot")
        return chart_data
        
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {e}")
        return {"error": str(e), "chart_type": "feature_importance"}
