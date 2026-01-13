"""
Geographic heatmap visualization module.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)

# Indian state coordinates (approximate centers for visualization)
STATE_COORDINATES = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
    "Delhi": (28.7041, 77.1025),
    "Jammu & Kashmir": (33.7782, 76.5762),
    "Ladakh": (34.1526, 77.5771),
    "Puducherry": (11.9416, 79.8083),
    "Chandigarh": (30.7333, 76.7794),
    "Andaman & Nicobar Islands": (11.7401, 92.6586),
    "Dadra & Nagar Haveli": (20.1809, 73.0169),
    "Daman & Diu": (20.4283, 72.8397),
    "Lakshadweep": (10.5667, 72.6417)
}


def create_geo_heatmap(
    df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    state_column: str = 'state',
    district_column: str = 'district'
) -> Dict[str, Any]:
    """
    Create geographic heatmap of anomalies.
    
    Args:
        df: DataFrame with geographic data
        anomaly_scores: Array of anomaly scores
        state_column: Name of state column
        district_column: Name of district column
        
    Returns:
        Dict with chart data and base64 encoded image
    """
    try:
        df = df.copy()
        df['anomaly_score'] = anomaly_scores
        
        # Aggregate by state
        if state_column in df.columns:
            state_stats = df.groupby(state_column).agg({
                'anomaly_score': ['mean', 'max', 'count']
            }).reset_index()
            state_stats.columns = ['state', 'mean_score', 'max_score', 'count']
            
            # Sort by mean anomaly score
            state_stats = state_stats.sort_values('mean_score', ascending=False)
        else:
            state_stats = pd.DataFrame()
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: State-wise anomaly heatmap (bar chart approximation)
        ax1 = axes[0]
        
        if not state_stats.empty:
            top_states = state_stats.head(15)
            y_pos = np.arange(len(top_states))
            colors = plt.cm.RdYlGn_r(top_states['mean_score'].values)
            
            bars = ax1.barh(y_pos, top_states['mean_score'], color=colors)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_states['state'].values)
            ax1.invert_yaxis()
            ax1.set_xlabel('Mean Anomaly Score')
            ax1.set_title('🗺️ State-wise Anomaly Heat (Top 15)')
            ax1.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Risk')
            ax1.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='High Risk')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add count annotations
            for bar, count in zip(bars, top_states['count'].values):
                ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'n={count}', va='center', fontsize=8)
        
        # Plot 2: Scatter map visualization
        ax2 = axes[1]
        
        if not state_stats.empty:
            # Plot states on approximate map
            for _, row in state_stats.iterrows():
                state = row['state']
                if state in STATE_COORDINATES:
                    lat, lon = STATE_COORDINATES[state]
                    score = row['mean_score']
                    count = row['count']
                    
                    # Color based on anomaly score
                    color = plt.cm.RdYlGn_r(score)
                    size = min(300, 50 + count * 2)  # Size based on count
                    
                    ax2.scatter(lon, lat, c=[color], s=size, alpha=0.7, edgecolors='black', linewidths=0.5)
                    
                    # Add label for high-risk states
                    if score > 0.5:
                        ax2.annotate(state[:3].upper(), (lon, lat), fontsize=7, ha='center')
            
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            ax2.set_title('🌍 Geographic Distribution of Anomalies')
            ax2.grid(True, alpha=0.3)
            
            # Add India approximate boundaries
            ax2.set_xlim(68, 98)
            ax2.set_ylim(8, 37)
        
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        # Prepare data for interactive visualization
        chart_data = {
            "chart_type": "geo_heatmap",
            "title": "Geographic Anomaly Distribution",
            "data": {
                "state_stats": state_stats.to_dict('records') if not state_stats.empty else [],
                "high_risk_states": state_stats[state_stats['mean_score'] > 0.5]['state'].tolist() if not state_stats.empty else [],
                "state_coordinates": {
                    state: {"lat": coords[0], "lon": coords[1]}
                    for state, coords in STATE_COORDINATES.items()
                }
            },
            "image_base64": image_base64
        }
        
        # District-level stats if available
        if district_column in df.columns:
            district_stats = df.groupby([state_column, district_column]).agg({
                'anomaly_score': ['mean', 'count']
            }).reset_index()
            district_stats.columns = ['state', 'district', 'mean_score', 'count']
            
            # Top 10 high-risk districts
            top_districts = district_stats.nlargest(10, 'mean_score')
            chart_data["data"]["top_risk_districts"] = top_districts.to_dict('records')
        
        logger.info("Created geographic heatmap")
        return chart_data
        
    except Exception as e:
        logger.error(f"Error creating geo heatmap: {e}")
        return {"error": str(e), "chart_type": "geo_heatmap"}


def create_cluster_visualization(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    feature_x: Optional[str] = None,
    feature_y: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create cluster visualization with anomaly highlighting.
    
    Args:
        df: DataFrame with features
        cluster_labels: Cluster labels from HDBSCAN
        anomaly_scores: Array of anomaly scores
        feature_x: Feature for x-axis (auto-select if None)
        feature_y: Feature for y-axis (auto-select if None)
        
    Returns:
        Dict with chart data and base64 encoded image
    """
    try:
        df = df.copy()
        
        # Auto-select features if not specified
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if 'encoded' not in c]
        
        if len(numeric_cols) < 2:
            return {"error": "Not enough numeric features for visualization", "chart_type": "cluster"}
        
        feature_x = feature_x or numeric_cols[0]
        feature_y = feature_y or numeric_cols[1]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Cluster visualization
        ax1 = axes[0]
        
        # Separate noise points and clustered points
        noise_mask = cluster_labels == -1
        
        # Plot clustered points
        unique_labels = set(cluster_labels) - {-1}
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = cluster_labels == label
            ax1.scatter(
                df.loc[mask, feature_x],
                df.loc[mask, feature_y],
                c=[color],
                label=f'Cluster {label}',
                alpha=0.6,
                s=50
            )
        
        # Plot noise points (anomalies)
        ax1.scatter(
            df.loc[noise_mask, feature_x],
            df.loc[noise_mask, feature_y],
            c='red',
            marker='x',
            label='Anomaly (Noise)',
            alpha=0.8,
            s=100
        )
        
        ax1.set_xlabel(feature_x.replace('_', ' ').title())
        ax1.set_ylabel(feature_y.replace('_', ' ').title())
        ax1.set_title('🔬 HDBSCAN Cluster Analysis')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly score heatmap
        ax2 = axes[1]
        
        scatter = ax2.scatter(
            df[feature_x],
            df[feature_y],
            c=anomaly_scores,
            cmap='RdYlGn_r',
            alpha=0.6,
            s=50
        )
        
        plt.colorbar(scatter, ax=ax2, label='Anomaly Score')
        ax2.set_xlabel(feature_x.replace('_', ' ').title())
        ax2.set_ylabel(feature_y.replace('_', ' ').title())
        ax2.set_title('🎯 Anomaly Score Overlay')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        # Cluster statistics
        cluster_stats = {}
        for label in set(cluster_labels):
            mask = cluster_labels == label
            cluster_stats[f"cluster_{label}"] = {
                "count": int(mask.sum()),
                "mean_anomaly_score": float(anomaly_scores[mask].mean()),
                "is_noise": label == -1
            }
        
        chart_data = {
            "chart_type": "cluster",
            "title": "Cluster Analysis and Anomaly Detection",
            "data": {
                "feature_x": feature_x,
                "feature_y": feature_y,
                "n_clusters": len(unique_labels),
                "n_noise": int(noise_mask.sum()),
                "cluster_stats": cluster_stats
            },
            "image_base64": image_base64
        }
        
        logger.info("Created cluster visualization")
        return chart_data
        
    except Exception as e:
        logger.error(f"Error creating cluster visualization: {e}")
        return {"error": str(e), "chart_type": "cluster"}
