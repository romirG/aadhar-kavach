"""
Comprehensive Visualization Demonstration Script

Generates all visualization types for UIDAI anomaly detection results.
Demonstrates:
- Time-series plots with anomaly overlays
- Geo-heatmaps of suspicious activity
- Anomaly score distributions
- Cluster visualizations (DBSCAN/HDBSCAN)
- Risk indicator gauges
- Interactive plotly versions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import visualization modules
from visualization.time_series import (
    create_time_series_plot,
    create_score_distribution_plot,
    create_feature_importance_plot
)
from visualization.geo_heatmap import (
    create_geo_heatmap,
    create_cluster_visualization
)
from visualization.distribution import (
    create_risk_dashboard,
    create_comparison_plot
)
from visualization.cluster_viz import (
    create_advanced_cluster_plot,
    create_3d_cluster_plot,
    create_cluster_comparison
)
from visualization.interactive_plots import (
    create_interactive_timeseries,
    create_interactive_scatter,
    create_interactive_heatmap,
    create_interactive_dashboard,
    save_interactive_html
)


def create_sample_data():
    """Create comprehensive sample UIDAI data for visualization."""
    np.random.seed(42)
    n_samples = 500
    
    # Create dates
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Create normal data
    data = {
        'date': dates,
        'state': np.random.choice(['Maharashtra', 'Karnataka', 'Delhi', 'Gujarat', 
                                  'Tamil Nadu', 'Uttar Pradesh', 'West Bengal'], n_samples),
        'district': np.random.choice(['District_A', 'District_B', 'District_C', 
                                     'District_D', 'District_E'], n_samples),
        'month': dates.month,
        'year': dates.year,
        'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
        'total_enrolments': np.random.poisson(50, n_samples),
        'total_demo_updates': np.random.poisson(20, n_samples),
        'total_bio_updates': np.random.poisson(15, n_samples),
        'state_event_count': np.random.randint(100, 500, n_samples),
        'district_event_count': np.random.randint(50, 200, n_samples),
        'operator_activity': np.random.poisson(30, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
    
    for idx in anomaly_indices[:15]:  # Geo-inconsistent
        df.loc[idx, 'state_event_count'] = np.random.randint(1500, 2500)
        df.loc[idx, 'district_event_count'] = np.random.randint(1000, 1800)
    
    for idx in anomaly_indices[15:30]:  # Volume spike
        df.loc[idx, 'total_enrolments'] = np.random.randint(300, 600)
        df.loc[idx, 'total_demo_updates'] = np.random.randint(150, 300)
    
    for idx in anomaly_indices[30:45]:  # Weekend fraud
        df.loc[idx, 'is_weekend'] = 1
        df.loc[idx, 'total_bio_updates'] = np.random.randint(100, 200)
    
    for idx in anomaly_indices[45:]:  # Operator fraud
        df.loc[idx, 'operator_activity'] = np.random.randint(200, 400)
    
    return df


def generate_anomaly_scores(df):
    """Generate synthetic anomaly scores and predictions."""
    n = len(df)
    
    # Base scores
    anomaly_scores = np.random.beta(2, 5, n)  # Most scores low
    
    # Boost scores for high values
    for col in ['total_enrolments', 'total_demo_updates', 'total_bio_updates', 
                'state_event_count', 'operator_activity']:
        threshold = df[col].quantile(0.9)
        high_mask = df[col] > threshold
        anomaly_scores[high_mask] = np.clip(anomaly_scores[high_mask] + 0.3, 0, 1)
    
    # Weekend boost
    weekend_mask = df['is_weekend'] == 1
    anomaly_scores[weekend_mask] = np.clip(anomaly_scores[weekend_mask] + 0.2, 0, 1)
    
    # Predictions
    predictions = np.where(anomaly_scores > 0.5, -1, 1)
    
    # Cluster labels (simulate HDBSCAN)
    from sklearn.cluster import KMeans
    feature_cols = ['total_enrolments', 'total_demo_updates', 'total_bio_updates',
                   'state_event_count', 'district_event_count']
    X = df[feature_cols].values
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Mark high anomaly scores as noise
    cluster_labels[anomaly_scores > 0.7] = -1
    
    return anomaly_scores, predictions, cluster_labels, X


def main():
    """Run all visualization demonstrations."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  UIDAI ANOMALY DETECTION - VISUALIZATION DEMONSTRATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'viz_outputs')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Generate sample data
    print("=" * 80)
    print("GENERATING SAMPLE DATA")
    print("=" * 80)
    df = create_sample_data()
    anomaly_scores, predictions, cluster_labels, X = generate_anomaly_scores(df)
    print(f"✓ Generated {len(df)} records")
    print(f"✓ Anomalies: {(predictions == -1).sum()} ({100*(predictions == -1).sum()/len(df):.1f}%)")
    print(f"✓ Clusters: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    print(f"✓ Outliers: {(cluster_labels == -1).sum()}\n")
    
    # 1. Time Series Plots
    print("=" * 80)
    print("1. TIME SERIES VISUALIZATIONS")
    print("=" * 80)
    
    ts_result = create_time_series_plot(
        df, anomaly_scores,
        value_columns=['total_enrolments', 'total_demo_updates', 'total_bio_updates']
    )
    if 'error' not in ts_result:
        # Save image
        import base64
        img_data = base64.b64decode(ts_result['image_base64'])
        with open(os.path.join(output_dir, 'timeseries.png'), 'wb') as f:
            f.write(img_data)
        print("✓ Created time series plot")
    
    # 2. Score Distribution
    print("\n" + "=" * 80)
    print("2. SCORE DISTRIBUTION")
    print("=" * 80)
    
    dist_result = create_score_distribution_plot(anomaly_scores, predictions)
    if 'error' not in dist_result:
        img_data = base64.b64decode(dist_result['image_base64'])
        with open(os.path.join(output_dir, 'score_distribution.png'), 'wb') as f:
            f.write(img_data)
        print("✓ Created score distribution plot")
        print(f"  - Mean score: {dist_result['data']['statistics']['mean']:.3f}")
        print(f"  - High risk: {dist_result['data']['statistics']['high_risk_count']}")
    
    # 3. Geographic Heatmap
    print("\n" + "=" * 80)
    print("3. GEOGRAPHIC HEATMAP")
    print("=" * 80)
    
    geo_result = create_geo_heatmap(df, anomaly_scores)
    if 'error' not in geo_result:
        img_data = base64.b64decode(geo_result['image_base64'])
        with open(os.path.join(output_dir, 'geo_heatmap.png'), 'wb') as f:
            f.write(img_data)
        print("✓ Created geographic heatmap")
        print(f"  - High risk states: {len(geo_result['data']['high_risk_states'])}")
    
    # 4. Cluster Visualization (Basic)
    print("\n" + "=" * 80)
    print("4. BASIC CLUSTER VISUALIZATION")
    print("=" * 80)
    
    cluster_result = create_cluster_visualization(df, cluster_labels, anomaly_scores)
    if 'error' not in cluster_result:
        img_data = base64.b64decode(cluster_result['image_base64'])
        with open(os.path.join(output_dir, 'clusters_basic.png'), 'wb') as f:
            f.write(img_data)
        print("✓ Created basic cluster visualization")
    
    # 5. Advanced Cluster Visualization
    print("\n" + "=" * 80)
    print("5. ADVANCED CLUSTER VISUALIZATIONS")
    print("=" * 80)
    
    for method in ['pca', 'tsne']:
        adv_cluster_result = create_advanced_cluster_plot(X, cluster_labels, anomaly_scores, method=method)
        if 'error' not in adv_cluster_result:
            img_data = base64.b64decode(adv_cluster_result['image_base64'])
            with open(os.path.join(output_dir, f'clusters_advanced_{method}.png'), 'wb') as f:
                f.write(img_data)
            print(f"✓ Created advanced cluster plot ({method.upper()})")
    
    # 6. 3D Cluster Visualization
    print("\n" + "=" * 80)
    print("6. 3D CLUSTER VISUALIZATION")
    print("=" * 80)
    
    cluster_3d_result = create_3d_cluster_plot(X, cluster_labels, anomaly_scores, method='pca')
    if 'error' not in cluster_3d_result:
        img_data = base64.b64decode(cluster_3d_result['image_base64'])
        with open(os.path.join(output_dir, 'clusters_3d.png'), 'wb') as f:
            f.write(img_data)
        print("✓ Created 3D cluster visualization")
    
    # 7. Cluster Method Comparison
    print("\n" + "=" * 80)
    print("7. CLUSTER METHOD COMPARISON")
    print("=" * 80)
    
    comparison_result = create_cluster_comparison(X, cluster_labels, anomaly_scores)
    if 'error' not in comparison_result:
        img_data = base64.b64decode(comparison_result['image_base64'])
        with open(os.path.join(output_dir, 'cluster_comparison.png'), 'wb') as f:
            f.write(img_data)
        print("✓ Created cluster method comparison")
    
    # 8. Risk Dashboard
    print("\n" + "=" * 80)
    print("8. COMPREHENSIVE RISK DASHBOARD")
    print("=" * 80)
    
    model_results = [
        {'model_name': 'Isolation Forest', 'anomaly_count': (predictions == -1).sum(), 'threshold': 0.5},
        {'model_name': 'HDBSCAN', 'anomaly_count': (cluster_labels == -1).sum(), 'threshold': 0.5},
        {'model_name': 'Autoencoder', 'anomaly_count': (anomaly_scores > 0.7).sum(), 'threshold': 0.7}
    ]
    
    dashboard_result = create_risk_dashboard(anomaly_scores, predictions, df, model_results)
    if 'error' not in dashboard_result:
        img_data = base64.b64decode(dashboard_result['image_base64'])
        with open(os.path.join(output_dir, 'risk_dashboard.png'), 'wb') as f:
            f.write(img_data)
        print("✓ Created comprehensive risk dashboard")
        print(f"  - Overall risk score: {dashboard_result['data']['summary']['overall_risk_score']:.3f}")
    
    # 9. Feature Importance
    print("\n" + "=" * 80)
    print("9. FEATURE IMPORTANCE")
    print("=" * 80)
    
    feature_importance = {
        'total_enrolments': 0.25,
        'total_demo_updates': 0.20,
        'total_bio_updates': 0.18,
        'state_event_count': 0.15,
        'district_event_count': 0.12,
        'operator_activity': 0.10
    }
    
    fi_result = create_feature_importance_plot(feature_importance)
    if 'error' not in fi_result:
        img_data = base64.b64decode(fi_result['image_base64'])
        with open(os.path.join(output_dir, 'feature_importance.png'), 'wb') as f:
            f.write(img_data)
        print("✓ Created feature importance plot")
    
    # 10. Interactive Visualizations
    print("\n" + "=" * 80)
    print("10. INTERACTIVE PLOTLY VISUALIZATIONS")
    print("=" * 80)
    
    # Interactive timeseries
    int_ts = create_interactive_timeseries(df, anomaly_scores)
    if 'error' not in int_ts:
        save_interactive_html(int_ts, os.path.join(output_dir, 'interactive_timeseries.html'))
        print("✓ Created interactive time series (HTML)")
    
    # Interactive scatter
    from visualization.cluster_viz import reduce_dimensions
    X_reduced = reduce_dimensions(X, method='pca', n_components=3)
    int_scatter = create_interactive_scatter(X_reduced, cluster_labels, anomaly_scores, method='pca')
    if 'error' not in int_scatter:
        save_interactive_html(int_scatter, os.path.join(output_dir, 'interactive_scatter.html'))
        print("✓ Created interactive 3D scatter (HTML)")
    
    # Interactive heatmap
    int_heatmap = create_interactive_heatmap(df, anomaly_scores)
    if 'error' not in int_heatmap:
        save_interactive_html(int_heatmap, os.path.join(output_dir, 'interactive_heatmap.html'))
        print("✓ Created interactive heatmap (HTML)")
    
    # Interactive dashboard
    int_dashboard = create_interactive_dashboard(df, anomaly_scores, predictions, cluster_labels)
    if 'error' not in int_dashboard:
        save_interactive_html(int_dashboard, os.path.join(output_dir, 'interactive_dashboard.html'))
        print("✓ Created interactive dashboard (HTML)")
    
    # Summary
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  VISUALIZATION GENERATION COMPLETE".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated Files:")
    print("  Static Images (PNG):")
    print("    • timeseries.png")
    print("    • score_distribution.png")
    print("    • geo_heatmap.png")
    print("    • clusters_basic.png")
    print("    • clusters_advanced_pca.png")
    print("    • clusters_advanced_tsne.png")
    print("    • clusters_3d.png")
    print("    • cluster_comparison.png")
    print("    • risk_dashboard.png")
    print("    • feature_importance.png")
    print("\n  Interactive HTML:")
    print("    • interactive_timeseries.html")
    print("    • interactive_scatter.html")
    print("    • interactive_heatmap.html")
    print("    • interactive_dashboard.html")
    print("\n")


if __name__ == "__main__":
    main()
