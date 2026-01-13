"""
Distribution and risk indicator visualization.
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


def create_risk_dashboard(
    anomaly_scores: np.ndarray,
    predictions: np.ndarray,
    df: pd.DataFrame,
    model_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create comprehensive risk dashboard with multiple indicators.
    
    Args:
        anomaly_scores: Array of anomaly scores
        predictions: Array of predictions
        df: Original DataFrame
        model_results: Results from individual models
        
    Returns:
        Dict with dashboard data and base64 encoded image
    """
    try:
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Risk Gauge (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        high_risk = (anomaly_scores > 0.8).sum()
        medium_risk = ((anomaly_scores > 0.5) & (anomaly_scores <= 0.8)).sum()
        low_risk = (anomaly_scores <= 0.5).sum()
        total = len(anomaly_scores)
        
        risk_score = float(np.mean(anomaly_scores))
        
        # Gauge visualization
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        ax1.plot(x, y, 'k-', linewidth=3)
        ax1.fill_between(x[:33], y[:33], 0, color='green', alpha=0.3)
        ax1.fill_between(x[33:66], y[33:66], 0, color='orange', alpha=0.3)
        ax1.fill_between(x[66:], y[66:], 0, color='red', alpha=0.3)
        
        # Needle
        needle_angle = np.pi * (1 - risk_score)
        ax1.arrow(0, 0, 0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle),
                 head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-0.2, 1.2)
        ax1.axis('off')
        ax1.set_title(f'🎯 Overall Risk Score: {risk_score:.2f}', fontsize=14, fontweight='bold')
        
        # 2. Risk Distribution Pie (Top Center)
        ax2 = fig.add_subplot(gs[0, 1])
        sizes = [high_risk, medium_risk, low_risk]
        labels = [f'High Risk\n({high_risk})', f'Medium Risk\n({medium_risk})', f'Normal\n({low_risk})']
        colors = ['#ff4444', '#ffaa00', '#44aa44']
        explode = (0.05, 0.02, 0)
        
        if sum(sizes) > 0:
            ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
        ax2.set_title('📊 Risk Distribution', fontsize=12, fontweight='bold')
        
        # 3. Key Metrics (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        metrics_text = f"""
        📈 ANALYSIS SUMMARY
        ─────────────────────
        Total Records: {total:,}
        Anomalies Detected: {(predictions == -1).sum():,}
        Anomaly Rate: {100 * (predictions == -1).sum() / total:.1f}%
        
        🚨 RISK BREAKDOWN
        ─────────────────────
        High Risk (>0.8): {high_risk:,} ({100*high_risk/total:.1f}%)
        Medium Risk (0.5-0.8): {medium_risk:,} ({100*medium_risk/total:.1f}%)
        Low Risk (<0.5): {low_risk:,} ({100*low_risk/total:.1f}%)
        
        📊 STATISTICS
        ─────────────────────
        Mean Score: {np.mean(anomaly_scores):.3f}
        Median Score: {np.median(anomaly_scores):.3f}
        Std Dev: {np.std(anomaly_scores):.3f}
        Max Score: {np.max(anomaly_scores):.3f}
        """
        
        ax3.text(0.1, 0.95, metrics_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # 4. Model Comparison (Middle Left)
        ax4 = fig.add_subplot(gs[1, 0])
        
        if model_results:
            model_names = [r['model_name'] for r in model_results]
            anomaly_counts = [r['anomaly_count'] for r in model_results]
            
            bars = ax4.bar(model_names, anomaly_counts, color=['#3498db', '#e74c3c', '#2ecc71'][:len(model_names)])
            ax4.set_ylabel('Anomalies Detected')
            ax4.set_title('🤖 Model Comparison', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            for bar, count in zip(bars, anomaly_counts):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=10)
        
        # 5. Score Histogram (Middle Center)
        ax5 = fig.add_subplot(gs[1, 1])
        
        ax5.hist(anomaly_scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax5.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Medium Threshold')
        ax5.axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='High Threshold')
        ax5.set_xlabel('Anomaly Score')
        ax5.set_ylabel('Frequency')
        ax5.set_title('📈 Score Distribution', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Top Anomalies Table (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Get top 10 anomalies
        top_indices = np.argsort(anomaly_scores)[-10:][::-1]
        
        table_text = "🔴 TOP 10 HIGHEST RISK RECORDS\n" + "─" * 40 + "\n"
        table_text += f"{'Rank':<6}{'Score':<10}{'Risk Level':<15}\n"
        table_text += "─" * 40 + "\n"
        
        for rank, idx in enumerate(top_indices, 1):
            score = anomaly_scores[idx]
            risk = "🔴 HIGH" if score > 0.8 else "🟠 MEDIUM" if score > 0.5 else "🟢 LOW"
            table_text += f"{rank:<6}{score:<10.3f}{risk:<15}\n"
        
        ax6.text(0.1, 0.95, table_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # 7-9. Time-based analysis (Bottom row)
        if 'date' in df.columns or 'month' in df.columns:
            ax7 = fig.add_subplot(gs[2, :])
            
            if 'month' in df.columns:
                df_temp = df.copy()
                df_temp['anomaly_score'] = anomaly_scores
                monthly_stats = df_temp.groupby('month').agg({
                    'anomaly_score': ['mean', 'count']
                }).reset_index()
                monthly_stats.columns = ['month', 'mean_score', 'count']
                
                ax7_twin = ax7.twinx()
                
                bars = ax7.bar(monthly_stats['month'], monthly_stats['count'], alpha=0.3, color='steelblue', label='Record Count')
                line = ax7_twin.plot(monthly_stats['month'], monthly_stats['mean_score'], 'ro-', linewidth=2, markersize=8, label='Mean Anomaly Score')
                
                ax7_twin.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7)
                ax7_twin.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
                
                ax7.set_xlabel('Month')
                ax7.set_ylabel('Record Count', color='steelblue')
                ax7_twin.set_ylabel('Mean Anomaly Score', color='red')
                ax7.set_title('📅 Monthly Anomaly Trend', fontsize=12, fontweight='bold')
                ax7.legend(loc='upper left')
                ax7_twin.legend(loc='upper right')
                ax7.grid(True, alpha=0.3)
        else:
            ax7 = fig.add_subplot(gs[2, :])
            ax7.text(0.5, 0.5, 'No temporal data available for trend analysis',
                    ha='center', va='center', fontsize=14, color='gray')
            ax7.axis('off')
        
        plt.suptitle('🛡️ UIDAI Fraud Detection Dashboard', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        # Prepare dashboard data
        dashboard_data = {
            "chart_type": "risk_dashboard",
            "title": "UIDAI Fraud Detection Dashboard",
            "data": {
                "summary": {
                    "total_records": int(total),
                    "total_anomalies": int((predictions == -1).sum()),
                    "anomaly_rate": float(100 * (predictions == -1).sum() / total),
                    "overall_risk_score": float(risk_score)
                },
                "risk_distribution": {
                    "high_risk": int(high_risk),
                    "medium_risk": int(medium_risk),
                    "low_risk": int(low_risk)
                },
                "statistics": {
                    "mean": float(np.mean(anomaly_scores)),
                    "median": float(np.median(anomaly_scores)),
                    "std": float(np.std(anomaly_scores)),
                    "min": float(np.min(anomaly_scores)),
                    "max": float(np.max(anomaly_scores))
                },
                "top_anomalies": [
                    {"index": int(idx), "score": float(anomaly_scores[idx])}
                    for idx in top_indices
                ],
                "model_comparison": model_results
            },
            "image_base64": image_base64
        }
        
        logger.info("Created risk dashboard")
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error creating risk dashboard: {e}")
        return {"error": str(e), "chart_type": "risk_dashboard"}


def create_comparison_plot(
    normal_data: np.ndarray,
    anomaly_data: np.ndarray,
    feature_names: List[str],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Create comparison between normal and anomalous records.
    
    Args:
        normal_data: Data for normal records
        anomaly_data: Data for anomalous records
        feature_names: Names of features
        top_k: Number of features to show
        
    Returns:
        Dict with chart data and base64 encoded image
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate mean values for each group
        normal_means = np.mean(normal_data, axis=0) if len(normal_data) > 0 else np.zeros(len(feature_names))
        anomaly_means = np.mean(anomaly_data, axis=0) if len(anomaly_data) > 0 else np.zeros(len(feature_names))
        
        # Calculate difference
        diff = np.abs(anomaly_means - normal_means)
        
        # Sort by difference and take top_k
        top_indices = np.argsort(diff)[-top_k:][::-1]
        
        top_features = [feature_names[i] for i in top_indices if i < len(feature_names)]
        top_normal = [normal_means[i] for i in top_indices if i < len(normal_means)]
        top_anomaly = [anomaly_means[i] for i in top_indices if i < len(anomaly_means)]
        
        x = np.arange(len(top_features))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, top_normal, width, label='Normal Records', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, top_anomaly, width, label='Anomalous Records', color='red', alpha=0.7)
        
        ax.set_ylabel('Mean Value (scaled)')
        ax.set_title('📊 Normal vs Anomalous Record Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace('_', '\n') for f in top_features], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        chart_data = {
            "chart_type": "comparison",
            "title": "Normal vs Anomalous Record Comparison",
            "data": {
                "features": top_features,
                "normal_means": top_normal,
                "anomaly_means": top_anomaly,
                "n_normal": len(normal_data),
                "n_anomaly": len(anomaly_data)
            },
            "image_base64": image_base64
        }
        
        logger.info("Created comparison plot")
        return chart_data
        
    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")
        return {"error": str(e), "chart_type": "comparison"}
