"""
Interactive plotly visualizations for anomaly detection.

Provides interactive versions of all visualizations with zoom, pan,
hover tooltips, and export to HTML functionality.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

logger = logging.getLogger(__name__)


def create_interactive_timeseries(
    df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    date_column: str = 'date',
    value_columns: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Create interactive time series plot with plotly.
    
    Args:
        df: DataFrame with time series data
        anomaly_scores: Array of anomaly scores
        date_column: Name of date column
        value_columns: Columns to plot
        threshold: Anomaly threshold
        
    Returns:
        Dict with plotly figure and HTML
    """
    try:
        df = df.copy()
        
        # Parse dates
        if date_column in df.columns:
            df['date_parsed'] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.dropna(subset=['date_parsed'])
            df = df.sort_values('date_parsed')
        else:
            df['date_parsed'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # Auto-detect value columns
        if value_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_columns = [c for c in numeric_cols if 'encoded' not in c][:3]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Event Volume Over Time', 'Anomaly Score Timeline'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # Plot 1: Time series with anomaly highlights
        for col in value_columns[:3]:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['date_parsed'],
                        y=df[col],
                        name=col.replace('_', ' ').title(),
                        mode='lines',
                        line=dict(width=2),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                     'Date: %{x}<br>' +
                                     'Value: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Add anomaly markers
        if len(anomaly_scores) == len(df):
            anomaly_mask = anomaly_scores > threshold
            if anomaly_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[anomaly_mask, 'date_parsed'],
                        y=[df[value_columns[0]].max() * 1.05] * anomaly_mask.sum(),
                        mode='markers',
                        name='Anomaly',
                        marker=dict(symbol='x', size=10, color='red'),
                        hovertemplate='<b>Anomaly Detected</b><br>' +
                                     'Date: %{x}<br>' +
                                     'Score: %{customdata:.3f}<extra></extra>',
                        customdata=anomaly_scores[anomaly_mask]
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Anomaly scores
        if len(anomaly_scores) == len(df):
            fig.add_trace(
                go.Scatter(
                    x=df['date_parsed'],
                    y=anomaly_scores,
                    name='Anomaly Score',
                    fill='tozeroy',
                    line=dict(color='orange', width=2),
                    hovertemplate='<b>Anomaly Score</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Score: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add threshold lines
            fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                         annotation_text=f"Threshold ({threshold})",
                         row=2, col=1)
            fig.add_hline(y=0.8, line_dash="dot", line_color="darkred",
                         annotation_text="High Risk (0.8)",
                         row=2, col=1)
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Anomaly Score", range=[0, 1], row=2, col=1)
        
        fig.update_layout(
            title_text="📈 Interactive Time Series Analysis",
            height=700,
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )
        
        # Convert to HTML
        html = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        return {
            "chart_type": "interactive_timeseries",
            "title": "Interactive Time Series Analysis",
            "plotly_json": json.loads(fig.to_json()),
            "html": html
        }
        
    except Exception as e:
        logger.error(f"Error creating interactive timeseries: {e}")
        return {"error": str(e), "chart_type": "interactive_timeseries"}


def create_interactive_scatter(
    X_reduced: np.ndarray,
    cluster_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    method: str = 'tsne'
) -> Dict[str, Any]:
    """
    Create interactive 3D scatter plot.
    
    Args:
        X_reduced: Reduced dimensional data (2D or 3D)
        cluster_labels: Cluster labels
        anomaly_scores: Anomaly scores
        method: Dimensionality reduction method name
        
    Returns:
        Dict with plotly figure and HTML
    """
    try:
        df_plot = pd.DataFrame({
            'x': X_reduced[:, 0],
            'y': X_reduced[:, 1],
            'cluster': cluster_labels,
            'anomaly_score': anomaly_scores,
            'is_outlier': cluster_labels == -1
        })
        
        if X_reduced.shape[1] >= 3:
            df_plot['z'] = X_reduced[:, 2]
            
            # 3D scatter
            fig = px.scatter_3d(
                df_plot,
                x='x', y='y', z='z',
                color='anomaly_score',
                symbol='is_outlier',
                color_continuous_scale='RdYlGn_r',
                hover_data={'cluster': True, 'anomaly_score': ':.3f'},
                labels={
                    'x': f'{method.upper()} 1',
                    'y': f'{method.upper()} 2',
                    'z': f'{method.upper()} 3',
                    'anomaly_score': 'Anomaly Score'
                },
                title=f'🔬 3D Interactive Cluster Visualization ({method.upper()})'
            )
            
            fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))
            
        else:
            # 2D scatter
            fig = px.scatter(
                df_plot,
                x='x', y='y',
                color='anomaly_score',
                symbol='is_outlier',
                color_continuous_scale='RdYlGn_r',
                hover_data={'cluster': True, 'anomaly_score': ':.3f'},
                labels={
                    'x': f'{method.upper()} 1',
                    'y': f'{method.upper()} 2',
                    'anomaly_score': 'Anomaly Score'
                },
                title=f'🔬 Interactive Cluster Visualization ({method.upper()})'
            )
            
            fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
        
        fig.update_layout(
            height=600,
            template='plotly_white',
            hovermode='closest'
        )
        
        html = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        return {
            "chart_type": "interactive_scatter",
            "title": f"Interactive Cluster Visualization ({method.upper()})",
            "plotly_json": json.loads(fig.to_json()),
            "html": html
        }
        
    except Exception as e:
        logger.error(f"Error creating interactive scatter: {e}")
        return {"error": str(e), "chart_type": "interactive_scatter"}


def create_interactive_heatmap(
    df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    state_column: str = 'state'
) -> Dict[str, Any]:
    """
    Create interactive choropleth map of India.
    
    Args:
        df: DataFrame with geographic data
        anomaly_scores: Anomaly scores
        state_column: Name of state column
        
    Returns:
        Dict with plotly figure and HTML
    """
    try:
        df = df.copy()
        df['anomaly_score'] = anomaly_scores
        
        # Aggregate by state
        state_stats = df.groupby(state_column).agg({
            'anomaly_score': ['mean', 'max', 'count']
        }).reset_index()
        state_stats.columns = ['state', 'mean_score', 'max_score', 'count']
        
        # Create bar chart (since we don't have geojson for India states)
        fig = go.Figure()
        
        # Sort by mean score
        state_stats = state_stats.sort_values('mean_score', ascending=True)
        
        fig.add_trace(go.Bar(
            y=state_stats['state'],
            x=state_stats['mean_score'],
            orientation='h',
            marker=dict(
                color=state_stats['mean_score'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Mean<br>Anomaly<br>Score")
            ),
            text=state_stats['count'],
            texttemplate='n=%{text}',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Mean Score: %{x:.3f}<br>' +
                         'Max Score: %{customdata[0]:.3f}<br>' +
                         'Records: %{customdata[1]}<extra></extra>',
            customdata=state_stats[['max_score', 'count']].values
        ))
        
        # Add threshold lines
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange",
                     annotation_text="Medium Risk")
        fig.add_vline(x=0.8, line_dash="dash", line_color="red",
                     annotation_text="High Risk")
        
        fig.update_layout(
            title_text="🗺️ Interactive Geographic Anomaly Distribution",
            xaxis_title="Mean Anomaly Score",
            yaxis_title="State",
            height=max(400, len(state_stats) * 20),
            template='plotly_white',
            showlegend=False
        )
        
        html = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        return {
            "chart_type": "interactive_heatmap",
            "title": "Interactive Geographic Distribution",
            "plotly_json": json.loads(fig.to_json()),
            "html": html,
            "data": state_stats.to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"Error creating interactive heatmap: {e}")
        return {"error": str(e), "chart_type": "interactive_heatmap"}


def create_interactive_dashboard(
    df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    predictions: np.ndarray,
    cluster_labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Create comprehensive interactive dashboard.
    
    Args:
        df: DataFrame with data
        anomaly_scores: Anomaly scores
        predictions: Predictions
        cluster_labels: Optional cluster labels
        
    Returns:
        Dict with plotly figure and HTML
    """
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Risk Distribution',
                'Score Distribution',
                'Top Risk States',
                'Monthly Trend'
            ),
            specs=[
                [{'type': 'pie'}, {'type': 'histogram'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. Risk Distribution Pie
        high_risk = (anomaly_scores > 0.8).sum()
        medium_risk = ((anomaly_scores > 0.5) & (anomaly_scores <= 0.8)).sum()
        low_risk = (anomaly_scores <= 0.5).sum()
        
        fig.add_trace(
            go.Pie(
                labels=['High Risk', 'Medium Risk', 'Normal'],
                values=[high_risk, medium_risk, low_risk],
                marker=dict(colors=['#ff4444', '#ffaa00', '#44aa44']),
                hovertemplate='<b>%{label}</b><br>' +
                             'Count: %{value}<br>' +
                             'Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Score Distribution Histogram
        fig.add_trace(
            go.Histogram(
                x=anomaly_scores,
                nbinsx=30,
                marker=dict(color='steelblue', line=dict(color='black', width=1)),
                hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Top Risk States
        if 'state' in df.columns:
            df_temp = df.copy()
            df_temp['anomaly_score'] = anomaly_scores
            state_stats = df_temp.groupby('state')['anomaly_score'].mean().sort_values(ascending=False).head(10)
            
            fig.add_trace(
                go.Bar(
                    x=state_stats.values,
                    y=state_stats.index,
                    orientation='h',
                    marker=dict(color=state_stats.values, colorscale='RdYlGn_r'),
                    hovertemplate='<b>%{y}</b><br>Mean Score: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Monthly Trend
        if 'month' in df.columns:
            df_temp = df.copy()
            df_temp['anomaly_score'] = anomaly_scores
            monthly = df_temp.groupby('month')['anomaly_score'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=monthly.index,
                    y=monthly.values,
                    mode='lines+markers',
                    line=dict(color='red', width=3),
                    marker=dict(size=10),
                    hovertemplate='Month: %{x}<br>Mean Score: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="🛡️ Interactive Anomaly Detection Dashboard",
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        
        return {
            "chart_type": "interactive_dashboard",
            "title": "Interactive Dashboard",
            "plotly_json": json.loads(fig.to_json()),
            "html": html
        }
        
    except Exception as e:
        logger.error(f"Error creating interactive dashboard: {e}")
        return {"error": str(e), "chart_type": "interactive_dashboard"}


def save_interactive_html(
    fig_data: Dict[str, Any],
    filepath: str
) -> bool:
    """
    Save interactive visualization to HTML file.
    
    Args:
        fig_data: Figure data from create_interactive_* functions
        filepath: Path to save HTML file
        
    Returns:
        True if successful
    """
    try:
        if 'html' in fig_data:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fig_data['html'])
            logger.info(f"Saved interactive visualization to {filepath}")
            return True
        else:
            logger.error("No HTML data in figure")
            return False
    except Exception as e:
        logger.error(f"Error saving HTML: {e}")
        return False
