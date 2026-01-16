"""
Biometric Re-enrollment Risk Service

This service provides comprehensive analysis for predicting which residents
need proactive biometric re-capture to avoid authentication failures.

PRIVACY NOTICE:
- Uses ONLY aggregated data from public government APIs
- No individual Aadhaar numbers or PII processed
- All operations on state/district/age-group level
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import io
import base64

# ML imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from services.data_gov_client import data_client
from services.feature_engineering import feature_engineer

logger = logging.getLogger(__name__)


class BiometricRiskService:
    """
    Service for analyzing biometric re-enrollment risk.
    
    Predicts which regions/demographics have high risk of biometric
    template aging and need proactive re-enrollment outreach.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained_model = None
        self.model_type = None
        self.analysis_results = {}
        self.visualizations = {}
        
    async def run_full_analysis(
        self,
        state_filter: Optional[str] = None,
        district_filter: Optional[str] = None,
        age_range: str = "all",  # "elderly", "adult", "all"
        risk_threshold: float = 0.5,
        records_limit: int = 5000
    ) -> Dict[str, Any]:
        """
        Run complete biometric re-enrollment risk analysis.
        
        Args:
            state_filter: Optional state to filter data
            district_filter: Optional district to filter
            age_range: Age group to focus on
            risk_threshold: Threshold for high-risk classification (0.3-0.9)
            records_limit: Max records to analyze
            
        Returns:
            Comprehensive analysis results with visualizations
        """
        logger.info(f"Starting biometric risk analysis: state={state_filter}, age={age_range}, threshold={risk_threshold}")
        
        try:
            # Step 1: Fetch data from government APIs
            data = await self._fetch_data(state_filter, district_filter, records_limit)
            
            if data['total_records'] == 0:
                return {
                    "success": False,
                    "error": "No data available for the selected filters",
                    "records_analyzed": 0
                }
            
            # Step 2: Engineer features
            features_df = self._engineer_features(data, age_range)
            
            # Step 3: Train risk prediction model
            model_results = self._train_risk_model(features_df, risk_threshold)
            
            # Step 4: Identify high-risk regions
            high_risk_regions = self._identify_high_risk_regions(features_df, risk_threshold)
            
            # Step 5: Cluster analysis for demographic groups
            cluster_analysis = self._cluster_demographics(features_df)
            
            # Step 6: Generate visualizations
            visualizations = self._generate_visualizations(features_df, high_risk_regions)
            
            # Step 7: Generate recommendations
            recommendations = self._generate_recommendations(high_risk_regions, cluster_analysis)
            
            # Compile final results
            results = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "state_filter": state_filter,
                    "district_filter": district_filter,
                    "age_range": age_range,
                    "risk_threshold": risk_threshold,
                    "records_limit": records_limit
                },
                "summary": {
                    "total_records_analyzed": data['total_records'],
                    "regions_analyzed": len(features_df),
                    "high_risk_count": len(high_risk_regions),
                    "average_risk_score": float(features_df['risk_score'].mean()) if 'risk_score' in features_df.columns else 0,
                    "model_accuracy": model_results.get('accuracy', 0)
                },
                "model_metrics": model_results,
                "high_risk_regions": high_risk_regions[:20],  # Top 20
                "cluster_analysis": cluster_analysis,
                "recommendations": recommendations,
                "visualizations": visualizations,
                "feature_importance": model_results.get('feature_importance', {})
            }
            
            self.analysis_results = results
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "records_analyzed": 0
            }
    
    async def _fetch_data(
        self,
        state_filter: Optional[str],
        district_filter: Optional[str],
        limit: int
    ) -> Dict[str, Any]:
        """Fetch data from all three government APIs"""
        
        filters = {}
        if state_filter:
            filters['state'] = state_filter
        if district_filter:
            filters['district'] = district_filter
            
        # Fetch from all datasets
        enrolment_data = data_client.fetch_data('enrolment', limit=limit, filters=filters if filters else None)
        demographic_data = data_client.fetch_data('demographic', limit=limit, filters=filters if filters else None)
        biometric_data = data_client.fetch_data('biometric', limit=limit, filters=filters if filters else None)
        
        total_records = 0
        
        enrolment_df = pd.DataFrame()
        demographic_df = pd.DataFrame()
        biometric_df = pd.DataFrame()
        
        if enrolment_data.get('success') and enrolment_data.get('records'):
            enrolment_df = pd.DataFrame(enrolment_data['records'])
            total_records += len(enrolment_df)
            
        if demographic_data.get('success') and demographic_data.get('records'):
            demographic_df = pd.DataFrame(demographic_data['records'])
            total_records += len(demographic_df)
            
        if biometric_data.get('success') and biometric_data.get('records'):
            biometric_df = pd.DataFrame(biometric_data['records'])
            total_records += len(biometric_df)
        
        return {
            'enrolment': enrolment_df,
            'demographic': demographic_df,
            'biometric': biometric_df,
            'total_records': total_records
        }
    
    def _engineer_features(self, data: Dict[str, pd.DataFrame], age_range: str) -> pd.DataFrame:
        """Engineer features for risk prediction"""
        
        enrolment_df = data['enrolment']
        demographic_df = data['demographic']
        biometric_df = data['biometric']
        
        # Process each dataset
        if not enrolment_df.empty:
            enrolment_df = feature_engineer.process_enrolment_data(enrolment_df)
        
        if not demographic_df.empty:
            demographic_df = feature_engineer.process_demographic_data(demographic_df)
            
        if not biometric_df.empty:
            biometric_df = feature_engineer.process_biometric_data(biometric_df)
        
        # Create comprehensive risk features
        features_df = feature_engineer.create_risk_features(
            enrolment_df=enrolment_df if not enrolment_df.empty else None,
            demographic_df=demographic_df if not demographic_df.empty else None,
            biometric_df=biometric_df if not biometric_df.empty else None,
            aggregation_level='state'
        )
        
        # Filter by age range if specified
        if age_range == "elderly" and 'elderly_ratio' in features_df.columns:
            # Focus on regions with high elderly population
            features_df = features_df[features_df['elderly_ratio'] > 0.15]
        elif age_range == "adult" and 'adult_ratio' in features_df.columns:
            features_df = features_df[features_df['adult_ratio'] > 0.5]
        
        # Create proxy risk labels if not available
        if 'risk_category' not in features_df.columns:
            features_df = feature_engineer.create_proxy_risk_labels(features_df)
        
        # Calculate risk score
        features_df['risk_score'] = self._calculate_risk_score(features_df)
        
        return features_df
    
    def _calculate_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite risk score based on multiple factors"""
        
        risk_score = pd.Series(0.5, index=df.index)  # Default baseline
        
        # Factor 1: Low biometric update ratio (high weight)
        if 'biometric_update_ratio' in df.columns:
            bio_ratio = df['biometric_update_ratio'].fillna(df['biometric_update_ratio'].median())
            risk_score += (1 - bio_ratio.clip(0, 1)) * 0.3
        
        # Factor 2: High elderly population (medium weight)
        if 'elderly_ratio' in df.columns:
            elderly = df['elderly_ratio'].fillna(0)
            risk_score += elderly.clip(0, 1) * 0.2
        
        # Factor 3: High-risk state flag
        if 'is_high_risk_state' in df.columns:
            risk_score += df['is_high_risk_state'].fillna(0) * 0.15
        
        # Factor 4: Low demographic update ratio
        if 'demographic_update_ratio' in df.columns:
            demo_ratio = df['demographic_update_ratio'].fillna(df['demographic_update_ratio'].median())
            risk_score += (1 - demo_ratio.clip(0, 1)) * 0.15
        
        # Factor 5: Time since last update (proxy via update frequency)
        if 'update_frequency' in df.columns:
            freq = df['update_frequency'].fillna(df['update_frequency'].median())
            freq_normalized = (freq - freq.min()) / (freq.max() - freq.min() + 1e-6)
            risk_score += (1 - freq_normalized) * 0.1
        
        return risk_score.clip(0, 1)
    
    def _train_risk_model(self, df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """Train ML model for risk classification"""
        
        if len(df) < 10:
            return {"error": "Insufficient data for model training", "accuracy": 0}
        
        # Prepare features
        exclude_cols = ['state', 'district', 'risk_category', 'risk_score', 'proxy_risk_score']
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
        
        if not feature_cols:
            return {"error": "No numeric features available", "accuracy": 0}
        
        X = df[feature_cols].fillna(0).values
        
        # Create binary labels based on threshold
        y = (df['risk_score'] >= threshold).astype(int).values
        
        # Handle class imbalance
        if len(np.unique(y)) < 2:
            # If only one class, use median as threshold
            median_risk = df['risk_score'].median()
            y = (df['risk_score'] >= median_risk).astype(int).values
        
        # Split data
        if len(df) < 20:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        try:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_train_scaled, y_train)
            self.trained_model = model
            self.model_type = 'xgboost'
        except Exception as e:
            # Fallback to Random Forest
            logger.warning(f"XGBoost failed, using Random Forest: {e}")
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train_scaled, y_train)
            self.trained_model = model
            self.model_type = 'random_forest'
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for name, imp in zip(feature_cols, model.feature_importances_):
                feature_importance[name] = float(imp)
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "model_type": self.model_type,
            "accuracy": float(accuracy),
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "high_risk_count": int(y.sum()),
            "low_risk_count": int(len(y) - y.sum()),
            "feature_importance": feature_importance,
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }
    
    def _identify_high_risk_regions(self, df: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
        """Identify regions with high re-enrollment risk"""
        
        high_risk = df[df['risk_score'] >= threshold].copy()
        high_risk = high_risk.sort_values('risk_score', ascending=False)
        
        regions = []
        for _, row in high_risk.iterrows():
            region = {
                'state': row.get('state', 'Unknown'),
                'risk_score': float(row['risk_score']),
                'risk_level': 'Critical' if row['risk_score'] >= 0.75 else 'High' if row['risk_score'] >= 0.6 else 'Moderate',
                'factors': []
            }
            
            # Add contributing factors
            if 'biometric_update_ratio' in row and row['biometric_update_ratio'] < 0.3:
                region['factors'].append('Low biometric update rate')
            if 'elderly_ratio' in row and row['elderly_ratio'] > 0.2:
                region['factors'].append('High elderly population')
            if 'is_high_risk_state' in row and row['is_high_risk_state']:
                region['factors'].append('Known high-risk state')
            if 'demographic_update_ratio' in row and row['demographic_update_ratio'] < 0.3:
                region['factors'].append('Low demographic update rate')
                
            # Add metrics
            if 'total_enrollments' in row:
                region['total_enrollments'] = int(row['total_enrollments'])
            if 'biometric_update_ratio' in row:
                region['biometric_update_ratio'] = float(row['biometric_update_ratio'])
            if 'elderly_ratio' in row:
                region['elderly_ratio'] = float(row['elderly_ratio'])
                
            regions.append(region)
        
        return regions
    
    def _cluster_demographics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cluster demographic groups to identify patterns"""
        
        cluster_cols = ['elderly_ratio', 'adult_ratio', 'biometric_update_ratio', 'demographic_update_ratio']
        available_cols = [c for c in cluster_cols if c in df.columns]
        
        if len(available_cols) < 2 or len(df) < 5:
            return {"clusters": [], "message": "Insufficient data for clustering"}
        
        X_cluster = df[available_cols].fillna(0).values
        
        # Determine optimal clusters (max 5)
        n_clusters = min(5, len(df) // 2)
        if n_clusters < 2:
            n_clusters = 2
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_cluster)
        
        # Analyze each cluster
        clusters = []
        for i in range(n_clusters):
            cluster_data = df[df['cluster'] == i]
            cluster_info = {
                'cluster_id': i,
                'size': len(cluster_data),
                'avg_risk_score': float(cluster_data['risk_score'].mean()),
                'characteristics': {}
            }
            
            for col in available_cols:
                cluster_info['characteristics'][col] = float(cluster_data[col].mean())
            
            # Label cluster based on characteristics
            if cluster_info['avg_risk_score'] >= 0.7:
                cluster_info['label'] = 'Critical Risk - Immediate Attention'
            elif cluster_info['avg_risk_score'] >= 0.5:
                cluster_info['label'] = 'High Risk - Priority Outreach'
            elif cluster_info['avg_risk_score'] >= 0.3:
                cluster_info['label'] = 'Moderate Risk - Monitor'
            else:
                cluster_info['label'] = 'Low Risk - Standard Processing'
                
            clusters.append(cluster_info)
        
        # Sort by risk
        clusters.sort(key=lambda x: x['avg_risk_score'], reverse=True)
        
        return {
            "n_clusters": n_clusters,
            "clusters": clusters
        }
    
    def _generate_visualizations(self, df: pd.DataFrame, high_risk_regions: List[Dict]) -> Dict[str, str]:
        """Generate visualization charts as base64 images"""
        
        visualizations = {}
        
        try:
            # 1. Risk Distribution by State
            visualizations['risk_distribution'] = self._plot_risk_distribution(df)
            
            # 2. Age Group Risk Heatmap
            visualizations['age_risk_heatmap'] = self._plot_age_risk_heatmap(df)
            
            # 3. Biometric Update Ratio Chart
            visualizations['biometric_ratio'] = self._plot_biometric_ratio(df)
            
            # 4. Feature Importance Chart
            if self.trained_model and hasattr(self.trained_model, 'feature_importances_'):
                visualizations['feature_importance'] = self._plot_feature_importance()
            
            # 5. Risk Categories Pie Chart
            visualizations['risk_categories'] = self._plot_risk_categories(df)
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _plot_risk_distribution(self, df: pd.DataFrame) -> str:
        """Plot risk score distribution by state"""
        plt.figure(figsize=(12, 6))
        
        if 'state' in df.columns and 'risk_score' in df.columns:
            plot_data = df.nlargest(15, 'risk_score')[['state', 'risk_score']]
            colors = ['#ff4444' if x >= 0.7 else '#ffaa00' if x >= 0.5 else '#00cc66' for x in plot_data['risk_score']]
            
            bars = plt.barh(plot_data['state'], plot_data['risk_score'], color=colors)
            plt.xlabel('Risk Score', fontsize=12)
            plt.ylabel('State', fontsize=12)
            plt.title('Biometric Re-enrollment Risk by State (Top 15)', fontsize=14, fontweight='bold')
            plt.xlim(0, 1)
            
            # Add value labels
            for bar, val in zip(bars, plot_data['risk_score']):
                plt.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=10)
        else:
            plt.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        return self._fig_to_base64(plt.gcf())
    
    def _plot_age_risk_heatmap(self, df: pd.DataFrame) -> str:
        """Plot heatmap of risk by age groups"""
        plt.figure(figsize=(10, 8))
        
        age_cols = ['elderly_ratio', 'adult_ratio']
        available_cols = [c for c in age_cols if c in df.columns]
        
        if available_cols and 'state' in df.columns and 'risk_score' in df.columns:
            plot_data = df[['state'] + available_cols + ['risk_score']].nlargest(12, 'risk_score')
            plot_data = plot_data.set_index('state')[available_cols]
            
            sns.heatmap(plot_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Ratio'})
            plt.title('Age Distribution in High-Risk Regions', fontsize=14, fontweight='bold')
            plt.xlabel('Age Group Ratio', fontsize=12)
            plt.ylabel('State', fontsize=12)
        else:
            plt.text(0.5, 0.5, 'Insufficient age data', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        return self._fig_to_base64(plt.gcf())
    
    def _plot_biometric_ratio(self, df: pd.DataFrame) -> str:
        """Plot biometric update ratios"""
        plt.figure(figsize=(12, 6))
        
        if 'state' in df.columns and 'biometric_update_ratio' in df.columns:
            plot_data = df.nsmallest(15, 'biometric_update_ratio')[['state', 'biometric_update_ratio']]
            colors = ['#ff4444' if x < 0.3 else '#ffaa00' if x < 0.5 else '#00cc66' for x in plot_data['biometric_update_ratio']]
            
            bars = plt.barh(plot_data['state'], plot_data['biometric_update_ratio'], color=colors)
            plt.xlabel('Biometric Update Ratio', fontsize=12)
            plt.ylabel('State', fontsize=12)
            plt.title('States with Lowest Biometric Update Rates', fontsize=14, fontweight='bold')
            plt.xlim(0, 1)
            
            for bar, val in zip(bars, plot_data['biometric_update_ratio']):
                plt.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=10)
        else:
            plt.text(0.5, 0.5, 'Insufficient biometric data', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        return self._fig_to_base64(plt.gcf())
    
    def _plot_feature_importance(self) -> str:
        """Plot feature importance from trained model"""
        plt.figure(figsize=(10, 6))
        
        if hasattr(self.trained_model, 'feature_importances_'):
            importance = self.trained_model.feature_importances_
            # Get feature names from feature engineer if available
            features = [f'Feature_{i}' for i in range(len(importance))]
            
            sorted_idx = np.argsort(importance)[-10:]  # Top 10
            pos = np.arange(len(sorted_idx))
            
            plt.barh(pos, importance[sorted_idx], color='#7b2ff7')
            plt.yticks(pos, [features[i] for i in sorted_idx])
            plt.xlabel('Importance', fontsize=12)
            plt.title('Top Feature Importance for Risk Prediction', fontsize=14, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'Model not trained', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        return self._fig_to_base64(plt.gcf())
    
    def _plot_risk_categories(self, df: pd.DataFrame) -> str:
        """Plot pie chart of risk categories"""
        plt.figure(figsize=(8, 8))
        
        if 'risk_score' in df.columns:
            critical = len(df[df['risk_score'] >= 0.75])
            high = len(df[(df['risk_score'] >= 0.5) & (df['risk_score'] < 0.75)])
            moderate = len(df[(df['risk_score'] >= 0.3) & (df['risk_score'] < 0.5)])
            low = len(df[df['risk_score'] < 0.3])
            
            sizes = [critical, high, moderate, low]
            labels = ['Critical (≥0.75)', 'High (0.5-0.75)', 'Moderate (0.3-0.5)', 'Low (<0.3)']
            colors = ['#ff4444', '#ff8800', '#ffcc00', '#00cc66']
            explode = (0.05, 0.02, 0, 0)
            
            # Filter out zero values
            non_zero = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colors, explode) if s > 0]
            if non_zero:
                sizes, labels, colors, explode = zip(*non_zero)
                plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%',
                       shadow=True, startangle=90)
            
            plt.title('Risk Category Distribution', fontsize=14, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        return self._fig_to_base64(plt.gcf())
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"
    
    def _generate_recommendations(self, high_risk_regions: List[Dict], cluster_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Priority 1: Critical risk regions
        critical_regions = [r for r in high_risk_regions if r.get('risk_level') == 'Critical']
        if critical_regions:
            recommendations.append({
                'priority': 1,
                'category': 'Immediate Action Required',
                'title': 'Deploy Emergency Re-enrollment Camps',
                'description': f'{len(critical_regions)} regions have critical biometric failure risk (>75%). Immediate mobile camp deployment recommended.',
                'regions': [r['state'] for r in critical_regions[:5]],
                'actions': [
                    'Deploy mobile biometric update units',
                    'Increase operator capacity by 50%',
                    'Extend enrollment center hours',
                    'Launch awareness campaign'
                ]
            })
        
        # Priority 2: High elderly population areas
        elderly_regions = [r for r in high_risk_regions if 'High elderly population' in r.get('factors', [])]
        if elderly_regions:
            recommendations.append({
                'priority': 2,
                'category': 'Vulnerable Population Focus',
                'title': 'Senior Citizen Outreach Program',
                'description': f'{len(elderly_regions)} regions have high elderly populations with aging biometrics.',
                'regions': [r['state'] for r in elderly_regions[:5]],
                'actions': [
                    'Home visit program for immobile seniors',
                    'Partner with senior citizen associations',
                    'Simplified re-enrollment process',
                    'Iris-only capture option for fingerprint issues'
                ]
            })
        
        # Priority 3: Low update rate regions
        low_update_regions = [r for r in high_risk_regions if 'Low biometric update rate' in r.get('factors', [])]
        if low_update_regions:
            recommendations.append({
                'priority': 3,
                'category': 'Infrastructure Enhancement',
                'title': 'Increase Update Center Capacity',
                'description': f'{len(low_update_regions)} regions have critically low biometric update rates.',
                'regions': [r['state'] for r in low_update_regions[:5]],
                'actions': [
                    'Add more enrollment operators',
                    'Setup additional Aadhaar Seva Kendras',
                    'Enable online appointment booking',
                    'Extended operating hours'
                ]
            })
        
        # Priority 4: Based on cluster analysis
        if cluster_analysis.get('clusters'):
            critical_clusters = [c for c in cluster_analysis['clusters'] if c.get('avg_risk_score', 0) >= 0.6]
            if critical_clusters:
                recommendations.append({
                    'priority': 4,
                    'category': 'Targeted Intervention',
                    'title': 'Demographic-Specific Programs',
                    'description': f'{len(critical_clusters)} demographic clusters identified with elevated risk patterns.',
                    'clusters': [c['label'] for c in critical_clusters],
                    'actions': [
                        'Design targeted awareness campaigns',
                        'Customize outreach by demographics',
                        'Partner with local government',
                        'Track progress with monthly reports'
                    ]
                })
        
        # Default recommendation if no high-risk found
        if not recommendations:
            recommendations.append({
                'priority': 1,
                'category': 'Maintenance',
                'title': 'Continue Regular Monitoring',
                'description': 'No critical risks detected. Maintain regular biometric update schedules.',
                'actions': [
                    'Continue quarterly data analysis',
                    'Monitor aging biometric templates',
                    'Proactive outreach for 5+ year old enrollments'
                ]
            })
        
        return recommendations


# Singleton instance
biometric_risk_service = BiometricRiskService()
