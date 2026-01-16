"""
Biometric Re-enrollment Risk Service

This service provides comprehensive analysis for predicting which residents
need proactive biometric re-capture to avoid authentication failures.

PRIVACY NOTICE:
- Uses ONLY aggregated data from public government APIs
- No individual Aadhaar numbers or PII processed
- All operations on state/district/age-group level

Enhanced Features (ChatGPT Specifications):
- SHAP Explainability (global & local)
- Survival Curve Analysis (Kaplan-Meier style)
- Centre Performance Dashboard
- 5-tier Age Bucket Analysis
- Threshold Sensitivity Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import io
import base64
import hashlib

# ML imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
import xgboost as xgb

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from .data_gov_client import data_client
from .feature_engineering import feature_engineer

logger = logging.getLogger(__name__)


class BiometricRiskService:
    """
    Service for analyzing biometric re-enrollment risk.
    
    Predicts which regions/demographics have high risk of biometric
    template aging and need proactive re-enrollment outreach.
    
    Enhanced with:
    - SHAP Explainability
    - Survival Curve Analysis
    - Centre Performance Dashboard
    - 5-tier Age Bucket Analysis
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained_model = None
        self.model_type = None
        self.analysis_results = {}
        self.visualizations = {}
        self.feature_cols = []  # Store feature column names for SHAP
        self.shap_values = None  # Store SHAP values
        self.shap_explainer = None  # Store SHAP explainer
        self.features_df = None  # Store features for later analysis
        
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
            self.features_df = features_df  # Store for later analysis
            
            # Step 3: Train risk prediction model
            model_results = self._train_risk_model(features_df, risk_threshold)
            
            # Step 4: Identify high-risk regions
            high_risk_regions = self._identify_high_risk_regions(features_df, risk_threshold)
            
            # Step 5: Cluster analysis for demographic groups
            cluster_analysis = self._cluster_demographics(features_df)
            
            # Step 6: SHAP Explainability Analysis (NEW)
            shap_analysis = self._compute_shap_explanations(features_df)
            
            # Step 7: Survival Curve Analysis (NEW)
            survival_data = self._generate_survival_data(features_df)
            
            # Step 8: Age Bucket Analysis (NEW)
            age_bucket_analysis = self._analyze_age_buckets(features_df)
            
            # Step 9: Centre Performance Analysis (NEW)
            centre_performance = self._analyze_centre_performance(features_df)
            
            # Step 10: Generate visualizations (enhanced)
            visualizations = self._generate_visualizations(features_df, high_risk_regions)
            
            # Step 11: Generate recommendations
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
                    "model_accuracy": model_results.get('accuracy', 0),
                    "shap_available": SHAP_AVAILABLE and self.shap_values is not None
                },
                "model_metrics": model_results,
                "high_risk_regions": high_risk_regions[:20],  # Top 20
                "cluster_analysis": cluster_analysis,
                "shap_analysis": shap_analysis,
                "survival_data": survival_data,
                "age_bucket_analysis": age_bucket_analysis,
                "centre_performance": centre_performance,
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
        """
        Calculate enhanced composite risk score based on multiple factors.
        
        Indicators (weighted by importance):
        - Time since last update (simulated) - 0.25
        - Low biometric update ratio - 0.20
        - Elderly population ratio - 0.20  
        - High-risk state flag - 0.10
        - Low demographic update ratio - 0.10
        - Centre quality score (proxy) - 0.10
        - Seasonal capture factor - 0.05
        """
        
        risk_score = pd.Series(0.3, index=df.index)  # Baseline
        
        # Factor 1: Time since last update (simulated from date field)
        # Higher time = higher risk (template aging)
        if 'date' in df.columns or 'last_update' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'last_update'
            try:
                dates = pd.to_datetime(df[date_col], errors='coerce')
                days_since = (pd.Timestamp.now() - dates).dt.days.fillna(365)
                # Normalize: 0-180 days = low risk, >365 days = high risk
                time_factor = np.clip(days_since / 730, 0, 1)  # 2 years max
                risk_score += time_factor * 0.25
            except:
                pass
        
        # Factor 2: Low biometric update ratio (high weight)
        if 'biometric_update_ratio' in df.columns:
            bio_ratio = df['biometric_update_ratio'].fillna(0.5)
            risk_score += (1 - bio_ratio.clip(0, 1)) * 0.20
        
        # Factor 3: Age-based risk (elderly population)
        # Create age bucket scores
        if 'elderly_ratio' in df.columns:
            elderly = df['elderly_ratio'].fillna(0).clip(0, 1)
            # Elderly have 3x fingerprint degradation rate
            risk_score += elderly * 0.20
        
        if 'age_65_plus_ratio' in df.columns:
            very_elderly = df['age_65_plus_ratio'].fillna(0).clip(0, 1)
            risk_score += very_elderly * 0.05  # Additional weight for 65+
        
        # Factor 4: High-risk state flag (known failure hotspots)
        if 'is_high_risk_state' in df.columns:
            risk_score += df['is_high_risk_state'].fillna(0) * 0.10
        
        # Factor 5: Low demographic update ratio
        if 'demographic_update_ratio' in df.columns:
            demo_ratio = df['demographic_update_ratio'].fillna(0.5)
            risk_score += (1 - demo_ratio.clip(0, 1)) * 0.10
        
        # Factor 6: Centre quality score (proxy via update frequency)
        if 'centre_quality_score' in df.columns:
            quality = df['centre_quality_score'].fillna(0.5).clip(0, 1)
            risk_score += (1 - quality) * 0.10
        elif 'update_frequency' in df.columns:
            freq = df['update_frequency'].fillna(df['update_frequency'].median())
            freq_norm = (freq - freq.min()) / (freq.max() - freq.min() + 1e-6)
            risk_score += (1 - freq_norm) * 0.10
        
        # Factor 7: Seasonal capture factor (monsoon = higher humidity issues)
        if 'capture_month' in df.columns:
            # Monsoon months (Jun-Sep) have ~15% lower quality
            monsoon = df['capture_month'].isin([6, 7, 8, 9]).astype(float)
            risk_score += monsoon * 0.05
        
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
    
    def _compute_shap_explanations(self, df: pd.DataFrame, top_k: int = 10) -> Dict[str, Any]:
        """
        Compute SHAP values for model explainability.
        
        Returns:
            - Global feature importance via mean absolute SHAP
            - Local explanations (top-3 factors per region)
        """
        if not SHAP_AVAILABLE or self.trained_model is None:
            return {
                "available": False,
                "message": "SHAP not available (install with: pip install shap)" if not SHAP_AVAILABLE else "Model not trained",
                "global_importance": {},
                "local_explanations": []
            }
        
        try:
            # Prepare features for SHAP
            exclude_cols = ['state', 'district', 'risk_category', 'risk_score', 'proxy_risk_score', 'cluster']
            feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
            
            if not feature_cols:
                return {"available": False, "message": "No numeric features for SHAP"}
            
            X = df[feature_cols].fillna(0).values
            X_scaled = self.scaler.transform(X)
            
            # Create SHAP explainer
            if self.model_type == 'xgboost':
                explainer = shap.TreeExplainer(self.trained_model)
                shap_values = explainer.shap_values(X_scaled)
            else:
                explainer = shap.TreeExplainer(self.trained_model)
                shap_values = explainer.shap_values(X_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
            
            self.shap_explainer = explainer
            self.shap_values = shap_values
            self.feature_cols = feature_cols
            
            # Global feature importance (mean absolute SHAP)
            global_importance = {}
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            for name, val in zip(feature_cols, mean_abs_shap):
                global_importance[name] = float(val)
            
            # Sort by importance
            global_importance = dict(sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:top_k])
            
            # Local explanations for top-k high risk regions
            local_explanations = []
            if 'risk_score' in df.columns:
                high_risk_indices = df.nlargest(min(top_k, len(df)), 'risk_score').index
                
                for idx in high_risk_indices:
                    row_idx = df.index.get_loc(idx)
                    row_shap = shap_values[row_idx]
                    
                    # Top 3 contributing factors
                    top3_indices = np.argsort(np.abs(row_shap))[-3:][::-1]
                    top3_factors = []
                    for i in top3_indices:
                        top3_factors.append({
                            "feature": feature_cols[i],
                            "shap_value": float(row_shap[i]),
                            "direction": "increases risk" if row_shap[i] > 0 else "decreases risk"
                        })
                    
                    local_explanations.append({
                        "region": df.loc[idx, 'state'] if 'state' in df.columns else f"Region_{idx}",
                        "risk_score": float(df.loc[idx, 'risk_score']) if 'risk_score' in df.columns else 0,
                        "top3_factors": top3_factors
                    })
            
            return {
                "available": True,
                "message": "SHAP analysis complete",
                "global_importance": global_importance,
                "local_explanations": local_explanations
            }
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return {
                "available": False,
                "message": f"SHAP analysis failed: {str(e)}",
                "global_importance": {},
                "local_explanations": []
            }
    
    def _generate_survival_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate Kaplan-Meier style survival curve data.
        
        Simulates time-to-first-failure based on risk factors.
        Shows probability of biometric working at time T.
        """
        try:
            # Time points (days since last capture)
            time_points = list(range(0, 1825, 30))  # 0 to 5 years, monthly
            
            # Base survival rate (exponential decay model)
            # Higher risk = faster decay
            base_lambda = 0.001  # Base failure rate
            
            # Calculate weighted average risk
            avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0.5
            
            # Adjust decay rate based on risk
            adjusted_lambda = base_lambda * (1 + avg_risk * 2)
            
            # Generate survival curve (S(t) = exp(-λt))
            survival_probs = [np.exp(-adjusted_lambda * t) for t in time_points]
            
            # Also generate risk-stratified curves
            curves = {
                "overall": survival_probs,
                "time_points_days": time_points,
                "time_points_years": [t/365 for t in time_points]
            }
            
            # If we have risk categories, generate per-category curves
            if 'risk_score' in df.columns:
                low_risk = df[df['risk_score'] < 0.4]
                high_risk = df[df['risk_score'] >= 0.6]
                
                if len(low_risk) > 0:
                    low_lambda = base_lambda * 0.5
                    curves["low_risk"] = [np.exp(-low_lambda * t) for t in time_points]
                
                if len(high_risk) > 0:
                    high_lambda = base_lambda * 3
                    curves["high_risk"] = [np.exp(-high_lambda * t) for t in time_points]
            
            # Key statistics
            median_survival = int(-np.log(0.5) / adjusted_lambda) if adjusted_lambda > 0 else 1825
            one_year_survival = np.exp(-adjusted_lambda * 365)
            three_year_survival = np.exp(-adjusted_lambda * 1095)
            
            # Calculate 5-year survival rate
            five_year_survival = np.exp(-adjusted_lambda * 1825)
            
            return {
                "curves": curves,
                "statistics": {
                    "median_survival_days": median_survival,
                    "one_year_survival_rate": float(one_year_survival),
                    "three_year_survival_rate": float(three_year_survival),
                    "five_year_survival_rate": float(five_year_survival),
                    "recommended_update_interval_days": min(median_survival, 730)  # Cap at 2 years
                },
                # Top-level fields for frontend compatibility
                "median_survival_years": round(median_survival / 365, 1),
                "five_year_survival_rate": float(five_year_survival),
                "methodology": "Exponential decay model adjusted for regional risk factors"
            }
            
        except Exception as e:
            logger.error(f"Survival analysis failed: {e}")
            return {"curves": {}, "statistics": {}, "error": str(e)}
    
    def _analyze_age_buckets(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze risk distribution across 5 age buckets.
        
        Buckets: 0-17, 18-34, 35-49, 50-64, 65+
        """
        try:
            age_buckets = ['0-17', '18-34', '35-49', '50-64', '65+']
            bucket_data = []
            
            # Get ratios for each bucket
            ratio_cols = {
                '0-17': 'ratio_0_17',
                '18-34': 'ratio_18_34',
                '35-49': 'ratio_35_49',
                '50-64': 'ratio_50_64',
                '65+': 'ratio_65_plus'
            }
            
            for bucket in age_buckets:
                col = ratio_cols[bucket]
                if col in df.columns:
                    avg_ratio = float(df[col].mean())
                    
                    # Estimate risk for this bucket
                    risk_multiplier = {
                        '0-17': 0.5,
                        '18-34': 0.7,
                        '35-49': 1.0,
                        '50-64': 1.5,
                        '65+': 2.5
                    }[bucket]
                    
                    base_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0.5
                    bucket_risk = min(1.0, base_risk * risk_multiplier / 1.5)  # Normalized
                    
                    bucket_data.append({
                        "bucket": bucket,
                        "population_ratio": avg_ratio,
                        "estimated_risk": float(bucket_risk),
                        "risk_multiplier": risk_multiplier,
                        "recommended_action": self._get_age_recommendation(bucket, bucket_risk)
                    })
                else:
                    bucket_data.append({
                        "bucket": bucket,
                        "population_ratio": 0,
                        "estimated_risk": 0,
                        "risk_multiplier": 1.0,
                        "recommended_action": "Insufficient data"
                    })
            
            # Summary statistics
            highest_risk_bucket = max(bucket_data, key=lambda x: x['estimated_risk'])
            
            return {
                "buckets": bucket_data,
                "highest_risk_bucket": highest_risk_bucket['bucket'],
                "highest_risk_value": highest_risk_bucket['estimated_risk'],
                "elderly_risk_ratio": bucket_data[-1]['estimated_risk'] / max(bucket_data[1]['estimated_risk'], 0.01) if bucket_data else 0
            }
            
        except Exception as e:
            logger.error(f"Age bucket analysis failed: {e}")
            return {"buckets": [], "error": str(e)}
    
    def _get_age_recommendation(self, bucket: str, risk: float) -> str:
        """Get recommendation based on age bucket and risk level"""
        if bucket == '65+':
            if risk >= 0.7:
                return "Immediate outreach - home visits recommended"
            return "Priority scheduling - iris-only option available"
        elif bucket == '50-64':
            if risk >= 0.6:
                return "Proactive scheduling recommended"
            return "Include in regular update campaigns"
        elif bucket == '0-17':
            return "Mandatory update at age 15 - schedule accordingly"
        else:
            if risk >= 0.5:
                return "Monitor and include in awareness campaigns"
            return "Standard processing - low priority"
    
    def _analyze_centre_performance(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze capture centre performance metrics.
        
        Per-centre/district metrics:
        - Failure proxy rate
        - Average quality score
        - High-risk enrollee count
        """
        try:
            centre_data = []
            
            # Group by state (as proxy for centre clusters)
            if 'state' not in df.columns:
                return []
            
            for state in df['state'].unique():
                state_data = df[df['state'] == state]
                
                # Quality metrics
                quality_score = float(state_data['centre_quality_score'].mean()) if 'centre_quality_score' in state_data.columns else 0.5
                bio_ratio = float(state_data['biometric_update_ratio'].mean()) if 'biometric_update_ratio' in state_data.columns else 0.5
                
                # Risk metrics
                avg_risk = float(state_data['risk_score'].mean()) if 'risk_score' in state_data.columns else 0.5
                high_risk_count = int((state_data['risk_score'] >= 0.6).sum()) if 'risk_score' in state_data.columns else 0
                
                # Estimated failure rate (proxy)
                failure_rate_proxy = max(0, 1 - quality_score) * avg_risk
                
                centre_data.append({
                    "centre_id": state,  # Using state as centre cluster ID
                    "captures": int(state_data['total_population'].sum()) if 'total_population' in state_data.columns else len(state_data),
                    "avg_quality_score": quality_score,
                    "biometric_update_ratio": bio_ratio,
                    "avg_risk_score": avg_risk,
                    "high_risk_count": high_risk_count,
                    "failure_rate_proxy": float(failure_rate_proxy),
                    "performance_grade": self._get_performance_grade(quality_score, failure_rate_proxy)
                })
            
            # Sort by failure rate (worst first)
            centre_data.sort(key=lambda x: x['failure_rate_proxy'], reverse=True)
            
            return centre_data[:20]  # Top 20 centres
            
        except Exception as e:
            logger.error(f"Centre performance analysis failed: {e}")
            return []
    
    def _get_performance_grade(self, quality: float, failure_rate: float) -> str:
        """Assign performance grade based on quality and failure rate"""
        score = quality * 0.6 + (1 - failure_rate) * 0.4
        if score >= 0.8:
            return "A - Excellent"
        elif score >= 0.6:
            return "B - Good"
        elif score >= 0.4:
            return "C - Needs Improvement"
        else:
            return "D - Critical Attention Required"
    
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
            
            # 6. NEW: Survival Curve (Kaplan-Meier style)
            visualizations['survival_curve'] = self._plot_survival_curve(df)
            
            # 7. NEW: Age Bucket Risk Bar Chart
            visualizations['age_bucket_chart'] = self._plot_age_bucket_chart(df)
            
            # 8. NEW: SHAP Feature Importance (if available)
            if SHAP_AVAILABLE and self.shap_values is not None:
                visualizations['shap_importance'] = self._plot_shap_importance()
            
            # 9. NEW: Centre Performance Chart
            visualizations['centre_performance'] = self._plot_centre_performance(df)
            
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
    
    def _plot_survival_curve(self, df: pd.DataFrame) -> str:
        """Plot Kaplan-Meier style survival curve"""
        plt.figure(figsize=(10, 6))
        
        try:
            # Time points (months)
            time_points = np.linspace(0, 60, 61)  # 0 to 5 years (60 months)
            
            # Base failure rate
            base_lambda = 0.001 * 30  # Monthly rate
            
            # Calculate overall survival curve
            avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0.5
            adjusted_lambda = base_lambda * (1 + avg_risk * 2)
            overall_survival = np.exp(-adjusted_lambda * time_points)
            
            # Plot overall curve
            plt.plot(time_points, overall_survival, 'b-', linewidth=2, label='Overall Population')
            
            # Risk-stratified curves
            if 'risk_score' in df.columns:
                # Low risk curve
                low_lambda = base_lambda * 0.5
                low_survival = np.exp(-low_lambda * time_points)
                plt.plot(time_points, low_survival, 'g--', linewidth=2, label='Low Risk (score < 0.4)')
                
                # High risk curve
                high_lambda = base_lambda * 3
                high_survival = np.exp(-high_lambda * time_points)
                plt.plot(time_points, high_survival, 'r--', linewidth=2, label='High Risk (score ≥ 0.6)')
            
            plt.xlabel('Months Since Last Capture', fontsize=12)
            plt.ylabel('Probability of Biometric Working', fontsize=12)
            plt.title('Template Survival Curve (Time-to-Failure)', fontsize=14, fontweight='bold')
            plt.legend(loc='lower left')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.05)
            plt.xlim(0, 60)
            
            # Add annotation for recommended update
            median_months = int(-np.log(0.5) / adjusted_lambda) if adjusted_lambda > 0 else 24
            plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            plt.axvline(x=median_months, color='orange', linestyle=':', alpha=0.5)
            plt.annotate(f'50% survival at {median_months} months', 
                        xy=(median_months, 0.5), xytext=(median_months+5, 0.55),
                        arrowprops=dict(arrowstyle='->', color='orange'),
                        fontsize=10, color='orange')
        except Exception as e:
            plt.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
        
        plt.tight_layout()
        return self._fig_to_base64(plt.gcf())
    
    def _plot_age_bucket_chart(self, df: pd.DataFrame) -> str:
        """Plot 5-tier age bucket risk bar chart"""
        plt.figure(figsize=(10, 6))
        
        try:
            buckets = ['0-17', '18-34', '35-49', '50-64', '65+']
            risk_multipliers = [0.5, 0.7, 1.0, 1.5, 2.5]
            
            # Calculate actual risks for each bucket
            base_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0.5
            bucket_risks = [min(1.0, base_risk * mult / 1.5) for mult in risk_multipliers]
            
            # Colors based on risk level
            colors = ['#00cc66' if r < 0.4 else '#ffaa00' if r < 0.6 else '#ff6644' if r < 0.8 else '#ff4444' 
                     for r in bucket_risks]
            
            bars = plt.bar(buckets, bucket_risks, color=colors, edgecolor='white', linewidth=2)
            
            # Add value labels
            for bar, val in zip(bars, bucket_risks):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.xlabel('Age Bucket', fontsize=12)
            plt.ylabel('Estimated Risk Score', fontsize=12)
            plt.title('Biometric Failure Risk by Age Group', fontsize=14, fontweight='bold')
            plt.ylim(0, 1.15)
            
            # Add risk level zones
            plt.axhline(y=0.4, color='green', linestyle='--', alpha=0.3, label='Low-Moderate')
            plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.3, label='Moderate-High')
            plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='High-Critical')
            plt.legend(loc='upper left', fontsize=9)
            
        except Exception as e:
            plt.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
        
        plt.tight_layout()
        return self._fig_to_base64(plt.gcf())
    
    def _plot_shap_importance(self) -> str:
        """Plot SHAP feature importance bar chart"""
        plt.figure(figsize=(10, 6))
        
        try:
            if self.shap_values is not None and self.feature_cols:
                # Calculate mean absolute SHAP values
                mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
                
                # Get top 10 features
                top_k = min(10, len(self.feature_cols))
                sorted_idx = np.argsort(mean_abs_shap)[-top_k:]
                
                features = [self.feature_cols[i] for i in sorted_idx]
                values = mean_abs_shap[sorted_idx]
                
                # Clean up feature names for display
                display_names = []
                for f in features:
                    name = f.replace('_', ' ').replace('ratio', '').strip()
                    if len(name) > 25:
                        name = name[:22] + '...'
                    display_names.append(name.title())
                
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))
                
                bars = plt.barh(display_names, values, color=colors)
                
                plt.xlabel('Mean |SHAP value| (impact on model output)', fontsize=11)
                plt.title('SHAP Feature Importance: What Drives Risk Predictions?', 
                         fontsize=13, fontweight='bold')
                
                # Add value labels
                for bar, val in zip(bars, values):
                    plt.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                            f'{val:.3f}', va='center', fontsize=9)
            else:
                plt.text(0.5, 0.5, 'SHAP values not computed\n(run analysis first)', 
                        ha='center', va='center', fontsize=14)
                
        except Exception as e:
            plt.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
        
        plt.tight_layout()
        return self._fig_to_base64(plt.gcf())
    
    def _plot_centre_performance(self, df: pd.DataFrame) -> str:
        """Plot centre performance chart"""
        plt.figure(figsize=(12, 6))
        
        try:
            if 'state' in df.columns and 'centre_quality_score' in df.columns:
                # Get top 15 states by lowest quality
                plot_data = df.groupby('state').agg({
                    'centre_quality_score': 'mean',
                    'risk_score': 'mean' if 'risk_score' in df.columns else 'count'
                }).reset_index()
                
                plot_data = plot_data.nsmallest(15, 'centre_quality_score')
                
                # Create color based on performance
                colors = ['#ff4444' if q < 0.4 else '#ffaa00' if q < 0.6 else '#00cc66' 
                         for q in plot_data['centre_quality_score']]
                
                bars = plt.barh(plot_data['state'], plot_data['centre_quality_score'], color=colors)
                
                # Add grade labels
                for bar, score in zip(bars, plot_data['centre_quality_score']):
                    grade = 'D' if score < 0.4 else 'C' if score < 0.6 else 'B' if score < 0.8 else 'A'
                    plt.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                            f'{score:.2f} ({grade})', va='center', fontsize=10)
                
                plt.xlabel('Centre Quality Score', fontsize=12)
                plt.ylabel('State/Centre', fontsize=12)
                plt.title('Capture Centre Performance (Lowest 15)', fontsize=14, fontweight='bold')
                plt.xlim(0, 1.1)
                
                # Add grade zone lines
                plt.axvline(x=0.4, color='red', linestyle='--', alpha=0.3)
                plt.axvline(x=0.6, color='orange', linestyle='--', alpha=0.3)
                plt.axvline(x=0.8, color='green', linestyle='--', alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'Insufficient centre data', ha='center', va='center', fontsize=14)
                
        except Exception as e:
            plt.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
        
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
