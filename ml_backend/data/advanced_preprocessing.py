"""
Advanced Preprocessing and Feature Engineering Pipeline for UIDAI Fraud Detection.

This module provides a comprehensive, production-grade pipeline for:
1. Loading and cleaning data from API responses
2. Handling missing, noisy, and inconsistent fields
3. Encoding categorical features
4. Creating fraud-indicative behavioral and temporal features
5. Normalizing and scaling for ML

Each feature is designed to detect specific fraud patterns in Aadhaar enrolment/update data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AdvancedPreprocessor:
    """
    Robust preprocessing pipeline for Aadhaar enrolment/update data.
    
    Handles:
    - Missing value imputation with smart strategies
    - Outlier detection and handling
    - Categorical encoding with unknown handling
    - Feature scaling with multiple strategies
    """
    
    def __init__(self, scaling_method: str = 'robust'):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: 'standard', 'minmax', or 'robust' (recommended for fraud data)
        """
        self.scaling_method = scaling_method
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = None
        self.feature_stats: Dict[str, Dict] = {}
        self.is_fitted = False
        
    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Smart missing value imputation.
        
        Strategy:
        - Numeric: Median (robust to outliers in fraud data)
        - Categorical: Mode or 'UNKNOWN'
        - Dates: Forward fill then backward fill
        """
        df = df.copy()
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue
                
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                # Use median for numeric (robust to fraudulent outliers)
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.debug(f"Imputed {missing_count} missing values in '{col}' with median={median_val}")
                
            elif 'date' in col.lower() or 'time' in col.lower():
                # Forward-fill dates, then backward-fill remaining
                df[col] = df[col].ffill().bfill()
                
            else:
                # Categorical: use mode or 'UNKNOWN'
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna('UNKNOWN', inplace=True)
                    
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
        """
        Handle outliers in numeric columns.
        
        Args:
            method: 'clip' (winsorize to 1-99 percentile) or 'remove'
            
        Why: Fraudulent spikes often appear as extreme outliers.
        We clip rather than remove to preserve potential fraud signals.
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].std() == 0:
                continue
                
            q01 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            
            if method == 'clip':
                df[col] = df[col].clip(lower=q01, upper=q99)
            elif method == 'remove':
                df = df[(df[col] >= q01) & (df[col] <= q99)]
                
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features with unknown value handling.
        
        Categories encoded:
        - gender: Male/Female/Other → 0/1/2
        - event_type: enrolment/demographic_update/biometric_update
        - state, district: Geographic identifiers
        - operator_id, center_id: For behavioral analysis
        """
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            # Skip columns that should remain as strings
            if any(skip in col.lower() for skip in ['id', 'name', 'description', 'reason']):
                continue
                
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Add 'UNKNOWN' to handle unseen values at prediction time
                    unique_vals = list(df[col].astype(str).unique()) + ['UNKNOWN']
                    self.label_encoders[col].fit(unique_vals)
                    
            if col in self.label_encoders:
                # Handle unseen values by mapping to 'UNKNOWN'
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in self.label_encoders[col].classes_ else 'UNKNOWN'
                )
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                
        return df
    
    def _create_scaler(self):
        """Create appropriate scaler based on configuration."""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler()
        else:  # 'robust' - recommended for fraud data
            return RobustScaler()
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fit preprocessor and transform data.
        
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        logger.info(f"Preprocessing {len(df)} records...")
        
        # Step 1: Impute missing values
        df = self._impute_missing(df)
        
        # Step 2: Handle outliers
        df = self._handle_outliers(df, method='clip')
        
        # Step 3: Encode categoricals
        df = self._encode_categoricals(df, fit=True)
        
        # Step 4: Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_names = [c for c in numeric_cols if not c.endswith('_id')]
        
        if len(feature_names) == 0:
            raise ValueError("No numeric features found after preprocessing")
        
        X = df[feature_names].values.astype(np.float32)
        
        # Step 5: Handle any remaining NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Step 6: Scale features
        self.scaler = self._create_scaler()
        X = self.scaler.fit_transform(X)
        
        self.is_fitted = True
        logger.info(f"Preprocessed into {X.shape[1]} features")
        
        return X, feature_names
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        df = self._impute_missing(df)
        df = self._handle_outliers(df, method='clip')
        df = self._encode_categoricals(df, fit=False)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_names = [c for c in numeric_cols if not c.endswith('_id')]
        
        X = df[feature_names].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)
        
        return X


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for fraud detection in Aadhaar data.
    
    Creates features that capture:
    1. Temporal patterns (time-based anomalies)
    2. Behavioral patterns (operator/center activity)
    3. Geographic patterns (location-based anomalies)
    4. Statistical patterns (deviation from norms)
    
    Each feature is designed to detect specific fraud patterns.
    """
    
    # Feature explanation documentation
    FEATURE_EXPLANATIONS = {
        # Temporal Features
        'hour_of_day': 'Fraudulent activities often occur at unusual hours (late night/early morning)',
        'day_of_week': 'Weekend/holiday patterns may indicate fraudulent batch operations',
        'is_weekend': 'Legitimate enrollments are rare on weekends in many centers',
        'is_business_hours': 'Activities outside business hours (9-18) are suspicious',
        'month': 'Seasonal patterns; fraudsters may operate in bursts',
        'quarter': 'Quarterly patterns aligned with reporting cycles',
        'is_month_end': 'Month-end surges may indicate quota manipulation',
        'is_month_start': 'Month-start patterns differ from regular days',
        'days_since_epoch': 'Temporal trend detection across the dataset',
        
        # Behavioral Features - Per Aadhaar ID
        'updates_per_aadhaar': 'High update frequency for same ID suggests data manipulation',
        'days_between_updates': 'Very frequent updates (same day) are highly suspicious',
        'update_velocity': 'Acceleration in update frequency indicates fraud',
        
        # Behavioral Features - Per Operator
        'events_per_operator': 'Operators with abnormally high activity may be fraudulent',
        'operator_daily_avg': 'Daily average helps normalize for tenure',
        'operator_zscore': 'Statistical deviation from typical operator behavior',
        'operator_diversity': 'Operators handling too many diverse locations are suspicious',
        
        # Behavioral Features - Per Center
        'events_per_center': 'Centers with activity spikes may be compromised',
        'center_daily_avg': 'Daily center throughput benchmark',
        'center_zscore': 'Deviation from typical center behavior',
        'center_operator_ratio': 'Too few operators for high volume indicates fraud',
        
        # Geographic Features
        'state_event_count': 'State-level activity for relative comparison',
        'state_event_pct': 'Percentage of events from each state',
        'district_event_count': 'District-level granularity',
        'district_event_pct': 'District contribution to state activity',
        'pincode_event_count': 'Pincode-level (hyperlocal) patterns',
        'geo_concentration': 'High concentration in few locations is suspicious',
        'location_hop_count': 'Frequent location changes per ID suggest fraud',
        
        # Age Distribution Features
        'age_0_5_ratio': 'Infant enrollment ratio (unusual spikes indicate fraud)',
        'age_5_17_ratio': 'Child/teen ratio (mandatory school enrollment patterns)',
        'age_18_plus_ratio': 'Adult ratio (should be highest in most areas)',
        'age_distribution_entropy': 'Uniform distribution across ages is suspicious',
        'age_delta': 'Age changes in updates should be consistent with time passed',
        
        # Update Type Features
        'demo_update_ratio': 'Ratio of demographic updates',
        'bio_update_ratio': 'Ratio of biometric updates (expensive, should be rare)',
        'update_type_entropy': 'Even distribution across types is suspicious',
        
        # Statistical Features
        'row_sum': 'Total activity measure',
        'row_mean': 'Average activity level',
        'row_std': 'Activity variability (low std is suspicious)',
        'row_max': 'Peak activity indicator',
        'row_min': 'Minimum activity',
        'row_range': 'Activity range',
        'row_skew': 'Asymmetry in activity distribution',
        'row_kurtosis': 'Tail behavior of activity distribution',
        
        # Velocity & Acceleration
        'activity_velocity': 'Rate of change in activity over time',
        'activity_acceleration': 'Change in velocity (sudden spikes)',
        
        # Cross-Feature Interactions
        'bio_demo_ratio': 'Ratio of biometric to demographic updates (should be low)',
        'weekend_activity_ratio': 'Weekend to weekday activity ratio',
        'night_activity_ratio': 'Night to day activity ratio'
    }
    
    def __init__(self):
        self.feature_stats: Dict[str, Dict] = {}
        self.is_fitted = False
        
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and validate date columns."""
        df = df.copy()
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
                
        # Create synthetic date if none exists
        if not date_cols:
            df['parsed_date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
            
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features for detecting temporal fraud patterns.
        
        Fraud patterns detected:
        - Off-hours activity (night/weekend enrollments)
        - Burst patterns (sudden spikes in activity)
        - Seasonal anomalies
        """
        df = df.copy()
        
        # Find the primary date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() and pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
                
        if date_col is None:
            df['parsed_date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
            date_col = 'parsed_date'
        
        dt = df[date_col]
        
        # Basic temporal features
        df['hour_of_day'] = dt.dt.hour
        df['day_of_week'] = dt.dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = dt.dt.day
        df['month'] = dt.dt.month
        df['quarter'] = dt.dt.quarter
        df['year'] = dt.dt.year
        
        # Binary temporal indicators
        df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        df['is_business_hours'] = ((dt.dt.hour >= 9) & (dt.dt.hour <= 18)).astype(int)
        df['is_night'] = ((dt.dt.hour >= 22) | (dt.dt.hour <= 5)).astype(int)
        df['is_month_end'] = (dt.dt.day >= 25).astype(int)
        df['is_month_start'] = (dt.dt.day <= 5).astype(int)
        
        # Days since epoch (for trend detection)
        df['days_since_epoch'] = (dt - pd.Timestamp('2020-01-01')).dt.days
        
        logger.debug(f"Created {12} temporal features")
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral features based on entity activity patterns.
        
        Entities analyzed:
        - Aadhaar IDs (update frequency, velocity)
        - Operators (activity volume, diversity)
        - Enrollment Centers (throughput, patterns)
        
        Fraud patterns detected:
        - Abnormally high activity per entity
        - Unusual diversity of operations
        - Activity velocity spikes
        """
        df = df.copy()
        
        # Operator-level features (if available)
        if 'operator' in df.columns or 'operator_id' in df.columns:
            op_col = 'operator' if 'operator' in df.columns else 'operator_id'
            
            # Events per operator
            op_counts = df.groupby(op_col).size()
            df['events_per_operator'] = df[op_col].map(op_counts)
            
            # Operator z-score (deviation from mean)
            op_mean = op_counts.mean()
            op_std = op_counts.std() if op_counts.std() > 0 else 1
            df['operator_zscore'] = (df['events_per_operator'] - op_mean) / op_std
            
            # Operator diversity (unique locations)
            if 'district' in df.columns:
                op_diversity = df.groupby(op_col)['district'].nunique()
                df['operator_diversity'] = df[op_col].map(op_diversity)
                
        # Center-level features (if available)
        if 'center' in df.columns or 'centre' in df.columns or 'center_id' in df.columns:
            center_col = next((c for c in ['center', 'centre', 'center_id'] if c in df.columns), None)
            if center_col:
                center_counts = df.groupby(center_col).size()
                df['events_per_center'] = df[center_col].map(center_counts)
                
                center_mean = center_counts.mean()
                center_std = center_counts.std() if center_counts.std() > 0 else 1
                df['center_zscore'] = (df['events_per_center'] - center_mean) / center_std
                
        # Aadhaar ID level features (if available, often anonymized)
        if 'aadhaar_id' in df.columns or 'uid' in df.columns:
            id_col = 'aadhaar_id' if 'aadhaar_id' in df.columns else 'uid'
            
            id_counts = df.groupby(id_col).size()
            df['updates_per_aadhaar'] = df[id_col].map(id_counts)
            
            # Flag high-update IDs
            df['is_high_update_id'] = (df['updates_per_aadhaar'] > id_counts.quantile(0.95)).astype(int)
            
        logger.debug("Created behavioral features")
        return df
    
    def _create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create geographic features for location-based fraud detection.
        
        Fraud patterns detected:
        - Geographic concentration (activity in few locations)
        - Location hopping (frequent location changes)
        - Unusual state/district patterns
        """
        df = df.copy()
        
        # State-level features
        if 'state' in df.columns:
            state_counts = df['state'].value_counts()
            df['state_event_count'] = df['state'].map(state_counts)
            df['state_event_pct'] = df['state_event_count'] / len(df) * 100
            
            # State rank (larger states should have more activity)
            state_rank = state_counts.rank(ascending=False)
            df['state_rank'] = df['state'].map(state_rank)
            
        # District-level features
        if 'district' in df.columns:
            district_counts = df['district'].value_counts()
            df['district_event_count'] = df['district'].map(district_counts)
            df['district_event_pct'] = df['district_event_count'] / len(df) * 100
            
            # District z-score
            dist_mean = district_counts.mean()
            dist_std = district_counts.std() if district_counts.std() > 0 else 1
            df['district_zscore'] = (df['district_event_count'] - dist_mean) / dist_std
            
        # Pincode-level features
        if 'pincode' in df.columns:
            pincode_counts = df['pincode'].value_counts()
            df['pincode_event_count'] = df['pincode'].map(pincode_counts)
            
            # Very localized activity is suspicious
            df['pincode_zscore'] = (df['pincode_event_count'] - pincode_counts.mean()) / (pincode_counts.std() or 1)
            
        # Geographic concentration index
        if 'state' in df.columns and 'district' in df.columns:
            geo_counts = df.groupby(['state', 'district']).size()
            hhi = (geo_counts / geo_counts.sum()).pow(2).sum()  # Herfindahl-Hirschman Index
            df['geo_concentration'] = hhi
            
        logger.debug("Created geographic features")
        return df
    
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-based features for demographic fraud detection.
        
        Fraud patterns detected:
        - Unusual age distributions (too uniform or skewed)
        - Age inconsistencies in updates
        - Suspicious infant/child enrollment patterns
        """
        df = df.copy()
        
        # Age group columns (from data.gov.in API)
        age_cols = [c for c in df.columns if 'age' in c.lower()]
        
        if len(age_cols) >= 2:
            # Convert to numeric
            for col in age_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Calculate total
            df['total_enrollments'] = df[age_cols].sum(axis=1)
            
            # Age group ratios
            if 'age_0_5' in df.columns or any('0_5' in c or '0-5' in c for c in age_cols):
                age_0_5_col = next((c for c in age_cols if '0_5' in c or '0-5' in c), age_cols[0])
                df['age_0_5_ratio'] = df[age_0_5_col] / (df['total_enrollments'] + 1)
                
            if any('5_17' in c or '5-17' in c for c in age_cols):
                age_5_17_col = next((c for c in age_cols if '5_17' in c or '5-17' in c), None)
                if age_5_17_col:
                    df['age_5_17_ratio'] = df[age_5_17_col] / (df['total_enrollments'] + 1)
                    
            if any('18' in c for c in age_cols):
                age_18_col = next((c for c in age_cols if '18' in c), None)
                if age_18_col:
                    df['age_18_plus_ratio'] = df[age_18_col] / (df['total_enrollments'] + 1)
            
            # Age distribution entropy (uniform = suspicious)
            age_values = df[age_cols].values
            age_probs = age_values / (age_values.sum(axis=1, keepdims=True) + 1e-10)
            df['age_entropy'] = -np.sum(age_probs * np.log(age_probs + 1e-10), axis=1)
            
        logger.debug("Created age-related features")
        return df
    
    def _create_update_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on update types (demographic vs biometric).
        
        Fraud patterns detected:
        - High biometric update ratios (expensive, should be rare)
        - Unusual update type distributions
        """
        df = df.copy()
        
        # Check for demographic update columns
        demo_cols = [c for c in df.columns if 'demo' in c.lower()]
        bio_cols = [c for c in df.columns if 'bio' in c.lower()]
        
        if demo_cols:
            for col in demo_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['total_demo_updates'] = df[demo_cols].sum(axis=1)
            
        if bio_cols:
            for col in bio_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['total_bio_updates'] = df[bio_cols].sum(axis=1)
            
        # Calculate ratios
        if 'total_demo_updates' in df.columns and 'total_bio_updates' in df.columns:
            total_updates = df['total_demo_updates'] + df['total_bio_updates'] + 1
            df['demo_update_ratio'] = df['total_demo_updates'] / total_updates
            df['bio_update_ratio'] = df['total_bio_updates'] / total_updates
            
            # Bio/Demo ratio (bio should be low)
            df['bio_demo_ratio'] = df['total_bio_updates'] / (df['total_demo_updates'] + 1)
            
            # Z-scores for update volumes
            demo_mean = df['total_demo_updates'].mean()
            demo_std = df['total_demo_updates'].std() or 1
            df['demo_update_zscore'] = (df['total_demo_updates'] - demo_mean) / demo_std
            
            bio_mean = df['total_bio_updates'].mean()
            bio_std = df['total_bio_updates'].std() or 1
            df['bio_update_zscore'] = (df['total_bio_updates'] - bio_mean) / bio_std
            
        logger.debug("Created update type features")
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features across numeric columns.
        
        Fraud patterns detected:
        - Unusual statistical distributions
        - Low variance (manufactured data)
        - Extreme values
        """
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude already-created features and encoded columns
        numeric_cols = [c for c in numeric_cols if not any(x in c for x in 
                       ['_zscore', '_pct', '_ratio', '_encoded', 'is_', 'days_since'])]
        
        if len(numeric_cols) >= 2:
            values = df[numeric_cols].values
            
            df['row_sum'] = np.sum(values, axis=1)
            df['row_mean'] = np.mean(values, axis=1)
            df['row_std'] = np.std(values, axis=1)
            df['row_max'] = np.max(values, axis=1)
            df['row_min'] = np.min(values, axis=1)
            df['row_range'] = df['row_max'] - df['row_min']
            
            # Higher-order statistics
            with np.errstate(all='ignore'):
                df['row_skew'] = stats.skew(values, axis=1, nan_policy='omit')
                df['row_kurtosis'] = stats.kurtosis(values, axis=1, nan_policy='omit')
                
            # Replace infinities and NaN
            df['row_skew'] = df['row_skew'].replace([np.inf, -np.inf], 0).fillna(0)
            df['row_kurtosis'] = df['row_kurtosis'].replace([np.inf, -np.inf], 0).fillna(0)
            
        logger.debug("Created statistical features")
        return df
    
    def engineer_features(self, df: pd.DataFrame, dataset_type: str = 'enrolment') -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: Raw DataFrame from API
            dataset_type: 'enrolment', 'demographic', or 'biometric'
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info(f"Engineering features for {len(df)} records (type: {dataset_type})")
        
        # Parse dates first
        df = self._parse_dates(df)
        
        # Apply all feature engineering steps
        df = self._create_temporal_features(df)
        df = self._create_behavioral_features(df)
        df = self._create_geographic_features(df)
        df = self._create_age_features(df)
        df = self._create_update_type_features(df)
        df = self._create_statistical_features(df)
        
        logger.info(f"Created {len(df.columns)} total columns")
        return df
    
    def get_feature_explanations(self) -> Dict[str, str]:
        """Return explanations for all features."""
        return self.FEATURE_EXPLANATIONS


class MLReadyPipeline:
    """
    Complete pipeline that combines preprocessing and feature engineering.
    
    Produces an ML-ready feature matrix with:
    - All features scaled and normalized
    - No missing values
    - Feature names and explanations
    """
    
    def __init__(self, scaling_method: str = 'robust'):
        self.preprocessor = AdvancedPreprocessor(scaling_method=scaling_method)
        self.feature_engineer = AdvancedFeatureEngineer()
        self.final_feature_names: List[str] = []
        
    def fit_transform(self, df: pd.DataFrame, dataset_type: str = 'enrolment') -> Tuple[np.ndarray, List[str]]:
        """
        Run complete pipeline and return ML-ready matrix.
        
        Args:
            df: Raw data from API
            dataset_type: Type of dataset
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Step 1: Feature Engineering
        df_engineered = self.feature_engineer.engineer_features(df, dataset_type)
        
        # Step 2: Preprocessing and Scaling
        X, feature_names = self.preprocessor.fit_transform(df_engineered)
        
        self.final_feature_names = feature_names
        
        return X, feature_names
    
    def get_feature_schema(self) -> Dict[str, Any]:
        """
        Get the final ML feature matrix schema.
        
        Returns:
            Dict with feature names, types, and explanations
        """
        explanations = self.feature_engineer.get_feature_explanations()
        
        schema = {
            'total_features': len(self.final_feature_names),
            'features': []
        }
        
        for name in self.final_feature_names:
            feature_info = {
                'name': name,
                'type': 'float32',
                'scaled': True,
                'explanation': explanations.get(name, 'Derived numeric feature')
            }
            schema['features'].append(feature_info)
            
        return schema
    
    def transform(self, df: pd.DataFrame, dataset_type: str = 'enrolment') -> np.ndarray:
        """Transform new data using fitted pipeline."""
        df_engineered = self.feature_engineer.engineer_features(df, dataset_type)
        return self.preprocessor.transform(df_engineered)


# =============================================================================
# FEATURE MATRIX SCHEMA DOCUMENTATION
# =============================================================================

FEATURE_MATRIX_SCHEMA = """
# ML-Ready Feature Matrix Schema for UIDAI Fraud Detection

## Overview
The pipeline produces a 2D numpy array of shape (n_samples, n_features) where:
- All features are float32
- All features are scaled (default: RobustScaler for outlier resistance)
- No missing values (imputed during preprocessing)

## Feature Categories

### 1. Temporal Features (12 features)
| Feature | Type | Range | Fraud Signal |
|---------|------|-------|--------------|
| hour_of_day | int | 0-23 | Off-hours activity |
| day_of_week | int | 0-6 | Weekend patterns |
| is_weekend | binary | 0-1 | Weekend fraud |
| is_business_hours | binary | 0-1 | After-hours activity |
| is_night | binary | 0-1 | Nighttime fraud |
| month | int | 1-12 | Seasonal patterns |
| quarter | int | 1-4 | Quarterly cycles |
| is_month_end | binary | 0-1 | Month-end spikes |
| is_month_start | binary | 0-1 | Month-start patterns |
| days_since_epoch | int | 0+ | Temporal trend |

### 2. Behavioral Features (8+ features)
| Feature | Type | Fraud Signal |
|---------|------|--------------|
| events_per_operator | int | High = suspicious |
| operator_zscore | float | >3 = outlier |
| operator_diversity | int | Too high = suspicious |
| events_per_center | int | Unusual volumes |
| center_zscore | float | >3 = outlier |
| updates_per_aadhaar | int | >3 = suspicious |
| is_high_update_id | binary | Flagged IDs |

### 3. Geographic Features (10+ features)
| Feature | Type | Fraud Signal |
|---------|------|--------------|
| state_event_count | int | Relative volume |
| state_event_pct | float | State share |
| district_event_count | int | Local volume |
| district_zscore | float | District anomaly |
| pincode_event_count | int | Hyperlocal |
| geo_concentration | float | High = suspicious |

### 4. Age Features (5+ features)
| Feature | Type | Fraud Signal |
|---------|------|--------------|
| age_0_5_ratio | float | Unusual infant rates |
| age_5_17_ratio | float | Child patterns |
| age_18_plus_ratio | float | Adult patterns |
| age_entropy | float | Low = suspicious |
| total_enrollments | int | Volume check |

### 5. Update Type Features (6+ features)
| Feature | Type | Fraud Signal |
|---------|------|--------------|
| total_demo_updates | int | Demographic volume |
| total_bio_updates | int | Biometric volume |
| demo_update_ratio | float | Demo proportion |
| bio_update_ratio | float | Bio proportion (high = suspicious) |
| bio_demo_ratio | float | High = unusual |
| demo_update_zscore | float | Volume anomaly |

### 6. Statistical Features (8 features)
| Feature | Type | Fraud Signal |
|---------|------|--------------|
| row_sum | float | Total activity |
| row_mean | float | Average level |
| row_std | float | Low = manufactured |
| row_max | float | Peak values |
| row_min | float | Baseline |
| row_range | float | Activity span |
| row_skew | float | Distribution shape |
| row_kurtosis | float | Tail behavior |

## Usage Example

```python
from advanced_preprocessing import MLReadyPipeline

# Initialize pipeline
pipeline = MLReadyPipeline(scaling_method='robust')

# Load data from API
import httpx
response = httpx.get('https://api.data.gov.in/resource/...')
df = pd.DataFrame(response.json()['records'])

# Get ML-ready features
X, feature_names = pipeline.fit_transform(df, dataset_type='enrolment')

# X is now ready for Isolation Forest, Autoencoder, HDBSCAN etc.
print(f"Feature matrix shape: {X.shape}")
print(f"Features: {feature_names}")

# Get schema documentation
schema = pipeline.get_feature_schema()
```
"""

if __name__ == "__main__":
    # Example usage
    print(FEATURE_MATRIX_SCHEMA)
