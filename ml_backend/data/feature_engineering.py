"""
Feature engineering module - Create derived features for anomaly detection.
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create derived features for anomaly detection."""
    
    def __init__(self):
        self.temporal_features: List[str] = []
        self.geo_features: List[str] = []
        self.behavioral_features: List[str] = []
        self.all_engineered_features: List[str] = []
    
    def engineer_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Create all engineered features based on dataset type.
        
        Args:
            df: Input DataFrame
            dataset_type: One of 'enrolment', 'demographic', 'biometric'
            
        Returns:
            DataFrame with additional engineered features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for feature engineering")
            return df
        
        df = df.copy()
        
        # Common transformations
        df = self._create_temporal_features(df)
        df = self._create_geo_features(df)
        
        # Dataset-specific features
        if dataset_type == 'enrolment':
            df = self._create_enrolment_features(df)
        elif dataset_type == 'demographic':
            df = self._create_demographic_features(df)
        elif dataset_type == 'biometric':
            df = self._create_biometric_features(df)
        
        # Create aggregate statistics
        df = self._create_aggregate_features(df)
        
        # Create anomaly indicator features
        df = self._create_anomaly_indicators(df)
        
        logger.info(f"Engineered {len(self.all_engineered_features)} new features")
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if 'date' in df.columns:
            try:
                df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Extract time components
                df['month'] = df['date_parsed'].dt.month
                df['year'] = df['date_parsed'].dt.year
                df['quarter'] = df['date_parsed'].dt.quarter
                df['day_of_week'] = df['date_parsed'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                df['is_month_end'] = df['date_parsed'].dt.is_month_end.astype(int)
                df['is_month_start'] = df['date_parsed'].dt.is_month_start.astype(int)
                
                # Days since reference
                min_date = df['date_parsed'].min()
                if pd.notna(min_date):
                    df['days_since_start'] = (df['date_parsed'] - min_date).dt.days
                
                self.temporal_features = ['month', 'year', 'quarter', 'day_of_week', 
                                          'is_weekend', 'is_month_end', 'is_month_start',
                                          'days_since_start']
                
                logger.info("Created temporal features")
                
            except Exception as e:
                logger.warning(f"Error creating temporal features: {e}")
        
        return df
    
    def _create_geo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geographic features."""
        # State density (events per state)
        if 'state' in df.columns:
            state_counts = df['state'].value_counts()
            df['state_event_count'] = df['state'].map(state_counts)
            df['state_event_pct'] = df['state_event_count'] / len(df) * 100
            self.geo_features.extend(['state_event_count', 'state_event_pct'])
        
        # District density
        if 'district' in df.columns:
            district_counts = df['district'].value_counts()
            df['district_event_count'] = df['district'].map(district_counts)
            df['district_event_pct'] = df['district_event_count'] / len(df) * 100
            self.geo_features.extend(['district_event_count', 'district_event_pct'])
        
        # Pincode analysis
        if 'pincode' in df.columns:
            pincode_counts = df['pincode'].value_counts()
            df['pincode_event_count'] = df['pincode'].map(pincode_counts)
            
            # Extract pincode region (first 2 digits)
            df['pincode_str'] = df['pincode'].astype(str)
            df['pincode_region'] = df['pincode_str'].str[:2]
            self.geo_features.extend(['pincode_event_count', 'pincode_region'])
        
        logger.info(f"Created {len(self.geo_features)} geo features")
        
        return df
    
    def _create_enrolment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to enrolment data."""
        age_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
        
        # Convert age columns to numeric
        for col in age_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Total enrolments
        existing_age_cols = [col for col in age_cols if col in df.columns]
        if existing_age_cols:
            df['total_enrolments'] = df[existing_age_cols].sum(axis=1)
            
            # Age distribution ratios
            if df['total_enrolments'].sum() > 0:
                for col in existing_age_cols:
                    ratio_col = f"{col}_ratio"
                    df[ratio_col] = df[col] / (df['total_enrolments'] + 1)  # +1 to avoid division by zero
                    self.behavioral_features.append(ratio_col)
            
            # Identify unusual age distributions
            df['child_heavy'] = (df.get('age_0_5', 0) + df.get('age_5_17', 0)) > df.get('age_18_greater', 0)
            
            self.behavioral_features.extend(['total_enrolments', 'child_heavy'])
        
        return df
    
    def _create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to demographic update data."""
        demo_cols = ['demo_age_5_17', 'demo_age_17_']
        
        for col in demo_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        existing_demo_cols = [col for col in demo_cols if col in df.columns]
        if existing_demo_cols:
            df['total_demo_updates'] = df[existing_demo_cols].sum(axis=1)
            
            # Update intensity
            if 'total_demo_updates' in df.columns:
                mean_updates = df['total_demo_updates'].mean()
                std_updates = df['total_demo_updates'].std()
                if std_updates > 0:
                    df['demo_update_zscore'] = (df['total_demo_updates'] - mean_updates) / std_updates
                else:
                    df['demo_update_zscore'] = 0
                    
                self.behavioral_features.extend(['total_demo_updates', 'demo_update_zscore'])
        
        return df
    
    def _create_biometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to biometric update data."""
        bio_cols = ['bio_age_5_17', 'bio_age_17_']
        
        for col in bio_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        existing_bio_cols = [col for col in bio_cols if col in df.columns]
        if existing_bio_cols:
            df['total_bio_updates'] = df[existing_bio_cols].sum(axis=1)
            
            # Biometric update rate comparison
            if 'total_bio_updates' in df.columns:
                mean_updates = df['total_bio_updates'].mean()
                std_updates = df['total_bio_updates'].std()
                if std_updates > 0:
                    df['bio_update_zscore'] = (df['total_bio_updates'] - mean_updates) / std_updates
                else:
                    df['bio_update_zscore'] = 0
                    
                self.behavioral_features.extend(['total_bio_updates', 'bio_update_zscore'])
        
        return df
    
    def _create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate statistical features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove already processed columns
        exclude_cols = ['month', 'year', 'quarter', 'day_of_week']
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        if len(numeric_cols) >= 2:
            # Row-wise statistics
            df['row_mean'] = df[numeric_cols].mean(axis=1)
            df['row_std'] = df[numeric_cols].std(axis=1)
            df['row_max'] = df[numeric_cols].max(axis=1)
            df['row_min'] = df[numeric_cols].min(axis=1)
            df['row_range'] = df['row_max'] - df['row_min']
            
            self.behavioral_features.extend(['row_mean', 'row_std', 'row_max', 'row_min', 'row_range'])
        
        return df
    
    def _create_anomaly_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that may indicate anomalies."""
        # High value indicators
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            if col in df.columns:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[f'{col}_is_extreme'] = ((df[col] > q99) | (df[col] < q01)).astype(int)
        
        # Collect all engineered features
        self.all_engineered_features = (
            self.temporal_features + 
            self.geo_features + 
            self.behavioral_features
        )
        
        return df
    
    def get_feature_profile(self) -> dict:
        """Get summary of engineered features."""
        return {
            "temporal": self.temporal_features,
            "geographic": self.geo_features,
            "behavioral": self.behavioral_features,
            "total": len(self.all_engineered_features)
        }


# Singleton instance
_engineer_instance: Optional[FeatureEngineer] = None


def get_feature_engineer() -> FeatureEngineer:
    """Get or create FeatureEngineer singleton."""
    global _engineer_instance
    if _engineer_instance is None:
        _engineer_instance = FeatureEngineer()
    return _engineer_instance
