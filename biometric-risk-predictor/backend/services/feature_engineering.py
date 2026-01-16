"""
Feature Engineering for Biometric Risk Prediction

PRIVACY NOTICE:
- All features derived from aggregated, anonymized data
- No individual-level features or PII
- Operations performed on state/district/age-group aggregates only

Enhanced with ChatGPT specifications:
- 5-tier age buckets (0-17, 18-34, 35-49, 50-64, 65+)
- Time since update calculations
- Centre quality score proxy
- Seasonal capture factors
- Manual labor proxy (rural districts)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder

from config import HIGH_RISK_STATES, AGE_BUCKETS, RISK_THRESHOLDS

# Enhanced 5-tier age buckets for detailed analysis
ENHANCED_AGE_BUCKETS = {
    '0-17': (0, 17),      # Children/minors
    '18-34': (18, 34),    # Young adults
    '35-49': (35, 49),    # Middle-aged
    '50-64': (50, 64),    # Pre-elderly
    '65+': (65, 120)      # Elderly (highest risk)
}

# Age bucket risk multipliers (based on biometric degradation rates)
AGE_BUCKET_RISK_MULTIPLIERS = {
    '0-17': 0.5,    # Low - growing/stable biometrics
    '18-34': 0.7,   # Low-moderate
    '35-49': 1.0,   # Baseline
    '50-64': 1.5,   # Elevated - beginning degradation
    '65+': 2.5      # High - significant fingerprint wear
}

# Monsoon months with higher humidity affecting capture quality
MONSOON_MONTHS = [6, 7, 8, 9]  # June-September

# Rural districts proxy for manual labor (higher biometric wear)
RURAL_HEAVY_LABOR_STATES = [
    'Bihar', 'Jharkhand', 'Chhattisgarh', 'Odisha', 'Madhya Pradesh',
    'Uttar Pradesh', 'Rajasthan', 'West Bengal'
]


class FeatureEngineer:
    """Feature engineering for biometric re-enrollment risk prediction"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []
        
    def process_enrolment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process enrollment data and extract features"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
        
        # Convert age columns to numeric
        for col in ['age_0_5', 'age_5_17', 'age_18_greater']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Total enrollments per record
        age_cols = [c for c in ['age_0_5', 'age_5_17', 'age_18_greater'] if c in df.columns]
        if age_cols:
            df['total_enrollments'] = df[age_cols].sum(axis=1)
            
            # Age distribution percentages
            for col in age_cols:
                df[f'{col}_pct'] = (df[col] / df['total_enrollments'].replace(0, 1)) * 100
        
        return df
    
    def process_demographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographic update data"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
        
        # Convert age columns to numeric
        for col in ['demo_age_5_17', 'demo_age_17_']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Total demographic updates
        demo_cols = [c for c in ['demo_age_5_17', 'demo_age_17_'] if c in df.columns]
        if demo_cols:
            df['total_demo_updates'] = df[demo_cols].sum(axis=1)
        
        return df
    
    def process_biometric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process biometric update data"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
        
        # Convert age columns to numeric
        for col in ['bio_age_5_17', 'bio_age_17_']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Total biometric updates
        bio_cols = [c for c in ['bio_age_5_17', 'bio_age_17_'] if c in df.columns]
        if bio_cols:
            df['total_bio_updates'] = df[bio_cols].sum(axis=1)
        
        return df
    
    def aggregate_by_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data at state level"""
        if df.empty or 'state' not in df.columns:
            return df
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        agg_dict = {col: 'sum' for col in numeric_cols if col not in ['month', 'year']}
        
        if agg_dict:
            return df.groupby('state').agg(agg_dict).reset_index()
        return df
    
    def aggregate_by_district(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data at district level"""
        if df.empty or 'district' not in df.columns:
            return df
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        agg_dict = {col: 'sum' for col in numeric_cols if col not in ['month', 'year']}
        
        group_cols = ['state', 'district'] if 'state' in df.columns else ['district']
        
        if agg_dict:
            return df.groupby(group_cols).agg(agg_dict).reset_index()
        return df
    
    def create_risk_features(
        self,
        enrolment_df: Optional[pd.DataFrame] = None,
        demographic_df: Optional[pd.DataFrame] = None,
        biometric_df: Optional[pd.DataFrame] = None,
        aggregation_level: str = 'state'
    ) -> pd.DataFrame:
        """
        Create comprehensive risk features from multiple datasets
        
        Args:
            enrolment_df: Processed enrollment data
            demographic_df: Processed demographic data
            biometric_df: Processed biometric data
            aggregation_level: 'state' or 'district'
            
        Returns:
            DataFrame with engineered risk features
        """
        dfs_to_merge = []
        
        # Process each dataset
        if enrolment_df is not None and not enrolment_df.empty:
            enrol = self.process_enrolment_data(enrolment_df)
            if aggregation_level == 'state':
                enrol = self.aggregate_by_state(enrol)
            else:
                enrol = self.aggregate_by_district(enrol)
            enrol = enrol.add_prefix('enrol_')
            if aggregation_level == 'state':
                enrol = enrol.rename(columns={'enrol_state': 'state'})
            else:
                enrol = enrol.rename(columns={'enrol_state': 'state', 'enrol_district': 'district'})
            dfs_to_merge.append(enrol)
        
        if demographic_df is not None and not demographic_df.empty:
            demo = self.process_demographic_data(demographic_df)
            if aggregation_level == 'state':
                demo = self.aggregate_by_state(demo)
            else:
                demo = self.aggregate_by_district(demo)
            demo = demo.add_prefix('demo_')
            if aggregation_level == 'state':
                demo = demo.rename(columns={'demo_state': 'state'})
            else:
                demo = demo.rename(columns={'demo_state': 'state', 'demo_district': 'district'})
            dfs_to_merge.append(demo)
        
        if biometric_df is not None and not biometric_df.empty:
            bio = self.process_biometric_data(biometric_df)
            if aggregation_level == 'state':
                bio = self.aggregate_by_state(bio)
            else:
                bio = self.aggregate_by_district(bio)
            bio = bio.add_prefix('bio_')
            if aggregation_level == 'state':
                bio = bio.rename(columns={'bio_state': 'state'})
            else:
                bio = bio.rename(columns={'bio_state': 'state', 'bio_district': 'district'})
            dfs_to_merge.append(bio)
        
        if not dfs_to_merge:
            return pd.DataFrame()
        
        # Merge all dataframes
        merge_cols = ['state', 'district'] if aggregation_level == 'district' else ['state']
        result = dfs_to_merge[0]
        for df in dfs_to_merge[1:]:
            result = result.merge(df, on=merge_cols, how='outer')
        
        # Fill missing values
        result = result.fillna(0)
        
        # Create derived features
        result = self._create_derived_features(result, aggregation_level)
        
        return result
    
    def _create_derived_features(self, df: pd.DataFrame, aggregation_level: str) -> pd.DataFrame:
        """
        Create enhanced derived features for risk prediction.
        
        Features added:
        - time_since_update_days: Days since last biometric update
        - age_bucket_encoded: 5-tier age bucket (0-17, 18-34, 35-49, 50-64, 65+)
        - centre_quality_score: Proxy quality score per centre/district
        - seasonal_capture_factor: Monsoon month degradation factor
        - manual_labor_proxy: Rural heavy-labor state flag
        - elderly_ratio: Proportion of elderly population
        - adult_ratio: Proportion of adult population
        """
        df = df.copy()
        
        # ====== 1. Time Since Update (Primary aging signal) ======
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if date_cols:
            date_col = date_cols[0]
            try:
                dates = pd.to_datetime(df[date_col], format='%d-%m-%Y', errors='coerce')
                df['time_since_update_days'] = (pd.Timestamp.now() - dates).dt.days.fillna(365)
                # Normalize to years for readability
                df['time_since_update_years'] = df['time_since_update_days'] / 365.25
            except Exception:
                df['time_since_update_days'] = 365  # Default 1 year
                df['time_since_update_years'] = 1.0
        else:
            # Simulate time based on row index if no date available
            np.random.seed(42)
            df['time_since_update_days'] = np.random.randint(30, 730, len(df))
            df['time_since_update_years'] = df['time_since_update_days'] / 365.25
        
        # ====== 2. Enhanced 5-Tier Age Buckets ======
        # Calculate age distribution ratios from available columns
        child_cols = [c for c in df.columns if 'age_0_5' in c or 'age_5_17' in c]
        adult_cols = [c for c in df.columns if 'age_18' in c or 'age_17_' in c]
        
        # Calculate totals for each age group
        df['child_count'] = df[child_cols].sum(axis=1) if child_cols else 0
        df['adult_count'] = df[adult_cols].sum(axis=1) if adult_cols else 0
        
        # Total population proxy
        total_cols = [c for c in df.columns if 'total' in c.lower() and ('enrol' in c.lower() or 'bio' in c.lower())]
        if total_cols:
            df['total_population'] = df[total_cols].sum(axis=1).replace(0, 1)
        else:
            df['total_population'] = (df['child_count'] + df['adult_count']).replace(0, 1)
        
        # Estimate age bucket distributions (using Indian demographic patterns)
        df['ratio_0_17'] = df['child_count'] / df['total_population']
        df['ratio_18_34'] = df['adult_count'] * 0.35 / df['total_population']  # ~35% of adults
        df['ratio_35_49'] = df['adult_count'] * 0.30 / df['total_population']  # ~30% of adults
        df['ratio_50_64'] = df['adult_count'] * 0.20 / df['total_population']  # ~20% of adults
        df['ratio_65_plus'] = df['adult_count'] * 0.15 / df['total_population']  # ~15% of adults (elderly)
        
        # Create age bucket risk score (weighted by risk multipliers)
        df['age_bucket_risk_score'] = (
            df['ratio_0_17'] * AGE_BUCKET_RISK_MULTIPLIERS['0-17'] +
            df['ratio_18_34'] * AGE_BUCKET_RISK_MULTIPLIERS['18-34'] +
            df['ratio_35_49'] * AGE_BUCKET_RISK_MULTIPLIERS['35-49'] +
            df['ratio_50_64'] * AGE_BUCKET_RISK_MULTIPLIERS['50-64'] +
            df['ratio_65_plus'] * AGE_BUCKET_RISK_MULTIPLIERS['65+']
        )
        
        # Elderly and adult ratios for compatibility
        df['elderly_ratio'] = df['ratio_50_64'] + df['ratio_65_plus']
        df['adult_ratio'] = 1 - df['ratio_0_17']
        df['age_65_plus_ratio'] = df['ratio_65_plus']
        
        # ====== 3. Biometric Update Ratio ======
        bio_total_col = [c for c in df.columns if 'bio_total' in c]
        demo_total_col = [c for c in df.columns if 'demo_total' in c]
        
        if bio_total_col and demo_total_col:
            bio_col = bio_total_col[0]
            demo_col = demo_total_col[0]
            total = df[bio_col] + df[demo_col]
            df['biometric_update_ratio'] = df[bio_col] / total.replace(0, 1)
            df['demographic_update_ratio'] = df[demo_col] / total.replace(0, 1)
        else:
            df['biometric_update_ratio'] = 0.5
            df['demographic_update_ratio'] = 0.5
        
        # ====== 4. Centre Quality Score (Proxy) ======
        # Higher update frequency = better centre quality
        if 'biometric_update_ratio' in df.columns:
            # Combine multiple signals for quality score
            bio_signal = df['biometric_update_ratio'].clip(0, 1)
            
            # Adjust for time - older updates = lower quality
            time_penalty = 1 - (df['time_since_update_days'].clip(0, 730) / 730) * 0.3
            
            df['centre_quality_score'] = (bio_signal * 0.7 + time_penalty * 0.3).clip(0, 1)
        else:
            df['centre_quality_score'] = 0.5
        
        # ====== 5. Seasonal Capture Factor ======
        # Extract month from date if available
        if date_cols:
            try:
                months = pd.to_datetime(df[date_cols[0]], format='%d-%m-%Y', errors='coerce').dt.month
                df['capture_month'] = months.fillna(1).astype(int)
                # Monsoon months have ~15% lower quality
                df['is_monsoon_capture'] = df['capture_month'].isin(MONSOON_MONTHS).astype(int)
                df['seasonal_capture_factor'] = 1.0 - (df['is_monsoon_capture'] * 0.15)
            except Exception:
                df['capture_month'] = 1
                df['is_monsoon_capture'] = 0
                df['seasonal_capture_factor'] = 1.0
        else:
            df['capture_month'] = 1
            df['is_monsoon_capture'] = 0
            df['seasonal_capture_factor'] = 1.0
        
        # ====== 6. High Risk State & Manual Labor Proxy ======
        if 'state' in df.columns:
            df['is_high_risk_state'] = df['state'].isin(HIGH_RISK_STATES).astype(int)
            df['is_rural_heavy_labor'] = df['state'].isin(RURAL_HEAVY_LABOR_STATES).astype(int)
        else:
            df['is_high_risk_state'] = 0
            df['is_rural_heavy_labor'] = 0
        
        # ====== 7. Encode Categorical Features ======
        # Encode state
        if 'state' in df.columns:
            if 'state' not in self.label_encoders:
                self.label_encoders['state'] = LabelEncoder()
                df['encoded_state'] = self.label_encoders['state'].fit_transform(df['state'].astype(str))
            else:
                known_states = set(self.label_encoders['state'].classes_)
                df['state_temp'] = df['state'].apply(lambda x: x if x in known_states else 'Unknown')
                if 'Unknown' not in known_states:
                    self.label_encoders['state'].classes_ = np.append(self.label_encoders['state'].classes_, 'Unknown')
                df['encoded_state'] = self.label_encoders['state'].transform(df['state_temp'].astype(str))
                df = df.drop('state_temp', axis=1)
        
        # Encode district
        if 'district' in df.columns:
            if 'district' not in self.label_encoders:
                self.label_encoders['district'] = LabelEncoder()
                df['encoded_district'] = self.label_encoders['district'].fit_transform(df['district'].astype(str))
            else:
                known_districts = set(self.label_encoders['district'].classes_)
                df['district_temp'] = df['district'].apply(lambda x: x if x in known_districts else 'Unknown')
                if 'Unknown' not in known_districts:
                    self.label_encoders['district'].classes_ = np.append(self.label_encoders['district'].classes_, 'Unknown')
                df['encoded_district'] = self.label_encoders['district'].transform(df['district_temp'].astype(str))
                df = df.drop('district_temp', axis=1)
        
        # Store feature names
        self.feature_names = [c for c in df.columns if c not in ['state', 'district']]
        
        return df
    
    def create_proxy_risk_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create proxy risk labels when ground truth is unavailable
        
        Risk factors:
        - High risk state
        - Low biometric update ratio
        - High elderly population percentage
        """
        df = df.copy()
        
        risk_score = np.zeros(len(df))
        
        # Factor 1: High risk state (weight: 0.3)
        if 'is_high_risk_state' in df.columns:
            risk_score += df['is_high_risk_state'] * 0.3
        
        # Factor 2: Low biometric update ratio (weight: 0.4)
        if 'biometric_update_ratio' in df.columns:
            # Lower ratio = higher risk
            risk_score += (1 - df['biometric_update_ratio']) * 0.4
        
        # Factor 3: High elderly percentage (weight: 0.3)
        elderly_cols = [c for c in df.columns if 'age_18_greater' in c or 'age_17_' in c]
        total_cols = [c for c in df.columns if 'total' in c and 'enrol' in c]
        
        if elderly_cols and total_cols:
            elderly_pct = df[elderly_cols].sum(axis=1) / df[total_cols].sum(axis=1).replace(0, 1)
            risk_score += elderly_pct * 0.3
        
        df['proxy_risk_score'] = risk_score
        df['risk_category'] = pd.cut(
            df['proxy_risk_score'],
            bins=[0, RISK_THRESHOLDS['low'], RISK_THRESHOLDS['medium'], RISK_THRESHOLDS['high'], 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of engineered feature names"""
        return self.feature_names


# Singleton instance
feature_engineer = FeatureEngineer()
