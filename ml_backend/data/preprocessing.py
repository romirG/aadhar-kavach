"""
Data preprocessing module - Clean and prepare data for ML models.
"""
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data cleaning, encoding, and scaling."""
    
    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.feature_names: List[str] = []
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fit preprocessors and transform data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (processed numpy array, feature names)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for preprocessing")
            return np.array([]), []
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Identify column types
        self._identify_columns(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Encode categorical columns
        df = self._encode_categorical(df)
        
        # Convert numeric columns (handle strings that should be numbers)
        df = self._convert_numeric(df)
        
        # Select only numeric columns for scaling
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("No numeric columns after preprocessing")
            return np.array([]), []
        
        # Scale numeric features
        scaled_data = self._scale_features(numeric_df)
        
        self.feature_names = list(numeric_df.columns)
        
        logger.info(f"Preprocessed {len(df)} records with {len(self.feature_names)} features")
        
        return scaled_data, self.feature_names
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessors."""
        if df.empty:
            return np.array([])
        
        df = df.copy()
        df = self._handle_missing_values(df)
        df = self._encode_categorical(df, fit=False)
        df = self._convert_numeric(df)
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Ensure same columns as training
        for col in self.feature_names:
            if col not in numeric_df.columns:
                numeric_df[col] = 0
        
        numeric_df = numeric_df[self.feature_names]
        
        return self._scale_features(numeric_df, fit=False)
    
    def _identify_columns(self, df: pd.DataFrame):
        """Identify numeric and categorical columns."""
        self.numeric_columns = []
        self.categorical_columns = []
        
        for col in df.columns:
            # Skip ID-like columns
            if 'id' in col.lower() or col.lower() in ['date', 'timestamp']:
                continue
                
            if df[col].dtype in ['int64', 'float64']:
                self.numeric_columns.append(col)
            elif df[col].dtype == 'object':
                # Check if it's actually numeric stored as string
                try:
                    pd.to_numeric(df[col], errors='raise')
                    self.numeric_columns.append(col)
                except (ValueError, TypeError):
                    self.categorical_columns.append(col)
        
        logger.info(f"Identified {len(self.numeric_columns)} numeric, {len(self.categorical_columns)} categorical columns")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        for col in df.columns:
            if df[col].isnull().any():
                if col in self.numeric_columns or df[col].dtype in ['int64', 'float64']:
                    # Use median for numeric columns
                    if col not in self.imputers:
                        self.imputers[col] = SimpleImputer(strategy='median')
                        df[col] = self.imputers[col].fit_transform(df[[col]]).ravel()
                    else:
                        df[col] = self.imputers[col].transform(df[[col]]).ravel()
                else:
                    # Use mode for categorical
                    df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical columns using LabelEncoder."""
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
                
            if fit:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    # Handle unknown categories
                    df[col] = df[col].astype(str)
                    df[f"{col}_encoded"] = self.encoders[col].fit_transform(df[col])
            else:
                if col in self.encoders:
                    df[col] = df[col].astype(str)
                    # Handle unseen categories
                    known_classes = set(self.encoders[col].classes_)
                    df[col] = df[col].apply(lambda x: x if x in known_classes else 'Unknown')
                    df[f"{col}_encoded"] = self.encoders[col].transform(df[col])
        
        return df
    
    def _convert_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert string columns that should be numeric."""
        for col in self.numeric_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale numeric features using StandardScaler."""
        if fit:
            self.scalers['main'] = StandardScaler()
            return self.scalers['main'].fit_transform(df)
        else:
            return self.scalers['main'].transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names after preprocessing."""
        return self.feature_names


# Singleton instance
_preprocessor_instance: Optional[DataPreprocessor] = None


def get_preprocessor() -> DataPreprocessor:
    """Get or create DataPreprocessor singleton."""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = DataPreprocessor()
    return _preprocessor_instance
