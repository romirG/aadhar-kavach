"""
<<<<<<< Updated upstream
UIDAI Data Fetcher Service

Implements dynamic data ingestion from:
1. data.gov.in portal (primary)
2. Mock data service (fallback)

Fetches:
- Total Enrolments by district
- Biometric Updates
- Demographic Updates
- Population data for normalization
=======
UIDAI Data Fetcher Service with Univariate Analysis

Implements:
1. Dynamic data ingestion from data.gov.in
2. Mock data fallback based on UIDAI saturation reports
3. Univariate analysis for missing value imputation
4. Normalized intensity calculation
>>>>>>> Stashed changes
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

# API Configuration
DATA_GOV_IN_API_KEY = os.getenv('DATA_GOV_API_KEY', '')
DATA_GOV_BASE_URL = 'https://api.data.gov.in/resource'

<<<<<<< Updated upstream
# UIDAI Aadhaar Saturation Resource IDs (data.gov.in)
RESOURCE_IDS = {
    'aadhaar_saturation': '31f84c63-ad8f-44f0-ac48-07f1cb11b9c5',
    'enrolment_statistics': 'bfa18f97-0d6f-4e6d-8f0e-9e92ae0d7b42',
    'demographic_updates': 'a3d4e5f6-7890-1234-5678-9abcdef01234',
    'biometric_updates': 'b4c5d6e7-8901-2345-6789-0abcdef12345'
=======
RESOURCE_IDS = {
    'aadhaar_saturation': '31f84c63-ad8f-44f0-ac48-07f1cb11b9c5',
    'enrolment_statistics': 'bfa18f97-0d6f-4e6d-8f0e-9e92ae0d7b42',
>>>>>>> Stashed changes
}


@dataclass
class DistrictData:
<<<<<<< Updated upstream
    """District enrollment data structure"""
=======
>>>>>>> Stashed changes
    district_code: str
    district_name: str
    state_code: str
    state_name: str
    total_enrolments: int
    biometric_updates: int
    demographic_updates: int
    population: int
    coverage_percent: float
    latitude: float
    longitude: float
    last_updated: str


<<<<<<< Updated upstream
class UIDAIDataFetcher:
    """
    Fetches enrollment data from UIDAI sources
    
    Primary: data.gov.in API
    Fallback: Mock data based on official saturation reports
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize data fetcher
        
        Args:
            api_key: data.gov.in API key (optional)
        """
        self.api_key = api_key or DATA_GOV_IN_API_KEY
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=6)  # 6-hour cache
        
    def fetch_from_data_gov(
        self,
        resource_id: str,
        limit: int = 1000,
        offset: int = 0
    ) -> Optional[Dict]:
        """
        Fetch data from data.gov.in API
        
        Args:
            resource_id: Resource identifier
            limit: Number of records
            offset: Pagination offset
            
        Returns:
            JSON response or None if failed
        """
=======
class UnivariateAnalyzer:
    """
    Univariate analysis for missing value imputation
    and outlier detection in enrollment data
    """
    
    @staticmethod
    def detect_missing(df: pd.DataFrame) -> Dict[str, int]:
        """Detect missing values in each column"""
        return df.isnull().sum().to_dict()
    
    @staticmethod
    def impute_missing(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Impute missing values using univariate statistics
        
        Strategies:
        - 'median': Use median (robust to outliers)
        - 'mean': Use mean
        - 'mode': Use mode (for categorical)
        - 'state_median': Use state-level median
        """
        result = df.copy()
        
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if result[col].isnull().any():
                if strategy == 'median':
                    fill_value = result[col].median()
                elif strategy == 'mean':
                    fill_value = result[col].mean()
                elif strategy == 'state_median' and 'state' in result.columns:
                    # Group by state and fill with state median
                    result[col] = result.groupby('state')[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                    # Fill remaining with global median
                    fill_value = result[col].median()
                else:
                    fill_value = result[col].median()
                
                result[col] = result[col].fillna(fill_value)
        
        return result
    
    @staticmethod
    def detect_outliers(series: pd.Series, method: str = 'iqr') -> pd.Series:
        """
        Detect outliers using IQR or Z-score method
        
        Returns boolean Series (True = outlier)
        """
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return (series < lower) | (series > upper)
        else:  # z-score
            z = (series - series.mean()) / series.std()
            return abs(z) > 3
    
    @staticmethod
    def winsorize_outliers(series: pd.Series, limits: tuple = (0.05, 0.95)) -> pd.Series:
        """Cap outliers at percentile limits"""
        lower = series.quantile(limits[0])
        upper = series.quantile(limits[1])
        return series.clip(lower=lower, upper=upper)
    
    @staticmethod
    def compute_summary_stats(series: pd.Series) -> Dict:
        """Compute univariate summary statistics"""
        return {
            'count': int(series.count()),
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'q25': float(series.quantile(0.25)),
            'median': float(series.median()),
            'q75': float(series.quantile(0.75)),
            'max': float(series.max()),
            'skewness': float(series.skew()),
            'kurtosis': float(series.kurtosis()),
            'missing': int(series.isnull().sum())
        }


class UIDAIDataFetcher:
    """
    Fetches enrollment data from UIDAI sources with preprocessing
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DATA_GOV_IN_API_KEY
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=6)
        self.analyzer = UnivariateAnalyzer()
        
    def fetch_from_data_gov(self, resource_id: str, limit: int = 1000) -> Optional[Dict]:
        """Fetch data from data.gov.in API"""
>>>>>>> Stashed changes
        if not self.api_key:
            print("No API key configured, using mock data")
            return None
            
        try:
            url = f"{DATA_GOV_BASE_URL}/{resource_id}"
            params = {
                'api-key': self.api_key,
                'format': 'json',
<<<<<<< Updated upstream
                'limit': limit,
                'offset': offset
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
=======
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
>>>>>>> Stashed changes
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return None
    
    def get_district_enrolments(self, use_cache: bool = True) -> List[DistrictData]:
<<<<<<< Updated upstream
        """
        Get enrollment data for all districts
        
        Returns district-level data with enrolments, updates, and population.
        Uses cache to avoid hitting API too frequently.
        """
        cache_key = 'district_enrolments'
        
        # Check cache
=======
        """Get enrollment data with caching"""
        cache_key = 'district_enrolments'
        
>>>>>>> Stashed changes
        if use_cache and cache_key in self.cache:
            if datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
                return self.cache[cache_key]
        
<<<<<<< Updated upstream
        # Try fetching from API
        data = self.fetch_from_data_gov(RESOURCE_IDS['aadhaar_saturation'])
        
        if data and 'records' in data:
            # Parse API response
            districts = self._parse_api_response(data['records'])
        else:
            # Fall back to mock data
            districts = self._generate_mock_districts()
        
        # Update cache
=======
        data = self.fetch_from_data_gov(RESOURCE_IDS['aadhaar_saturation'])
        
        if data and 'records' in data:
            districts = self._parse_api_response(data['records'])
        else:
            districts = self._generate_mock_districts()
        
>>>>>>> Stashed changes
        self.cache[cache_key] = districts
        self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
        
        return districts
    
    def _parse_api_response(self, records: List[Dict]) -> List[DistrictData]:
<<<<<<< Updated upstream
        """Parse data.gov.in API response into DistrictData objects"""
        districts = []
        
=======
        """Parse API response"""
        districts = []
>>>>>>> Stashed changes
        for record in records:
            try:
                districts.append(DistrictData(
                    district_code=str(record.get('district_code', '')),
                    district_name=record.get('district_name', ''),
                    state_code=str(record.get('state_code', '')),
                    state_name=record.get('state_name', ''),
                    total_enrolments=int(record.get('total_enrolments', 0)),
                    biometric_updates=int(record.get('biometric_updates', 0)),
                    demographic_updates=int(record.get('demographic_updates', 0)),
                    population=int(record.get('population', 100000)),
                    coverage_percent=float(record.get('coverage_percent', 0)),
                    latitude=float(record.get('latitude', 0)),
                    longitude=float(record.get('longitude', 0)),
                    last_updated=record.get('last_updated', datetime.now().isoformat())
                ))
<<<<<<< Updated upstream
            except (ValueError, TypeError) as e:
                print(f"Error parsing record: {e}")
                continue
        
        return districts
    
    def _generate_mock_districts(self) -> List[DistrictData]:
        """
        Generate realistic mock district data based on UIDAI saturation reports
        
        Data based on actual UIDAI statistics:
        - ~140 Crore Aadhaar numbers generated
        - High saturation in urban areas
        - Lower coverage in NE states and remote areas
        """
        np.random.seed(42)
        
        # State-level data with realistic coordinates and coverage
=======
            except (ValueError, TypeError):
                continue
        return districts
    
    def _generate_mock_districts(self) -> List[DistrictData]:
        """Generate realistic mock data based on UIDAI stats"""
        np.random.seed(42)
        
>>>>>>> Stashed changes
        states_data = {
            'Maharashtra': {'pop': 12.5e7, 'cov': 98.2, 'count': 36, 'center': (73.8, 19.1)},
            'Uttar Pradesh': {'pop': 23.15e7, 'cov': 94.2, 'count': 75, 'center': (80.9, 26.8)},
            'Bihar': {'pop': 12.4e7, 'cov': 88.5, 'count': 38, 'center': (85.3, 25.6)},
            'West Bengal': {'pop': 9.9e7, 'cov': 92.1, 'count': 23, 'center': (87.8, 22.8)},
            'Madhya Pradesh': {'pop': 8.5e7, 'cov': 91.3, 'count': 52, 'center': (78.6, 23.5)},
            'Rajasthan': {'pop': 7.9e7, 'cov': 89.7, 'count': 33, 'center': (74.2, 27.0)},
            'Tamil Nadu': {'pop': 7.8e7, 'cov': 98.2, 'count': 38, 'center': (78.6, 11.1)},
            'Karnataka': {'pop': 6.7e7, 'cov': 96.5, 'count': 31, 'center': (75.7, 15.3)},
            'Gujarat': {'pop': 6.5e7, 'cov': 95.8, 'count': 33, 'center': (71.2, 22.3)},
            'Andhra Pradesh': {'pop': 5.3e7, 'cov': 97.1, 'count': 13, 'center': (79.7, 15.9)},
            'Odisha': {'pop': 4.6e7, 'cov': 90.4, 'count': 30, 'center': (85.1, 20.5)},
            'Telangana': {'pop': 3.8e7, 'cov': 96.8, 'count': 33, 'center': (79.0, 18.1)},
            'Kerala': {'pop': 3.5e7, 'cov': 99.1, 'count': 14, 'center': (76.3, 10.8)},
            'Jharkhand': {'pop': 3.7e7, 'cov': 87.2, 'count': 24, 'center': (85.3, 23.6)},
            'Assam': {'pop': 3.5e7, 'cov': 85.6, 'count': 35, 'center': (92.9, 26.2)},
            'Punjab': {'pop': 3.0e7, 'cov': 94.5, 'count': 23, 'center': (75.3, 31.1)},
            'Haryana': {'pop': 2.8e7, 'cov': 93.2, 'count': 22, 'center': (76.1, 29.1)},
            'Chhattisgarh': {'pop': 2.9e7, 'cov': 89.8, 'count': 33, 'center': (81.9, 21.3)},
            'Delhi': {'pop': 1.9e7, 'cov': 96.2, 'count': 11, 'center': (77.1, 28.7)},
            'Jammu & Kashmir': {'pop': 1.4e7, 'cov': 82.4, 'count': 20, 'center': (76.8, 34.1)},
            'Uttarakhand': {'pop': 1.1e7, 'cov': 91.6, 'count': 13, 'center': (79.0, 30.1)},
            'Himachal Pradesh': {'pop': 0.75e7, 'cov': 95.4, 'count': 12, 'center': (77.2, 31.1)},
            'Tripura': {'pop': 0.4e7, 'cov': 88.9, 'count': 8, 'center': (91.7, 23.9)},
            'Meghalaya': {'pop': 0.35e7, 'cov': 79.3, 'count': 12, 'center': (91.4, 25.5)},
            'Manipur': {'pop': 0.31e7, 'cov': 81.2, 'count': 16, 'center': (93.9, 24.8)},
            'Nagaland': {'pop': 0.22e7, 'cov': 76.8, 'count': 16, 'center': (94.5, 26.2)},
            'Goa': {'pop': 0.16e7, 'cov': 98.5, 'count': 2, 'center': (73.8, 15.4)},
            'Arunachal Pradesh': {'pop': 0.15e7, 'cov': 72.1, 'count': 26, 'center': (94.7, 28.2)},
            'Mizoram': {'pop': 0.12e7, 'cov': 84.6, 'count': 11, 'center': (92.7, 23.2)},
            'Sikkim': {'pop': 0.07e7, 'cov': 93.8, 'count': 6, 'center': (88.5, 27.3)},
        }
        
        districts = []
        district_idx = 1
        
        for state_name, state_info in states_data.items():
            state_pop = state_info['pop']
            base_coverage = state_info['cov']
            num_districts = state_info['count']
            center_lon, center_lat = state_info['center']
            
            for i in range(num_districts):
<<<<<<< Updated upstream
                # Generate district-level variations
                district_pop = int(state_pop / num_districts * (0.5 + np.random.random()))
                
                # Coverage varies by ±5% from state average
                coverage = base_coverage + np.random.uniform(-5, 3)
                coverage = max(60, min(100, coverage))
                
                enrolments = int(district_pop * coverage / 100)
                
                # Updates are ~5-10% of total yearly
                biometric_updates = int(enrolments * 0.08 * np.random.uniform(0.5, 1.5))
                demographic_updates = int(enrolments * 0.05 * np.random.uniform(0.5, 1.5))
                
                # Randomize coordinates around state center
=======
                district_pop = int(state_pop / num_districts * (0.5 + np.random.random()))
                coverage = base_coverage + np.random.uniform(-5, 3)
                coverage = max(60, min(100, coverage))
                enrolments = int(district_pop * coverage / 100)
                biometric_updates = int(enrolments * 0.08 * np.random.uniform(0.5, 1.5))
                demographic_updates = int(enrolments * 0.05 * np.random.uniform(0.5, 1.5))
>>>>>>> Stashed changes
                lat = center_lat + np.random.uniform(-1.5, 1.5)
                lon = center_lon + np.random.uniform(-1.5, 1.5)
                
                districts.append(DistrictData(
                    district_code=f"D{district_idx:04d}",
                    district_name=f"District_{i+1}",
                    state_code=state_name[:2].upper(),
                    state_name=state_name,
                    total_enrolments=enrolments,
                    biometric_updates=biometric_updates,
                    demographic_updates=demographic_updates,
                    population=district_pop,
                    coverage_percent=round(coverage, 1),
                    latitude=lat,
                    longitude=lon,
                    last_updated=datetime.now().isoformat()
                ))
<<<<<<< Updated upstream
                
=======
>>>>>>> Stashed changes
                district_idx += 1
        
        return districts
    
    def to_dataframe(self, districts: List[DistrictData] = None) -> pd.DataFrame:
<<<<<<< Updated upstream
        """Convert district data to pandas DataFrame"""
=======
        """Convert to DataFrame with preprocessing"""
>>>>>>> Stashed changes
        if districts is None:
            districts = self.get_district_enrolments()
        
        data = []
        for d in districts:
            data.append({
                'district_code': d.district_code,
                'district': d.district_name,
                'state_code': d.state_code,
                'state': d.state_name,
                'enrolments': d.total_enrolments,
                'biometric_updates': d.biometric_updates,
                'demographic_updates': d.demographic_updates,
                'population': d.population,
                'coverage': d.coverage_percent,
                'lat': d.latitude,
                'lon': d.longitude,
                'last_updated': d.last_updated
            })
        
<<<<<<< Updated upstream
        return pd.DataFrame(data)
    
    def calculate_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Normalized Enrollment Intensity
        
        Formula: (current_enrolments - previous_enrolments) / population
        
        This prevents 1000%+ velocity spikes in low-population areas
        """
        result = df.copy()
        
        # Simulate previous period (10% lower on average with variance)
=======
        df = pd.DataFrame(data)
        
        # Apply univariate analysis - impute missing values
        missing = self.analyzer.detect_missing(df)
        if any(v > 0 for v in missing.values()):
            print(f"Missing values detected: {missing}")
            df = self.analyzer.impute_missing(df, strategy='state_median')
        
        return df
    
    def calculate_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Normalized Intensity with preprocessing"""
        result = df.copy()
        
        # Simulate previous period
>>>>>>> Stashed changes
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        prev_factor = 0.95 + np.random.uniform(-0.05, 0.03, len(df))
        result['previous_enrolments'] = (result['enrolments'] * prev_factor).astype(int)
        
<<<<<<< Updated upstream
        # Calculate normalized intensity (per-capita change)
=======
        # Normalized Intensity = (Current - Previous) / Population
>>>>>>> Stashed changes
        result['intensity'] = (
            (result['enrolments'] - result['previous_enrolments']) / 
            result['population'].clip(lower=1)
        )
        
<<<<<<< Updated upstream
        # Log transform for better visualization
        result['log_intensity'] = np.sign(result['intensity']) * np.log1p(np.abs(result['intensity']))
        
        # Min-Max scale to [0, 1]
=======
        # Detect and winsorize outliers
        outliers = self.analyzer.detect_outliers(result['intensity'], method='iqr')
        if outliers.any():
            print(f"Detected {outliers.sum()} outliers in intensity")
            result['intensity'] = self.analyzer.winsorize_outliers(result['intensity'])
        
        # Log transform
        result['log_intensity'] = np.sign(result['intensity']) * np.log1p(np.abs(result['intensity']))
        
        # Min-Max scale
>>>>>>> Stashed changes
        min_val = result['log_intensity'].min()
        max_val = result['log_intensity'].max()
        if max_val - min_val > 1e-10:
            result['scaled_intensity'] = (result['log_intensity'] - min_val) / (max_val - min_val)
        else:
            result['scaled_intensity'] = 0.5
        
        return result


<<<<<<< Updated upstream
# Singleton instance
_fetcher = None

def get_data_fetcher() -> UIDAIDataFetcher:
    """Get singleton data fetcher instance"""
=======
_fetcher = None

def get_data_fetcher() -> UIDAIDataFetcher:
>>>>>>> Stashed changes
    global _fetcher
    if _fetcher is None:
        _fetcher = UIDAIDataFetcher()
    return _fetcher


def fetch_district_data() -> pd.DataFrame:
<<<<<<< Updated upstream
    """Convenience function to get district data as DataFrame with intensity"""
=======
    """Convenience function to get district data with intensity"""
>>>>>>> Stashed changes
    fetcher = get_data_fetcher()
    df = fetcher.to_dataframe()
    df = fetcher.calculate_intensity(df)
    return df
