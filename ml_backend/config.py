"""
UIDAI ML Fraud Detection Backend - Configuration
"""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    data_gov_api_key: str = ""
    gemini_api_key_1: str = ""
    
    # Server Config
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Data.gov.in API Config
    data_gov_base_url: str = "https://api.data.gov.in/resource"
    enrolment_resource_id: str = "ecd49b12-3084-4521-8f7e-ca8bf72069ba"
    demographic_resource_id: str = "19eac040-0b94-49fa-b239-4f2fd8677d53"
    biometric_resource_id: str = "65454dab-1517-40a3-ac1d-47d4dfe6891c"
    
    # ML Config
    isolation_forest_contamination: float = 0.1
    autoencoder_epochs: int = 50
    autoencoder_latent_dim: int = 16
    hdbscan_min_cluster_size: int = 5
    ensemble_weights: dict = {"isolation_forest": 0.4, "autoencoder": 0.35, "hdbscan": 0.25}
    
    # Thresholds
    high_risk_threshold: float = 0.8
    medium_risk_threshold: float = 0.5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Dataset definitions
DATASETS = {
    "enrolment": {
        "id": "enrolment",
        "name": "Aadhaar Monthly Enrolment Data",
        "description": "Monthly enrolment statistics by state, district, and age groups",
        "resource_id": "ecd49b12-3084-4521-8f7e-ca8bf72069ba",
        "fields": ["date", "state", "district", "pincode", "age_0_5", "age_5_17", "age_18_greater"]
    },
    "demographic": {
        "id": "demographic",
        "name": "Aadhaar Demographic Monthly Update Data",
        "description": "Monthly demographic update statistics by state and district",
        "resource_id": "19eac040-0b94-49fa-b239-4f2fd8677d53",
        "fields": ["date", "state", "district", "pincode", "demo_age_5_17", "demo_age_17_"]
    },
    "biometric": {
        "id": "biometric",
        "name": "Aadhaar Biometric Monthly Update Data",
        "description": "Monthly biometric update statistics by state and district",
        "resource_id": "65454dab-1517-40a3-ac1d-47d4dfe6891c",
        "fields": ["date", "state", "district", "pincode", "bio_age_5_17", "bio_age_17_"]
    }
}
