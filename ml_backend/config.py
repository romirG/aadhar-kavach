"""
Configuration for Biometric Re-enrollment Risk Predictor

PRIVACY NOTICE:
- This application uses ONLY aggregated, anonymized data from public government APIs
- No individual Aadhaar numbers or personally identifiable information is processed
- All data is at state/district/age-group level only
"""

import os
from dotenv import load_dotenv

load_dotenv()

# data.gov.in API Configuration
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "579b464db66ec23bdd0000015cfbfd5b9e5a4b366992c1f538e4a2b8")
DATA_GOV_BASE_URL = "https://api.data.gov.in/resource"

# Dataset Resource IDs
DATASETS = {
    "enrolment": {
        "id": "ecd49b12-3084-4521-8f7e-ca8bf72069ba",
        "name": "Aadhaar Monthly Enrolment Data",
        "description": "New enrollments per month with age group distributions",
        "fields": ["date", "state", "district", "pincode", "age_0_5", "age_5_17", "age_18_greater"]
    },
    "demographic": {
        "id": "19eac040-0b94-49fa-b239-4f2fd8677d53",
        "name": "Aadhaar Demographic Monthly Update Data",
        "description": "Demographic updates with age group distributions",
        "fields": ["date", "state", "district", "pincode", "demo_age_5_17", "demo_age_17_"]
    },
    "biometric": {
        "id": "65454dab-1517-40a3-ac1d-47d4dfe6891c",
        "name": "Aadhaar Biometric Monthly Update Data",
        "description": "Biometric updates (fingerprint/iris) with age groups",
        "fields": ["date", "state", "district", "pincode", "bio_age_5_17", "bio_age_17_"]
    }
}

# High-risk states (historically higher biometric failure rates)
HIGH_RISK_STATES = ["Jharkhand", "Bihar", "Chhattisgarh", "Madhya Pradesh", "Odisha"]

# Model Configuration
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "kmeans": {
        "n_clusters": 4,  # Low, Medium, High, Critical risk
        "random_state": 42
    },
    "isolation_forest": {
        "contamination": 0.1,  # Expected proportion of outliers
        "random_state": 42
    }
}

# Risk thresholds
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8
}

# Age buckets for feature engineering
AGE_BUCKETS = {
    "child": (0, 18),
    "adult": (18, 60),
    "elderly": (60, 120)
}

# Output directories
OUTPUT_DIR = "outputs"
VISUALIZATION_DIR = f"{OUTPUT_DIR}/visualizations"
MODEL_DIR = f"{OUTPUT_DIR}/models"
REPORT_DIR = f"{OUTPUT_DIR}/reports"
