"""
Biometric Risk Predictor - Configuration

PRIVACY NOTICE:
- Uses ONLY aggregated, anonymized data from public government APIs
- No individual Aadhaar numbers or PII processed
"""

import os
from dotenv import load_dotenv

load_dotenv()

# data.gov.in API Configuration
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "579b464db66ec23bdd00000107b605eca2de45d37099f6ba865c6cea")
DATA_GOV_BASE_URL = "https://api.data.gov.in/resource"

# Dataset Resource IDs
DATASETS = {
    "enrolment": {
        "id": "ecd49b12-3084-4521-8f7e-ca8bf72069ba",
        "name": "Aadhaar Monthly Enrolment Data",
        "description": "New enrollments per month with age group distributions",
        "resource_id": "ecd49b12-3084-4521-8f7e-ca8bf72069ba",
        "fields": ["date", "state", "district", "pincode", "age_0_5", "age_5_17", "age_18_greater"]
    },
    "demographic": {
        "id": "19eac040-0b94-49fa-b239-4f2fd8677d53",
        "name": "Aadhaar Demographic Monthly Update Data",
        "description": "Demographic updates with age group distributions",
        "resource_id": "19eac040-0b94-49fa-b239-4f2fd8677d53",
        "fields": ["date", "state", "district", "pincode", "demo_age_5_17", "demo_age_17_"]
    },
    "biometric": {
        "id": "65454dab-1517-40a3-ac1d-47d4dfe6891c",
        "name": "Aadhaar Biometric Monthly Update Data",
        "description": "Biometric updates (fingerprint/iris) with age groups",
        "resource_id": "65454dab-1517-40a3-ac1d-47d4dfe6891c",
        "fields": ["date", "state", "district", "pincode", "bio_age_5_17", "bio_age_17_"]
    }
}

# High-risk states (historically higher biometric failure rates)
HIGH_RISK_STATES = ["Jharkhand", "Bihar", "Chhattisgarh", "Madhya Pradesh", "Odisha"]
