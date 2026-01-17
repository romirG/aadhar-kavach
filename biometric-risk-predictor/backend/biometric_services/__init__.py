# Biometric Risk Predictor - Services Package
"""
Services for the Biometric Re-enrollment Risk Predictor.

Modules:
- biometric_risk_service: Core ML risk prediction logic
- data_gov_client: Government API data fetching
- feature_engineering: Feature extraction and engineering
- groq_ai_service: AI-powered recommendations via Groq LLaMA
"""

from .biometric_risk_service import biometric_risk_service
from .data_gov_client import data_client

__all__ = ['biometric_risk_service', 'data_client']
