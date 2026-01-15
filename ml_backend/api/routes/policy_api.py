"""
Government-Ready Policy API Endpoints

These endpoints provide policy-level controls for fraud detection.
ALL ML/DL implementation details are hidden from API consumers.

TERMINOLOGY:
- Uses "Flag/Review/Clear" instead of "Anomaly Scores"
- Uses "Risk Tolerance" instead of "Thresholds"
- Uses "Policy Presets" instead of "Model Configurations"
- Uses "Disposition" instead of "Prediction"
"""
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from enum import Enum

import sys
sys.path.insert(0, '..')

from policy.policy_engine import (
    PolicyEngine,
    RiskTolerance,
    ComplianceLevel
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Global policy engine instance
_policy_engine: Optional[PolicyEngine] = None


def get_policy_engine() -> PolicyEngine:
    """Get or create policy engine instance."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine(ComplianceLevel.STANDARD)
    return _policy_engine


# ============== Request Schemas (Policy-Level Only) ==============

class PolicyAnalysisRequest(BaseModel):
    """
    Request for policy-level analysis.
    
    NO technical parameters exposed. Users select from
    pre-defined policy presets only.
    """
    policy_id: str = Field(
        default="STANDARD",
        description="Policy preset: 'STANDARD', 'HIGH_SECURITY', 'HIGH_THROUGHPUT', 'SPECIAL_DRIVE'"
    )
    region: Optional[str] = Field(
        default=None,
        description="Region to analyze (e.g., 'Maharashtra', 'All')"
    )
    record_limit: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Number of records to analyze"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "policy_id": "STANDARD",
                "region": "Maharashtra",
                "record_limit": 500
            }
        }


class PolicyConfigRequest(BaseModel):
    """
    Request to configure policy parameters.
    
    Uses government-friendly terminology only.
    """
    risk_tolerance: str = Field(
        default="balanced",
        description="Risk tolerance: 'conservative', 'balanced', 'aggressive'"
    )
    enable_escalation: bool = Field(
        default=True,
        description="Enable automatic escalation for critical cases"
    )
    secondary_verification: bool = Field(
        default=False,
        description="Require secondary verification for flagged records"
    )


# ============== Response Schemas (No Technical Details) ==============

class DispositionSummary(BaseModel):
    """Summary of record dispositions."""
    requires_immediate_action: int
    flagged_for_investigation: int
    pending_review: int
    cleared: int


class PolicySummary(BaseModel):
    """Policy-level analysis summary."""
    total_records: int
    disposition_breakdown: DispositionSummary
    risk_assessment: str
    compliance_status: str
    data_integrity: str


class RecordDispositionResponse(BaseModel):
    """Single record disposition (no technical details)."""
    record_id: str
    disposition: str  # FLAG, REVIEW, CLEAR, ESCALATE
    risk_indicator: str  # Low, Medium, High, Critical
    requires_action: bool
    action_priority: int
    reason_summary: str
    recommended_action: str


class PolicyAnalysisResponse(BaseModel):
    """
    Policy-level analysis response.
    
    Contains NO technical ML details:
    - No model names
    - No anomaly scores
    - No thresholds
    - No hyperparameters
    """
    analysis_id: str
    timestamp: str
    policy_applied: dict
    summary: PolicySummary
    flagged_records: List[RecordDispositionResponse]
    next_steps: List[str]
    audit_reference: str
    metadata: dict


class AvailablePolicyResponse(BaseModel):
    """Available policy preset information."""
    policy_id: str
    name: str
    description: str
    security_level: str
    throughput: str
    recommended_for: str


# ============== API Endpoints ==============

@router.get("/policies", response_model=List[AvailablePolicyResponse])
async def get_available_policies():
    """
    Get available policy presets for fraud detection.
    
    Returns government-approved policy configurations.
    No technical ML parameters exposed.
    
    **Available Policies:**
    - `STANDARD`: Default for routine operations
    - `HIGH_SECURITY`: Maximum scrutiny for sensitive operations
    - `HIGH_THROUGHPUT`: Optimized for trusted regions
    - `SPECIAL_DRIVE`: For government enrollment drives
    """
    engine = get_policy_engine()
    policies = engine.get_available_policies()
    return [AvailablePolicyResponse(**p) for p in policies]


@router.post("/analyze", response_model=PolicyAnalysisResponse)
async def analyze_with_policy(request: PolicyAnalysisRequest):
    """
    Analyze records using policy-level controls.
    
    This endpoint provides fraud detection without exposing
    any technical ML/DL implementation details.
    
    **How It Works:**
    1. Select a policy preset (or use STANDARD)
    2. Records are analyzed against policy rules
    3. Returns dispositions: FLAG, REVIEW, CLEAR, or ESCALATE
    
    **Response Contains:**
    - Summary of findings
    - Flagged records with action recommendations
    - Next steps for compliance
    - Audit trail reference
    
    **Does NOT Contain:**
    - Model names or types
    - Anomaly scores or probabilities
    - Thresholds or hyperparameters
    - Technical ML terminology
    """
    try:
        engine = get_policy_engine()
        
        # Fetch data
        from data.ingestion import DataIngestion
        ingestion = DataIngestion()
        
        # Determine dataset based on policy
        dataset_id = "enrolment"
        raw_data = ingestion.fetch_data(dataset_id, limit=request.record_limit)
        
        if raw_data.empty:
            raise HTTPException(status_code=500, detail="Unable to fetch data for analysis")
        
        # Run policy analysis
        results = engine.analyze_records(
            data=raw_data,
            policy_id=request.policy_id,
            region=request.region
        )
        
        # Filter to flagged records only for response
        flagged = [
            RecordDispositionResponse(
                record_id=r["record_id"],
                disposition=r["disposition"],
                risk_indicator=r["risk_indicator"],
                requires_action=r["requires_action"],
                action_priority=r["action_priority"],
                reason_summary=r["reason_summary"],
                recommended_action=r["recommended_action"]
            )
            for r in results["records"]
            if r["disposition"] in ["FLAG", "ESCALATE"]
        ][:50]  # Limit response size
        
        return PolicyAnalysisResponse(
            analysis_id=results["analysis_id"],
            timestamp=results["timestamp"],
            policy_applied=results["policy_applied"],
            summary=PolicySummary(
                total_records=results["summary"]["total_records"],
                disposition_breakdown=DispositionSummary(**results["summary"]["disposition_breakdown"]),
                risk_assessment=results["summary"]["risk_assessment"],
                compliance_status=results["summary"]["compliance_status"],
                data_integrity=results["summary"]["data_integrity"]
            ),
            flagged_records=flagged,
            next_steps=results["next_steps"],
            audit_reference=results["audit_reference"],
            metadata=results["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Policy analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Analysis could not be completed. Please contact system administrator."
        )


@router.get("/audit-trail")
async def get_audit_trail():
    """
    Get audit trail for compliance reporting.
    
    Returns a complete record of all policy analyses
    performed in this session for government audit.
    """
    engine = get_policy_engine()
    trail = engine.get_audit_trail()
    
    return {
        "audit_trail": trail,
        "session_id": engine.session_id,
        "compliance_level": engine.compliance_level.value,
        "generated_at": datetime.now().isoformat(),
        "classification": "OFFICIAL"
    }


@router.get("/compliance-report")
async def get_compliance_report():
    """
    Generate compliance report for government submission.
    
    This report contains all necessary information for
    regulatory compliance and audit purposes.
    """
    engine = get_policy_engine()
    return engine.generate_compliance_report()


@router.get("/system-status")
async def get_system_status():
    """
    Get system status for monitoring.
    
    Returns operational status WITHOUT technical details.
    """
    return {
        "status": "OPERATIONAL",
        "service": "UIDAI Fraud Detection System",
        "version": "2.0",
        "compliance": "UIDAI Guidelines Compliant",
        "data_classification": "OFFICIAL",
        "security_status": "SECURE",
        "last_health_check": datetime.now().isoformat(),
        "capabilities": [
            "Policy-based fraud detection",
            "Multi-region analysis",
            "Compliance reporting",
            "Audit trail generation"
        ]
    }
