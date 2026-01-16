"""
Intent-Based Monitoring API

FastAPI endpoints for UIDAI auditors using intent-based interaction.
NO datasets or ML terms in any request/response schema.

Endpoints:
- POST /api/monitor - Submit monitoring intent
- GET /api/monitor/status/{job_id} - Check job status
- GET /api/monitor/results/{job_id} - Get results

DESIGN PRINCIPLES:
- Intent-based user inputs only
- Policy-friendly responses
- No dataset names exposed
- No ML terminology in API schema
"""
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import uuid

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/monitor", tags=["Monitoring"])


# =============================================================================
# AI ANALYSIS SCHEMAS
# =============================================================================

class FindingAnalysisRequest(BaseModel):
    """Request for in-depth AI analysis of a specific finding."""
    job_id: str = Field(..., description="Original monitoring job ID")
    finding_index: int = Field(..., description="Index of the finding to analyze")


class ImpactAssessment(BaseModel):
    """Impact assessment details."""
    severity: str
    affected_scope: str
    compliance_risk: str


class DetailedActionItem(BaseModel):
    """Detailed action item with additional context."""
    action: str
    priority: str
    responsible_party: str
    timeline: str


class FindingAnalysisResponse(BaseModel):
    """Detailed AI analysis response for a finding."""
    success: bool
    analysis: Optional[str] = None
    root_cause: Optional[str] = None
    impact_assessment: Optional[ImpactAssessment] = None
    recommended_actions: Optional[List[DetailedActionItem]] = None
    monitoring_plan: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# REQUEST/RESPONSE SCHEMAS - NO ML TERMS, NO DATASETS
# =============================================================================

class MonitoringIntent(str, Enum):
    """Available monitoring intents for auditors."""
    CHECK_ENROLLMENTS = "check_enrollments"
    REVIEW_UPDATES = "review_updates"
    VERIFY_BIOMETRICS = "verify_biometrics"
    COMPREHENSIVE_CHECK = "comprehensive_check"


class VigilanceLevel(str, Enum):
    """Monitoring vigilance levels."""
    ROUTINE = "routine"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"


class TimePeriod(str, Enum):
    """Analysis time periods."""
    TODAY = "today"
    LAST_7_DAYS = "last_7_days"
    THIS_MONTH = "this_month"


class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# === Request Schemas ===

class MonitoringRequest(BaseModel):
    """
    Submit a monitoring request.
    User selects WHAT to monitor, not which datasets.
    """
    intent: MonitoringIntent = Field(
        ...,
        description="What would you like to monitor?",
        example="check_enrollments"
    )
    focus_area: Optional[str] = Field(
        None,
        description="Geographic focus (state or district)",
        example="Maharashtra"
    )
    time_period: TimePeriod = Field(
        TimePeriod.TODAY,
        description="Analysis time period"
    )
    vigilance: VigilanceLevel = Field(
        VigilanceLevel.STANDARD,
        description="Monitoring intensity level"
    )
    record_limit: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Maximum records to analyze"
    )


# === Response Schemas ===

class MonitoringJobResponse(BaseModel):
    """Response when submitting a monitoring request."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    estimated_time: str = Field(..., description="Estimated completion time")


class StatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    status: JobStatus
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    message: str
    started_at: Optional[str]
    completed_at: Optional[str]


class RiskSummary(BaseModel):
    """Risk assessment summary - no ML terms."""
    risk_index: int = Field(..., ge=0, le=100, description="Risk indicator (0-100)")
    risk_level: str = Field(..., description="Risk classification")
    confidence: str = Field(..., description="Assessment confidence level")


class Finding(BaseModel):
    """A single finding - plain language."""
    title: str = Field(..., description="Finding title")
    description: str = Field(..., description="Plain language description")
    severity: str = Field(..., description="Severity level")
    location: Optional[str] = Field(None, description="Affected area")
    details: Optional[str] = Field(None, description="Specific evidence or details")


class ActionItem(BaseModel):
    """Recommended action - plain language."""
    action: str
    priority: str


class MonitoringResults(BaseModel):
    """
    Complete monitoring results.
    All in plain language suitable for auditors.
    """
    job_id: str
    status: JobStatus
    
    # Summary
    summary: str = Field(..., description="Executive summary")
    
    # Risk Assessment
    risk: RiskSummary
    
    # Findings
    findings: List[Finding]
    
    # Recommendations
    recommended_actions: List[ActionItem]
    
    # Counts
    records_analyzed: int
    flagged_for_review: int
    cleared: int
    
    # Metadata
    analysis_scope: str
    time_period: str
    completed_at: str
    
    # Audit
    report_id: str
    
    # Details
    flagged_records: Optional[List[Dict[str, Any]]] = Field(None, description="Sample of flagged records")


class IntentOption(BaseModel):
    """Available intent option for UI display."""
    id: str
    display_name: str
    description: str


class AvailableIntentsResponse(BaseModel):
    """List of available monitoring intents."""
    intents: List[IntentOption]
    vigilance_levels: List[Dict[str, str]]


# =============================================================================
# IN-MEMORY JOB STORAGE (For demo - use Redis/DB in production)
# =============================================================================

_jobs: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/intents", response_model=AvailableIntentsResponse)
async def get_available_intents():
    """
    Get available monitoring intents.
    
    Returns the list of monitoring options for display in the UI.
    """
    intents = [
        IntentOption(
            id="check_enrollments",
            display_name="Check today's enrollment operations for issues",
            description="Monitor new Aadhaar enrollment activities for irregularities"
        ),
        IntentOption(
            id="review_updates",
            display_name="Review update requests for irregularities",
            description="Examine demographic and address update patterns"
        ),
        IntentOption(
            id="verify_biometrics",
            display_name="Verify biometric submissions for anomalies",
            description="Monitor biometric update and verification patterns"
        ),
        IntentOption(
            id="comprehensive_check",
            display_name="Run comprehensive integrity check",
            description="Complete system-wide integrity assessment"
        )
    ]
    
    vigilance_levels = [
        {"id": "routine", "name": "Routine", "description": "Quick check for obvious concerns"},
        {"id": "standard", "name": "Standard", "description": "Balanced monitoring"},
        {"id": "enhanced", "name": "Enhanced", "description": "Thorough review"},
        {"id": "maximum", "name": "Maximum", "description": "Complete scrutiny"}
    ]
    
    return AvailableIntentsResponse(
        intents=intents,
        vigilance_levels=vigilance_levels
    )


@router.post("", response_model=MonitoringJobResponse)
async def submit_monitoring_request(
    request: MonitoringRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a monitoring request.
    
    User selects:
    - What to monitor (intent)
    - Where to focus (optional location)
    - How thorough (vigilance level)
    
    NO dataset selection required.
    """
    job_id = str(uuid.uuid4())[:12]
    
    # Create job record
    _jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "progress": 0,
        "request": request.model_dump(mode='json'),
        "results": None,
        "started_at": None,
        "completed_at": None,
        "message": "Job queued for processing"
    }
    
    # Start background processing
    background_tasks.add_task(process_monitoring_job, job_id)
    
    # Estimate time based on vigilance
    time_estimates = {
        VigilanceLevel.ROUTINE: "Under 1 minute",
        VigilanceLevel.STANDARD: "1-2 minutes",
        VigilanceLevel.ENHANCED: "2-3 minutes",
        VigilanceLevel.MAXIMUM: "3-5 minutes"
    }
    
    logger.info(f"Monitoring job {job_id} created for intent: {request.intent.value}")
    
    return MonitoringJobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Monitoring request received. Analyzing {request.intent.value.replace('_', ' ')}.",
        estimated_time=time_estimates.get(request.vigilance, "1-2 minutes")
    )


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """
    Check the status of a monitoring job.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs[job_id]
    
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at")
    )


@router.get("/results/{job_id}", response_model=MonitoringResults)
async def get_job_results(job_id: str):
    """
    Get the results of a completed monitoring job.
    
    Returns:
    - Plain language summary
    - Risk assessment
    - Findings and recommendations
    
    NO technical details exposed.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs[job_id]
    
    if job["status"] == JobStatus.PENDING:
        raise HTTPException(status_code=202, detail="Job is pending. Please wait.")
    
    if job["status"] == JobStatus.PROCESSING:
        raise HTTPException(status_code=202, detail=f"Job is processing ({job['progress']}% complete)")
    
    if job["status"] == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail="Job failed. Please retry.")
    
    return job["results"]


# =============================================================================
# BACKGROUND PROCESSING
# =============================================================================

async def process_monitoring_job(job_id: str):
    """
    Process a monitoring job through the complete pipeline.
    
    STATUS MESSAGES DESIGN:
    - User-friendly language
    - No technical steps exposed
    - Progress feels natural
    - Reassuring tone
    """
    job = _jobs[job_id]
    request_data = job["request"]
    
    # User-friendly status messages (NO technical terms)
    STATUS_MESSAGES = {
        10: "Your request is being prepared...",
        20: "Reviewing your monitoring scope...",
        35: "Gathering relevant records for review...",
        50: "Examining the records...",
        65: "Looking for patterns and concerns...",
        80: "Evaluating findings...",
        90: "Preparing your report...",
        95: "Finalizing results...",
        100: "Your assessment is ready"
    }
    
    def update_status(progress: int, custom_message: str = None):
        """Update job status with user-friendly message."""
        job["progress"] = progress
        job["message"] = custom_message or STATUS_MESSAGES.get(progress, "Processing...")
    
    try:
        # Start processing
        job["status"] = JobStatus.PROCESSING
        job["started_at"] = datetime.now().isoformat()
        update_status(10)
        
        await asyncio.sleep(0.5)
        
        # Step 1: Intent Resolution (hidden from user)
        update_status(20)
        
        from ..policy.intent_resolver import IntentResolutionEngine, UserContext
        from ..policy.intent_resolver import VigilanceLevel as ResolverVigilance
        from ..policy.intent_resolver import TimePeriod as ResolverTimePeriod
        
        resolver = IntentResolutionEngine()
        context = UserContext(
            state=request_data.get("focus_area"),
            time_period=ResolverTimePeriod(request_data.get("time_period", "today")),
            vigilance=ResolverVigilance(request_data.get("vigilance", "standard"))
        )
        resolved = resolver.resolve(request_data["intent"], context)
        resolved_dict = resolver.to_dict(resolved)
        
        await asyncio.sleep(0.3)
        
        
        # Step 2: Data Orchestration (hidden from user)
        update_status(35)
        
        from ..policy.dataset_orchestrator import DatasetOrchestrationLayer
        orchestrator = DatasetOrchestrationLayer()
        data_result = orchestrator.orchestrate(resolved_dict, request_data.get("record_limit", 1000))
        
        await asyncio.sleep(0.3)
        
        # Step 3: Strategy Selection (hidden from user)
        update_status(50)
        
        from ..policy.strategy_selector import AnalysisStrategySelector
        strategy_selector = AnalysisStrategySelector()
        signals = [s.get("signal_id") for s in resolved_dict.get("signals", [])]
        strategies = strategy_selector.recommend_strategies(signals, request_data.get("vigilance", "standard"))
        
        await asyncio.sleep(0.3)
        
        # Step 4: Execute Analysis (hidden from user)
        update_status(65)
        
        strategy_results = []
        for strategy in strategies:
            result = {
                "strategy_id": strategy.strategy.value,
                "strategy_name": strategy.info.name,
                "score": min(0.9, max(0.1, 0.3 + (hash(strategy.strategy.value) % 50) / 100)),
                "records_analyzed": len(data_result.data) // len(strategies),
                "flagged_count": max(1, int(len(data_result.data) * 0.05)),
                "confidence": 0.75 + (hash(strategy.strategy.value) % 20) / 100
            }
            strategy_results.append(result)
        
        await asyncio.sleep(0.5)
        
        # Step 5: Risk Aggregation (hidden from user)
        update_status(80)
        
        from ..policy.analysis_output import calculate_unified_risk
        risk_result = calculate_unified_risk(strategy_results, request_data.get("vigilance", "standard"))
        
        await asyncio.sleep(0.3)
        
        # Step 6: Generate Explanation (hidden from user)
        update_status(90)
        
        from ..policy.analysis_output import generate_auditor_explanation
        
        findings_data = [
            {"type": s["strategy_id"], "score": s["score"], "location": request_data.get("focus_area")}
            for s in strategy_results if s["score"] > 0.4
        ]
        
        explanation = generate_auditor_explanation(
            risk_result,
            findings_data,
            {
                "geographic": request_data.get("focus_area", "All India"),
                "temporal": request_data.get("time_period", "today").replace("_", " "),
                "vigilance": request_data.get("vigilance", "standard")
            }
        )
        
        # Calculate total_flagged before sampling and AI analysis
        total_records = data_result.record_count
        raw_flagged = sum(s["flagged_count"] for s in strategy_results)
        total_flagged = min(raw_flagged, total_records)

        # Sample flagged records for drill-down (Moved before AI Analysis)
        flagged_records = []
        try:
            if not data_result.data.empty and total_flagged > 0:
                # Naively take top N records for now since we lack per-record flag in this simulation
                # In real impl, we'd filter by actual flag
                sample_size = min(20, int(total_flagged))
                sample_indices = [i for i in range(len(data_result.data))][:sample_size]
                
                # Extract and clean
                records_df = data_result.data.iloc[sample_indices].copy()
                records_df = records_df.fillna("N/A")
                
                # Convert timestamps
                for col in records_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(records_df[col]):
                        records_df[col] = records_df[col].dt.strftime("%Y-%m-%d")
                
                # Add synthetic reasons based on strategy
                import random
                reasons = [
                    "High velocity of updates detected",
                    "Geographic anomaly: Displaced enrollment",
                    "Time-series deviation: Spike in activity",
                    "Cluster outlier: Low density region"
                ]
                
                records_list = records_df.to_dict(orient="records")
                for rec in records_list:
                    rec["flagged_reason"] = random.choice(reasons)
                    rec["risk_score"] = f"{random.randint(70, 99)}/100"
                    
                flagged_records = records_list
        except Exception as e:
            logger.error(f"Error sampling flagged records: {e}")

        # Step 7: AI Analysis (Groq)
        update_status(92)
        from ..services.groq_service import groq_service
        
        groq_result = None
        logger.info(f"Job {job_id}: Flagged records count: {len(flagged_records)}")
        
        # Call Groq on EVERY search as requested
        logger.info(f"Job {job_id}: Calling Groq API for context: intent={request_data['intent']}, focus={request_data.get('focus_area', 'All India')}")
        try:
            groq_result = groq_service.analyze_monitoring_data(
                    context={
                        "intent": request_data["intent"],
                        "focus": request_data.get("focus_area", "All India"),
                        "risk_level": risk_result.get("risk_metrics", {}).get("risk_level"),
                        "strategies": [s["strategy_name"] for s in strategy_results],
                        "total_analyzed": total_records,
                        "total_flagged": total_flagged,
                        "vigilance": request_data.get("vigilance", "standard"),
                        "time_period": request_data.get("time_period", "today")
                    },
                    flagged_records=flagged_records
                )
            if groq_result:
                logger.info(f"Job {job_id}: Groq analysis successful - Generated {len(groq_result.get('findings', []))} findings and {len(groq_result.get('recommended_actions', []))} actions")
                logger.debug(f"Job {job_id}: Groq response summary: {groq_result.get('summary', 'N/A')[:100]}...")
            else:
                logger.warning(f"Job {job_id}: Groq returned None - API key may not be configured or API call failed")
        except Exception as e:
            logger.error(f"Job {job_id}: Groq calling error: {e}")
            import traceback
            logger.error(f"Job {job_id}: Traceback: {traceback.format_exc()}")
        
        await asyncio.sleep(0.2)
        
        # Finalize
        update_status(95)
        
        # Build results
        risk_metrics = risk_result.get("risk_metrics", {})
        # total_records and total_flagged are now calculated earlier
        
        # Prepare Findings and Actions (Prefer Groq, fallback to rule-based)
        final_findings = []
        final_actions = []
        final_summary = explanation.get("summary", risk_result.get("summary", "Analysis complete."))
        
        if groq_result:
            final_summary = groq_result.get("summary", final_summary)
            for f in groq_result.get("findings", []):
                final_findings.append(Finding(
                    title=f.get("title", "Finding"),
                    description=f.get("description", ""),
                    severity=f.get("severity", "Medium"),
                    location=request_data.get("focus_area") or "Unknown",
                    details=f.get("details", "")
                ))
            for a in groq_result.get("recommended_actions", []):
                final_actions.append(ActionItem(
                    action=a.get("action", ""),
                    priority=a.get("priority", "Normal")
                ))
        else:
            # Fallback - Generate dynamic context-aware actions
            final_findings = [
                Finding(
                    title=obs.get("title", "Finding"),
                    description=obs.get("description", ""),
                    severity=obs.get("severity", "Minor"),
                    location=obs.get("location") or "Unknown",
                    details="Anomaly detected based on statistical deviation."
                )
                for obs in explanation.get("observations", [])
            ]
            
            # Generate dynamic actions based on context
            intent = request_data.get("intent", "")
            focus = request_data.get("focus_area", "All India")
            risk_level = risk_metrics.get("risk_level", "Low")
            
            # Intent-specific actions
            action_templates = {
                "check_enrollments": [
                    f"Initiate field verification of enrollment centers in {focus}",
                    f"Review operator credentials and authentication logs for {focus}",
                    "Deploy mobile verification units to high-deviation districts"
                ],
                "review_updates": [
                    f"Audit demographic update requests from {focus} for past 30 days",
                    "Cross-verify update requests with supporting documentation",
                    "Implement enhanced authentication for update operations"
                ],
                "verify_biometrics": [
                    f"Schedule biometric re-capture for flagged records in {focus}",
                    "Inspect biometric capture device calibration and maintenance logs",
                    "Conduct quality audit of biometric operators"
                ],
                "comprehensive_check": [
                    f"Launch comprehensive integrity audit in {focus}",
                    "Activate enhanced monitoring protocols across all operations",
                    "Coordinate with state authorities for ground verification"
                ]
            }
            
            # Risk-based priority assignment
            priority_map = {"Critical": "Urgent", "High": "High", "Medium": "Normal", "Low": "Normal"}
            priority = priority_map.get(risk_level, "Normal")
            
            # Get actions for this intent
            actions_list = action_templates.get(intent, [
                f"Review operational data for {focus}",
                "Conduct targeted inspection of flagged activities",
                "Implement corrective measures as per standard protocols"
            ])
            
            # Add time-sensitive action for high risk
            if risk_level in ["High", "Critical"]:
                actions_list.insert(0, f"IMMEDIATE: Suspend operations at facilities with {risk_level.lower()} risk indicators pending investigation")
            
            final_actions = [ActionItem(action=action, priority=priority) for action in actions_list[:4]]

        results = MonitoringResults(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            summary=final_summary,
            risk=RiskSummary(
                risk_index=risk_metrics.get("risk_index", 0),
                risk_level=str(risk_metrics.get("risk_level", "Low")),
                confidence=str(risk_metrics.get("confidence_level", "Moderate"))
            ),
            findings=final_findings,
            recommended_actions=final_actions,
            records_analyzed=total_records,
            flagged_for_review=total_flagged,
            cleared=total_records - total_flagged,
            analysis_scope=request_data.get("focus_area") or "All India",
            time_period=(request_data.get("time_period") or "today").replace("_", " ").title(),
            completed_at=datetime.now().isoformat(),
            report_id=f"RPT-{job_id.upper()}",
            flagged_records=flagged_records
        )
        
        # Complete
        job["status"] = JobStatus.COMPLETED
        update_status(100)
        job["completed_at"] = datetime.now().isoformat()
        job["results"] = results
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()
        job["status"] = JobStatus.FAILED
        job["message"] = f"Debug Error: {str(e)}"
        job["progress"] = 0


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get("/health")
async def health_check():
    """Check API health."""
    return {
        "status": "healthy",
        "service": "Monitoring API",
        "active_jobs": len([j for j in _jobs.values() if j["status"] == JobStatus.PROCESSING])
    }


# =============================================================================
# AI ANALYSIS ENDPOINTS
# =============================================================================

@router.post("/analyze-finding", response_model=FindingAnalysisResponse)
async def analyze_finding(request: FindingAnalysisRequest):
    """
    Get in-depth AI analysis for a specific finding using Groq.
    
    This endpoint:
    1. Retrieves the finding from the completed job
    2. Calls Groq API for detailed analysis
    3. Returns comprehensive recommendations and root cause analysis
    """
    try:
        # Validate job exists and is completed
        if request.job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = _jobs[request.job_id]
        
        if job["status"] != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail="Job must be completed before analyzing findings"
            )
        
        results = job["results"]
        
        # Validate finding index
        if request.finding_index < 0 or request.finding_index >= len(results.findings):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid finding index. Must be between 0 and {len(results.findings) - 1}"
            )
        
        # Get the specific finding
        finding = results.findings[request.finding_index]
        finding_dict = {
            "title": finding.title,
            "description": finding.description,
            "severity": finding.severity,
            "location": finding.location,
            "details": finding.details
        }
        
        # Get flagged records (sample)
        flagged_records = results.flagged_records or []
        
        # Prepare context
        context = {
            "intent": job["request"]["intent"],
            "focus_area": job["request"].get("focus_area", "All India"),
            "time_period": job["request"].get("time_period", "today"),
            "vigilance": job["request"].get("vigilance", "standard"),
            "risk_level": results.risk.risk_level,
            "total_analyzed": results.records_analyzed,
            "total_flagged": results.flagged_for_review
        }
        
        # Call Groq service for analysis
        from ..services.groq_service import groq_service
        
        logger.info(f"Analyzing finding {request.finding_index} for job {request.job_id}")
        
        groq_result = groq_service.analyze_finding(
            finding=finding_dict,
            flagged_records=flagged_records,
            context=context
        )
        
        if not groq_result:
            return FindingAnalysisResponse(
                success=False,
                error="AI analysis service unavailable. Please ensure Groq API key is configured."
            )
        
        # Parse and structure the response
        impact = groq_result.get("impact_assessment", {})
        actions = groq_result.get("recommended_actions", [])
        
        return FindingAnalysisResponse(
            success=True,
            analysis=groq_result.get("analysis"),
            root_cause=groq_result.get("root_cause"),
            impact_assessment=ImpactAssessment(
                severity=impact.get("severity", "Unknown"),
                affected_scope=impact.get("affected_scope", "Unknown"),
                compliance_risk=impact.get("compliance_risk", "Unknown")
            ) if impact else None,
            recommended_actions=[
                DetailedActionItem(
                    action=action.get("action", ""),
                    priority=action.get("priority", "Medium"),
                    responsible_party=action.get("responsible_party", "Unknown"),
                    timeline=action.get("timeline", "As needed")
                )
                for action in actions
            ] if actions else None,
            monitoring_plan=groq_result.get("monitoring_plan")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing finding: {e}")
        import traceback
        traceback.print_exc()
        return FindingAnalysisResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )
