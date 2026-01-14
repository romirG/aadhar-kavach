"""
Enhanced Dataset Selection API with session management.
Allows users to select and configure datasets for analysis.
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, '..')

from config import DATASETS

logger = logging.getLogger(__name__)
router = APIRouter()

# Session storage (use Redis/DB in production)
active_sessions: Dict[str, Dict[str, Any]] = {}


class DatasetSelectionRequest(BaseModel):
    """Request to select a dataset for analysis."""
    dataset_id: str = Field(..., description="Dataset ID: 'enrolment', 'demographic', or 'biometric'")
    record_limit: int = Field(default=1000, ge=100, le=10000, description="Number of records to fetch")
    date_range: Optional[Dict[str, str]] = Field(default=None, description="Optional date range filter")
    state_filter: Optional[str] = Field(default=None, description="Optional state filter")


class DatasetSelectionResponse(BaseModel):
    """Response after dataset selection."""
    session_id: str
    dataset_id: str
    dataset_name: str
    record_limit: int
    filters_applied: Dict[str, Any]
    estimated_records: int
    selected_at: datetime
    message: str


class SessionInfo(BaseModel):
    """Current session information."""
    session_id: str
    dataset_id: Optional[str]
    dataset_name: Optional[str]
    record_limit: int
    filters: Dict[str, Any]
    analysis_jobs: list
    created_at: datetime
    last_activity: datetime


@router.post("/select-dataset", response_model=DatasetSelectionResponse)
async def select_dataset(request: DatasetSelectionRequest):
    """
    Select a dataset for analysis.
    
    This endpoint allows you to:
    - Choose one of the 3 available UIDAI datasets
    - Configure record limits
    - Apply optional filters
    - Get a session ID for subsequent operations
    
    **Available Datasets:**
    - `enrolment`: Aadhaar Monthly Enrolment Data
    - `demographic`: Aadhaar Demographic Monthly Update Data
    - `biometric`: Aadhaar Biometric Monthly Update Data
    
    **Example Request:**
    ```json
    {
        "dataset_id": "enrolment",
        "record_limit": 1000,
        "state_filter": "Maharashtra"
    }
    ```
    """
    # Validate dataset
    if request.dataset_id not in DATASETS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid dataset_id",
                "available": list(DATASETS.keys()),
                "message": f"Dataset '{request.dataset_id}' not found"
            }
        )
    
    dataset = DATASETS[request.dataset_id]
    
    # Generate session ID
    import uuid
    session_id = str(uuid.uuid4())
    
    # Store session
    filters = {}
    if request.date_range:
        filters["date_range"] = request.date_range
    if request.state_filter:
        filters["state"] = request.state_filter
    
    active_sessions[session_id] = {
        "session_id": session_id,
        "dataset_id": request.dataset_id,
        "dataset_name": dataset["name"],
        "record_limit": request.record_limit,
        "filters": filters,
        "analysis_jobs": [],
        "created_at": datetime.now(),
        "last_activity": datetime.now()
    }
    
    logger.info(f"Dataset selected: {request.dataset_id} (session: {session_id})")
    
    return DatasetSelectionResponse(
        session_id=session_id,
        dataset_id=request.dataset_id,
        dataset_name=dataset["name"],
        record_limit=request.record_limit,
        filters_applied=filters,
        estimated_records=request.record_limit,
        selected_at=datetime.now(),
        message=f"Dataset '{dataset['name']}' selected. Use session_id for subsequent operations."
    )


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """
    Get current session information.
    
    Returns the selected dataset, filters, and analysis history.
    """
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired"
        )
    
    session = active_sessions[session_id]
    session["last_activity"] = datetime.now()
    
    return SessionInfo(**session)


@router.delete("/session/{session_id}")
async def end_session(session_id: str):
    """
    End a session and clean up resources.
    """
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )
    
    del active_sessions[session_id]
    logger.info(f"Session ended: {session_id}")
    
    return {"message": f"Session '{session_id}' ended successfully"}


@router.get("/sessions/active")
async def list_active_sessions():
    """
    List all active sessions (admin endpoint).
    """
    return {
        "active_sessions": len(active_sessions),
        "sessions": [
            {
                "session_id": s["session_id"],
                "dataset_id": s["dataset_id"],
                "created_at": s["created_at"].isoformat(),
                "last_activity": s["last_activity"].isoformat()
            }
            for s in active_sessions.values()
        ]
    }
