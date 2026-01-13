"""
Dataset routes - List and select datasets for analysis.
"""
import logging
from typing import List
from fastapi import APIRouter, HTTPException

import sys
sys.path.insert(0, '..')

from config import DATASETS
from api.schemas import DatasetInfo, DatasetListResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def get_available_datasets() -> List[DatasetInfo]:
    """Get list of available datasets."""
    return [
        DatasetInfo(
            id=dataset["id"],
            name=dataset["name"],
            description=dataset["description"],
            fields=dataset["fields"]
        )
        for dataset in DATASETS.values()
    ]


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets():
    """
    List all available datasets for analysis.
    
    Returns information about the 3 data.gov.in APIs:
    - Aadhaar Monthly Enrolment Data
    - Aadhaar Demographic Monthly Update Data
    - Aadhaar Biometric Monthly Update Data
    """
    datasets = get_available_datasets()
    logger.info(f"Listing {len(datasets)} available datasets")
    return DatasetListResponse(datasets=datasets, total=len(datasets))


@router.get("/datasets/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str):
    """
    Get detailed information about a specific dataset.
    
    Args:
        dataset_id: One of 'enrolment', 'demographic', or 'biometric'
    """
    if dataset_id not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found. Available: {list(DATASETS.keys())}"
        )
    
    dataset = DATASETS[dataset_id]
    logger.info(f"Retrieved dataset info for: {dataset_id}")
    
    return DatasetInfo(
        id=dataset["id"],
        name=dataset["name"],
        description=dataset["description"],
        fields=dataset["fields"]
    )
