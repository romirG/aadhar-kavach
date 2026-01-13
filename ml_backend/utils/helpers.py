"""
Utility helper functions.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to 0-1 range."""
    min_val = scores.min()
    max_val = scores.max()
    
    if max_val - min_val > 0:
        return (scores - min_val) / (max_val - min_val)
    return np.zeros_like(scores)


def get_risk_label(score: float, high_threshold: float = 0.8, medium_threshold: float = 0.5) -> str:
    """Get risk label from score."""
    if score >= high_threshold:
        return "High"
    elif score >= medium_threshold:
        return "Medium"
    return "Low"


def format_number(value: float, precision: int = 2) -> str:
    """Format number for display."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{precision}f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.{precision}f}K"
    return f"{value:.{precision}f}"


def calculate_percentile(value: float, distribution: np.ndarray) -> float:
    """Calculate percentile of value in distribution."""
    return float(np.mean(distribution <= value) * 100)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    return a / b if b != 0 else default


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def validate_dataset_id(dataset_id: str, valid_ids: List[str]) -> bool:
    """Validate dataset ID."""
    return dataset_id in valid_ids


def create_error_response(error: Exception, context: str = "") -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "success": False,
        "error": str(error),
        "context": context,
        "timestamp": datetime.now().isoformat()
    }


def batch_process(items: List[Any], batch_size: int = 100):
    """Generator for batch processing."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed_ms: float = 0
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        logger.info(f"{self.name} completed in {self.elapsed_ms:.2f}ms")
