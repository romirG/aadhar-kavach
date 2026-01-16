# UIDAI ML Backend - API Reference

## Overview

FastAPI-based backend for UIDAI fraud detection system with async-ready endpoints,
modular services, and ML inference integration.

## Base URL
```
http://localhost:8000/api/ml
```

## Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/datasets` | GET | List available datasets |
| `/datasets/{id}` | GET | Get dataset details |
| `/select-dataset` | POST | Select and configure dataset |
| `/datasets/{id}/analyze` | POST | Start analysis job |
| `/analysis/{job_id}/status` | GET | Check job status |
| `/analysis/{job_id}/results` | GET | Get analysis results |
| `/analysis/{job_id}/summary` | GET | Get auditor summary |
| `/visualizations/{job_id}` | GET | Get all visualizations |

---

## 1. List Datasets

**GET** `/api/ml/datasets`

List all available UIDAI datasets.

### Request
```bash
curl -X GET "http://localhost:8000/api/ml/datasets"
```

### Response
```json
{
  "datasets": [
    {
      "id": "enrolment",
      "name": "Aadhaar Monthly Enrolment Data",
      "description": "State-wise Aadhaar enrolment and update status",
      "fields": ["state", "district", "month", "year", "total_enrolments"]
    },
    {
      "id": "demographic",
      "name": "Aadhaar Demographic Monthly Update Data",
      "description": "Demographic update requests across India",
      "fields": ["state", "district", "month", "year", "total_updates"]
    },
    {
      "id": "biometric",
      "name": "Aadhaar Biometric Monthly Update Data",
      "description": "Biometric update requests across India",
      "fields": ["state", "district", "month", "year", "total_updates"]
    }
  ],
  "total": 3
}
```

---

## 2. Select Dataset

**POST** `/api/ml/select-dataset`

Select and configure a dataset for analysis.

### Request
```bash
curl -X POST "http://localhost:8000/api/ml/select-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "enrolment",
    "record_limit": 1000,
    "state_filter": "Maharashtra"
  }'
```

### Response
```json
{
  "session_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
  "dataset_id": "enrolment",
  "dataset_name": "Aadhaar Monthly Enrolment Data",
  "record_limit": 1000,
  "filters_applied": {
    "state": "Maharashtra"
  },
  "estimated_records": 1000,
  "selected_at": "2026-01-15T00:10:00",
  "message": "Dataset 'Aadhaar Monthly Enrolment Data' selected."
}
```

---

## 3. Start Analysis

**POST** `/api/ml/datasets/{dataset_id}/analyze`

Trigger ML analysis pipeline for selected dataset.

### Request
```bash
curl -X POST "http://localhost:8000/api/ml/datasets/enrolment/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 1000,
    "models": ["isolation_forest", "hdbscan", "autoencoder"]
  }'
```

### Response
```json
{
  "job_id": "xyz12345-6789-abcd-ef01-234567890abc",
  "dataset_id": "enrolment",
  "status": "pending",
  "created_at": "2026-01-15T00:15:00",
  "message": "Analysis started. Use job_id to check status."
}
```

---

## 4. Check Analysis Status

**GET** `/api/ml/analysis/{job_id}/status`

Check the progress of an analysis job.

### Request
```bash
curl -X GET "http://localhost:8000/api/ml/analysis/xyz12345-6789-abcd-ef01-234567890abc/status"
```

### Response
```json
{
  "job_id": "xyz12345-6789-abcd-ef01-234567890abc",
  "status": "processing",
  "progress": 75.0,
  "message": "Running ensemble ML models...",
  "started_at": "2026-01-15T00:15:05"
}
```

---

## 5. Get Results

**GET** `/api/ml/analysis/{job_id}/results`

Get complete analysis results.

### Request
```bash
curl -X GET "http://localhost:8000/api/ml/analysis/xyz12345-6789-abcd-ef01-234567890abc/results"
```

### Response
```json
{
  "job_id": "xyz12345-6789-abcd-ef01-234567890abc",
  "dataset_id": "enrolment",
  "status": "completed",
  "total_records": 1000,
  "anomaly_count": 47,
  "anomaly_percentage": 4.7,
  "models_used": ["isolation_forest", "hdbscan", "autoencoder"],
  "model_results": [
    {
      "model_name": "isolation_forest",
      "anomaly_count": 52,
      "threshold": 0.5,
      "execution_time_ms": 245.3
    },
    {
      "model_name": "hdbscan",
      "anomaly_count": 38,
      "threshold": 0.5,
      "execution_time_ms": 189.7
    },
    {
      "model_name": "autoencoder",
      "anomaly_count": 45,
      "threshold": 0.5,
      "execution_time_ms": 1523.8
    }
  ],
  "anomalies": [
    {
      "record_id": "rec_001",
      "anomaly_score": 0.92,
      "risk_level": "High",
      "confidence": 0.95,
      "reasons": [
        "Unusually high update frequency from same operator",
        "Geo-inconsistent updates within short timeframe",
        "Weekend activity detected (unusual for government ops)"
      ],
      "features": {
        "total_enrolments": 458,
        "total_updates": 312,
        "state_event_count": 2100
      }
    }
  ],
  "execution_time_ms": 4532.5,
  "created_at": "2026-01-15T00:15:00",
  "completed_at": "2026-01-15T00:15:05"
}
```

---

## 6. Get Visualizations

**GET** `/api/ml/visualizations/{job_id}`

Get all generated visualizations.

### Request
```bash
curl -X GET "http://localhost:8000/api/ml/visualizations/xyz12345-6789-abcd-ef01-234567890abc"
```

### Response
```json
{
  "job_id": "xyz12345-6789-abcd-ef01-234567890abc",
  "charts": [
    {
      "chart_type": "time_series",
      "title": "Event Volume Over Time",
      "data": {
        "dates": ["2024-01", "2024-02", "..."],
        "series": {"total_enrolments": [100, 120, "..."]},
        "anomaly_scores": [0.1, 0.85, "..."]
      },
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    },
    {
      "chart_type": "distribution",
      "title": "Anomaly Score Distribution",
      "data": {
        "statistics": {"mean": 0.32, "std": 0.21, "high_risk_count": 15}
      },
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    },
    {
      "chart_type": "geo_heatmap",
      "title": "Geographic Anomaly Distribution",
      "data": {
        "high_risk_states": ["Maharashtra", "Gujarat"],
        "state_stats": [{"state": "Maharashtra", "mean_score": 0.68}]
      },
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    },
    {
      "chart_type": "cluster",
      "title": "HDBSCAN Cluster Analysis",
      "data": {
        "n_clusters": 5,
        "n_outliers": 47,
        "cluster_stats": {}
      },
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    },
    {
      "chart_type": "risk_dashboard",
      "title": "Risk Dashboard",
      "data": {
        "risk_distribution": {"High": 15, "Medium": 25, "Low": 7},
        "overall_risk_score": 0.47
      },
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ],
  "generated_at": "2026-01-15T00:15:10"
}
```

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "detail": {
    "error": "Error type",
    "message": "Human-readable error message",
    "available": ["list", "of", "valid", "options"]
  }
}
```

### Common HTTP Status Codes
- `200` - Success
- `202` - Accepted (processing)
- `400` - Bad request
- `404` - Not found
- `500` - Server error

---

## Rate Limits

- Analysis jobs: 10 per minute
- Visualization requests: 30 per minute
- Dataset queries: 100 per minute

## Authentication

Currently open for development. Production will use API keys:
```bash
curl -H "X-API-Key: your-api-key" ...
```
