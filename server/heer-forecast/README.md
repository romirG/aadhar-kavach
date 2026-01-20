# Heer Enrollment Forecaster

ARIMA-based time-series enrollment forecasting feature, integrated from [heer-suidai](https://github.com/Heer-create-lgtm/heer-suidai) repository.

## Overview

This module provides district-level enrollment predictions using Moving Average with Trend analysis, a lightweight JavaScript alternative to Python's ARIMA models. It generates 6-month forecasts with confidence intervals.

## Features

- **District-Level Predictions**: Forecast enrollment for individual districts
- **Time-Series Analysis**: Moving average with trend detection
- **Confidence Intervals**: 90% confidence bounds for predictions
- **Interactive Visualization**: Chart.js-based forecast charts
- **Historical Statistics**: Mean, std dev, min/max enrollment data

## API Endpoints

### GET `/api/heer-forecast/districts`
Get list of districts with sufficient data for forecasting.

**Query Parameters:**
- `limit` (optional): Number of records to fetch (default: 500)

**Response:**
```json
{
  "success": true,
  "count": 85,
  "districts": ["AIZAWL", "AMRAVATI", ...]
}
```

### GET `/api/heer-forecast/predict/:district`
Generate enrollment forecast for a specific district.

**Path Parameters:**
- `district`: District name (URL encoded)

**Query Parameters:**
- `periods` (optional): Number of periods to forecast (1-24, default: 6)
- `limit` (optional): Historical data limit (default: 500)

**Response:**
```json
{
  "success": true,
  "district": "AIZAWL",
  "periods": 6,
  "confidence_level": 0.90,
  "model_type": "Moving Average with Trend",
  "historical_stats": {
    "data_points": 12,
    "mean": 45230,
    "std": 2134,
    "last_date": "2024-12-31",
    "min": 42100,
    "max": 48500
  },
  "forecasts": [
    {
      "period": 1,
      "date": "2025-01-31",
      "predicted_enrollment": 46100,
      "lower_bound": 43200,
      "upper_bound": 49000
    },
    ...
  ]
}
```

### GET `/api/heer-forecast/states`
Get state-level forecast summary.

**Query Parameters:**
- `limit` (optional): Data limit (default: 500)
- `top` (optional): Top N states to return (default: 10)

**Response:**
```json
{
  "success": true,
  "count": 10,
  "states": [
    {
      "state": "UTTAR PRADESH",
      "total_enrollment": 2345678,
      "districts_count": 75,
      "projected_6m": 2463000
    },
    ...
  ],
  "note": "Projections based on 5% growth rate over 6 months"
}
```

## Architecture

### Folder Structure
```
server/heer-forecast/
├── README.md                    # This file
├── routes/
│   └── heer-forecast.js        # Express routes for forecasting API
└── models/
    └── forecast.py              # Original Python ARIMA model (reference)
```

### Design Decisions

1. **JavaScript Implementation**: Instead of running Python ARIMA models, we use a lightweight Moving Average with Trend algorithm in JavaScript. This:
   - Eliminates Python dependency
   - Maintains existing Node.js/Express architecture
   - Provides reasonable forecasts for visualization
   - Avoids subprocess overhead

2. **Separate Folder**: The `heer-forecast` folder keeps all forecasting code isolated, preventing conflicts with existing features.

3. **Data Source**: Uses the same `getEnrolmentData()` from existing data API, ensuring consistency.

## Usage in Frontend

The forecaster is accessible from the main dashboard:

1. Click "Enrollment Forecast" feature card
2. Select a district from dropdown (85+ available)
3. Click "Generate Forecast" button
4. View:
   - Historical statistics
   - 6-month forecast chart with confidence intervals
   - Detailed prediction table

## Integration Notes

- **Non-breaking**: All existing features remain unaffected
- **API Prefix**: Uses `/api/heer-forecast` route prefix
- **Dependencies**: No new npm packages required
- **Chart.js**: Uses existing Chart.js library from index.html

## Original Source

Adapted from Heer's ARIMA-based enrollment forecaster:
- Repository: https://github.com/Heer-create-lgtm/heer-suidai
- Original implementation: Python FastAPI with statsmodels ARIMA
- Conversion: JavaScript with Moving Average + Trend algorithm

## Future Enhancements

To use full ARIMA capabilities:
1. Set up Python FastAPI microservice on port 8001
2. Copy original `forecast.py` model
3. Proxy requests from Node.js to Python backend
4. Install dependencies: `statsmodels`, `pandas`, `numpy`, `scipy`

## Testing

Test the API:
```bash
# Get districts
curl http://localhost:3001/api/heer-forecast/districts

# Get forecast for specific district
curl http://localhost:3001/api/heer-forecast/predict/AIZAWL?periods=6

# Get state summary
curl http://localhost:3001/api/heer-forecast/states?top=10
```

## Credits

- Original ARIMA implementation: Heer ([Heer-create-lgtm](https://github.com/Heer-create-lgtm))
- Integration: UIDAI Datathon Team
- Adapted for Node.js/Express architecture
