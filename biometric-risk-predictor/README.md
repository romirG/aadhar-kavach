# Biometric Re-enrollment Risk Predictor

A comprehensive ML-powered system for predicting biometric re-enrollment risk across Aadhaar authentication centers in India.

> **⚠️ This is a TRUE MODULE** - All biometric risk predictor code lives here. The main applications (Express server and FastAPI) import from this folder.

## 📁 Folder Structure

```
biometric-risk-predictor/
├── frontend/                    # Web UI Components
│   ├── index.html              # Main dashboard (self-contained HTML/CSS/JS)
│   └── assets/                 # Static assets (images, icons)
│
├── backend/                    # Python ML Backend
│   ├── api/                    # FastAPI route handlers
│   │   └── risk_predictor.py   # All API endpoints for risk analysis
│   │
│   ├── services/               # Business logic & ML services
│   │   ├── biometric_risk_service.py    # Core risk prediction service
│   │   ├── data_gov_client.py           # Government API client
│   │   ├── feature_engineering.py       # Feature extraction & engineering
│   │   └── groq_ai_service.py           # AI recommendations (Groq LLaMA)
│   │
│   ├── models/                 # ML model storage
│   │   └── (trained models saved here)
│   │
│   └── data/                   # Data processing utilities
│       └── (data files)
│
└── README.md                   # This file
```

## 🔗 Integration Points

### Frontend (Express Server)
The Express server at `server/index.js` serves the frontend:
```javascript
// Serves from /biometric/ path
app.use('/biometric', express.static('../biometric-risk-predictor/frontend'));
```
**Access URL**: `http://localhost:3001/biometric/`

### Backend (FastAPI)
The ML backend at `ml_backend/main.py` imports the router:
```python
# Imports from biometric-risk-predictor/backend/
from api.risk_predictor import router as risk_router
app.include_router(risk_router)
```

## 🚀 Features

### Frontend Dashboard
- **KPI Cards**: Real-time metrics with mini-charts
- **State Risk Distribution**: Horizontal bar chart with 15 states
- **Risk Category Breakdown**: Doughnut chart (Critical/High/Medium/Low)
- **Age Bucket Analysis**: Risk scores by age groups
- **Monthly Trend Analysis**: Time series risk trends
- **Survival Curves**: Template aging analysis
- **SHAP Feature Importance**: ML model explainability
- **Centre Performance**: Per-centre quality metrics
- **ROC/PR Curves**: Model performance visualization
- **AI Recommendations**: Groq LLaMA-powered insights

### Backend API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/risk/analyze` | POST | Run full risk analysis |
| `/api/risk/health` | GET | Service health check |
| `/api/risk/states` | GET | Get available states |
| `/api/risk/high-risk-regions` | GET | Get flagged regions |
| `/api/risk/recommendations` | GET | Get action items |
| `/api/risk/visualizations` | GET | Get chart data |
| `/api/risk/model-info` | GET | ML model details |
| `/api/risk/shap-explanation` | GET | SHAP values |
| `/api/risk/age-bucket-analysis` | GET | Age-wise risk |
| `/api/risk/threshold-sensitivity` | GET | Threshold impact |

## 🛠️ Technology Stack

### Frontend
- **HTML5/CSS3**: Responsive design with white theme
- **JavaScript (ES6+)**: Vanilla JS for interactivity
- **Chart.js**: Interactive data visualizations
- **No build required**: Self-contained HTML file

### Backend
- **Python 3.9+**: Core language
- **FastAPI**: High-performance API framework
- **XGBoost/Random Forest**: ML models for risk prediction
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: ML utilities
- **SHAP**: Model explainability
- **Groq API**: LLaMA 3.3 for AI recommendations

## 📦 Installation & Running

### Frontend
The frontend is served as a static HTML file. Simply access:
```
http://localhost:3001/risk_analysis.html
```
Or open `frontend/index.html` directly in a browser.

### Backend
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn api.risk_predictor:router --host 0.0.0.0 --port 8000
```

Note: The backend is integrated into the main `ml_backend` folder at the project root. These files are organized copies for reference.

## 🔒 Privacy Safeguards

- Uses **only aggregated data** from public government APIs
- **No individual Aadhaar numbers** are processed
- All analysis at **state/district/age-group level**
- Complies with UIDAI data protection guidelines

## 📊 Data Sources

All data is fetched from [data.gov.in](https://data.gov.in) public APIs:
- Aadhaar Enrolment Statistics
- Biometric Update Records
- Demographic Distribution
- Authentication Transaction Logs

## 🎯 Use Cases

1. **Proactive Outreach**: Identify residents needing biometric re-capture
2. **Resource Planning**: Allocate mobile units to high-risk areas
3. **Centre Monitoring**: Track capture quality across centers
4. **Policy Insights**: Data-driven recommendations for UIDAI

## 📝 License

Internal UIDAI project - Not for public distribution.
