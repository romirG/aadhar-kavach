# Biometric Risk Predictor - Backend

Python ML backend for the Biometric Re-enrollment Risk Predictor.

## 📁 Structure

```
backend/
├── api/
│   └── risk_predictor.py       # FastAPI router with all endpoints
│
├── services/
│   ├── biometric_risk_service.py   # Core ML risk prediction service
│   ├── data_gov_client.py          # data.gov.in API client
│   ├── feature_engineering.py      # Feature extraction & engineering
│   └── groq_ai_service.py          # Groq LLaMA AI recommendations
│
├── models/                         # Trained ML models (XGBoost, RF)
│
├── data/                          # Data files & cache
│
└── README.md                      # This file
```

## 🔧 Dependencies

```
fastapi>=0.104.0
uvicorn>=0.24.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
httpx>=0.25.0
python-dotenv>=1.0.0
groq>=0.4.0
shap>=0.43.0
```

## 🚀 Running

### Standalone
```bash
pip install -r requirements.txt
uvicorn api.risk_predictor:router --host 0.0.0.0 --port 8000 --reload
```

### As part of main ML backend
The files are integrated into the main `ml_backend` at project root:
```bash
cd ../../ml_backend
python main.py
```

## 📡 API Endpoints

### Core Analysis
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/risk/health` | GET | Health check |
| `/api/risk/analyze` | POST | Run full analysis |
| `/api/risk/states` | GET | Available states |

### Results
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/risk/summary` | GET | Analysis summary |
| `/api/risk/high-risk-regions` | GET | Flagged regions |
| `/api/risk/recommendations` | GET | Action items |
| `/api/risk/visualizations` | GET | Chart data |

### Model Info
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/risk/model-info` | GET | Model details |
| `/api/risk/shap-explanation` | GET | SHAP values |

### Advanced Analytics
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/risk/age-bucket-analysis` | GET | Age-wise risk |
| `/api/risk/centre-performance` | GET | Centre metrics |
| `/api/risk/survival-curve` | GET | Template aging |
| `/api/risk/threshold-sensitivity` | GET | Threshold impact |

### AI Recommendations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/risk/ai/deep-analysis` | POST | AI insights |
| `/api/risk/ai/policy-framework` | POST | Policy recommendations |

## 🔐 Environment Variables

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
DATA_GOV_API_KEY=your_data_gov_api_key
```

## 📊 ML Models Used

1. **XGBoost Classifier** - Primary risk prediction model
2. **Random Forest** - Ensemble fallback
3. **SHAP TreeExplainer** - Model explainability

## 🔒 Privacy

- Only aggregated government data is used
- No individual Aadhaar numbers processed
- State/district/age-group level analysis only
