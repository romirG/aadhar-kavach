# Aadhar Kavach — UIDAI Analytics Intelligence Platform

A full-stack analytics and ML platform providing real-time insights into Aadhaar enrollment patterns, biometric health, gender inclusion gaps, and operational risk monitoring across India — powered by live government APIs and production-grade ML models.

---

## 📌 Overview

**Aadhar Kavach** is a multi-module intelligence dashboard that ingests live data from [data.gov.in](https://data.gov.in) public APIs and runs ML models to surface actionable insights for UIDAI administrators. It covers:

- **Enrollment Forecasting** — ARIMA time-series models predicting district-level enrollment trends
- **Biometric Risk Prediction** — XGBoost/Random Forest models identifying centers and states at high risk of biometric re-enrollment failure
- **Gender Inclusion Tracker** — LightGBM models flagging districts with low female enrollment coverage
- **Geospatial Penetration Map** — Interactive Leaflet.js map showing Aadhaar saturation by region
- **Operations Monitoring** — Real-time AI-powered (Groq LLaMA) anomaly detection and operational alerts
- **Vulnerable Population Tracker** — Analytics focused on under-served age groups and demographics

All data is **aggregated at state/district level** — no individual Aadhaar data is processed.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│              React + Vite Frontend (Port 5173)          │
│         (Tailwind CSS · shadcn/ui · Recharts)           │
└─────────────────────┬──────────────────────────────────┘
                      │ HTTP
┌─────────────────────▼──────────────────────────────────┐
│         Express.js API Gateway (Port 3001)              │
│   Routes: enrolment · demographic · biometric · AI      │
│   Proxy:  /api/forecast → ML Backend                   │
│           /api/monitor  → ML Backend                   │
└───────────┬───────────────────────┬────────────────────┘
            │                       │
┌───────────▼───────┐   ┌───────────▼──────────────────┐
│  Python Flask      │   │  Python FastAPI ML Backend   │
│  Analytics Server  │   │        (Port 8000)           │
│  (python_backend/) │   │  ARIMA · XGBoost · LightGBM  │
│                    │   │  SHAP · Groq LLaMA · HDBSCAN │
└────────────────────┘   └──────────────────────────────┘
            │                       │
            └───────────┬───────────┘
                        ▼
              data.gov.in Public APIs
              (Aadhaar Enrollment, Biometric,
               Demographic datasets)
```

---

## 🧩 Modules

| Module | Location | Description |
|--------|----------|-------------|
| **React Dashboard** | `src/` | Main frontend — charts, maps, dashboards (Vite + React + shadcn/ui) |
| **Express Server** | `server/` | API gateway, data.gov.in integration, static file serving |
| **ML Backend** | `ml_backend/` | FastAPI service — ARIMA forecasting, anomaly detection, explainability |
| **Python Analytics** | `python_backend/` | Flask server for demographic analytics and gender inclusion |
| **Biometric Risk Predictor** | `biometric-risk-predictor/` | Self-contained ML module (XGBoost/RF) for biometric re-enrollment risk |
| **Geospatial Map** | `geospatial-penetration-map/` | Leaflet.js penetration map (served as static files) |
| **Operations Monitoring** | `operations_monitoring/` | Real-time Groq AI anomaly detection backend |

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** v18+ and npm
- **Python** 3.10+
- API keys (see [Environment Setup](#-environment-setup))

---

### 1. Frontend + Express Gateway

```bash
# Install root frontend dependencies
npm install

# Start Vite dev server (port 5173)
npm run dev

# In a separate terminal — start Express gateway (port 3001)
cd server
npm install
node index.js
```

---

### 2. ML Backend (FastAPI)

```bash
cd ml_backend

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

# Start FastAPI server (port 8000)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

---

### 3. Python Analytics Server (Flask)

```bash
cd python_backend
pip install -r requirements.txt

# Main analytics server
python flask_server.py
```

---

### 4. Biometric Risk Predictor (standalone)

```bash
cd biometric-risk-predictor/backend
pip install -r requirements.txt
uvicorn api.risk_predictor:router --host 0.0.0.0 --port 8001
```

The frontend is served automatically by the Express server at `/biometric/`.

---

## 🔑 Environment Setup

Copy the example files and fill in your keys:

```bash
cp server/.env.example server/.env
cp ml_backend/.env.example ml_backend/.env
```

### `server/.env`

```env
DATA_GOV_API_KEY=your_data_gov_in_api_key
GEMINI_API_KEY_1=your_gemini_key
# Add GEMINI_API_KEY_2 through _5 for rotation under rate limits
PORT=3001
```

### `ml_backend/.env`

```env
DATA_GOV_API_KEY=your_data_gov_in_api_key
GROQ_API_KEY=your_groq_api_key
HOST=0.0.0.0
PORT=8000
DEBUG=true
DEFAULT_RISK_THRESHOLD=0.85
RANDOM_SEED=42
ENROLMENT_RESOURCE_ID=ecd49b12-3084-4521-8f7e-ca8bf72069ba
DEMOGRAPHIC_RESOURCE_ID=19eac040-0b94-49fa-b239-4f2fd8677d53
BIOMETRIC_RESOURCE_ID=65454dab-1517-40a3-ac1d-47d4dfe6891c
```

Get your keys at:
- **data.gov.in API key**: [data.gov.in](https://data.gov.in/user/register)
- **Groq API key**: [console.groq.com](https://console.groq.com/keys)
- **Google Gemini API key**: [aistudio.google.com](https://aistudio.google.com/apikey)

---

## 📡 Key API Endpoints

### Express Gateway (`:3001`)

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `GET /api/enrolment/*` | Aadhaar enrollment data |
| `GET /api/demographic/*` | Demographic breakdown |
| `GET /api/biometric/*` | Biometric statistics |
| `GET /api/dashboard/*` | Aggregated dashboard metrics |
| `GET /api/hotspots/*` | Enrollment hotspot analysis |
| `GET /api/ai/*` | Gemini AI-powered insights |
| `GET /api/geo-penetration/*` | Geospatial coverage data |
| `POST /api/forecast/*` | → Proxied to ML Backend ARIMA |
| `GET /api/monitor/*` | → Proxied to ML Backend monitoring |

### ML Backend (`:8000`)

| Endpoint | Description |
|----------|-------------|
| `GET /docs` | Interactive Swagger UI |
| `POST /api/forecast/train` | Train ARIMA enrollment models |
| `GET /api/forecast/predict/{district}` | District-level enrollment forecast |
| `GET /api/forecast/districts` | List of trained districts |
| `POST /api/risk/analyze` | Biometric re-enrollment risk analysis |
| `GET /api/risk/high-risk-regions` | Flagged high-risk states/districts |
| `GET /api/risk/recommendations` | AI-generated action items |
| `GET /api/risk/shap-explanation` | SHAP model explainability values |
| `GET /api/datasets/` | Available data.gov.in datasets |
| `POST /api/analyze/` | Run EDA pipeline |
| `POST /api/train/` | Train LightGBM gender-gap model |
| `POST /api/predict/` | Predict high-risk districts |

---

## 🐳 Docker & Deployment

### Run with Docker

The ML backend uses a root-level Dockerfile (`Dockerfile.mlbackend`) so the build context includes `operations_monitoring/` and `biometric-risk-predictor/` alongside the core `ml_backend/`.

```bash
# ML Backend — build from project root
docker build -f Dockerfile.mlbackend -t uidai-ml-backend .
docker run -p 8000:8000 --env-file ml_backend/.env uidai-ml-backend

# Express Server
cd server
docker build -t uidai-express .
docker run -p 3001:3001 --env-file .env uidai-express
```

### Deploy on Render

A `render.yaml` blueprint is included. The ML backend service uses `runtime: docker` with `Dockerfile.mlbackend`. Set these env vars in the Render dashboard:
- `DATA_GOV_API_KEY`
- `GROQ_API_KEY`
- `GEMINI_API_KEY_1`

See [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md) for full instructions.

---

## 🛠️ Tech Stack

### Frontend
- **React 18** + **Vite** + **TypeScript**
- **Tailwind CSS** + **shadcn/ui** (Radix UI primitives)
- **Recharts** — data visualizations
- **Leaflet / react-leaflet** — geospatial maps
- **TanStack Query** — server state management

### Backend
- **Node.js / Express.js** — API gateway and static serving
- **Python / FastAPI** — ML inference API
- **Python / Flask** — analytics server
- **statsmodels ARIMA** — enrollment time-series forecasting
- **XGBoost / Random Forest** — biometric risk prediction
- **LightGBM + Optuna** — gender inclusion gap modeling
- **SHAP** — model explainability
- **Groq (LLaMA 3.3)** — AI recommendations
- **Google Gemini** — AI-powered dashboard insights
- **HDBSCAN** — anomaly detection clustering

### Infrastructure
- **Docker** — containerized services
- **Render** — cloud deployment (ML + Express)
- **Vercel** — optional frontend deployment

---

## 📊 Data Sources

All data fetched from [data.gov.in](https://data.gov.in) open government APIs:

| Dataset | Description |
|---------|-------------|
| Aadhaar Enrolment Statistics | State/district/age-group enrollment counts |
| Biometric Update Records | Re-enrollment and update transactions |
| Demographic Distribution | Gender and age breakdown of enrollees |
| Authentication Transactions | Auth success/failure rates by region |

No individual Aadhaar data is used. All analysis is aggregated at state/district/age-group level, complying with UIDAI data protection guidelines.

---

## 🔒 Privacy & Security

- **No PII**: Only aggregated, anonymized public government statistics
- **No individual Aadhaar numbers** are stored or processed
- **CORS** configured for production origins only
- **Non-root Docker users** for all containers

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.

---

*Built as part of the UIDAI Data Hackathon.*
