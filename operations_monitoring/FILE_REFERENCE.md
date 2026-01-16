# Operations Monitoring - Complete File List

## Backend Files (Python/FastAPI)

### Core API Files
```
ml_backend/api/routes/monitor.py
└── Main monitoring API with all endpoints
    - Job management
    - Status tracking
    - Results retrieval
    - AI analysis integration
```

### Services
```
ml_backend/services/groq_service.py
└── Groq AI integration service
    - analyze_monitoring_data() method
    - analyze_finding() method for deep-dive
```

### Policy Engine
```
ml_backend/policy/
├── intent_resolver.py          # Maps user intents to ML strategies
├── dataset_orchestrator.py     # Fetches and prepares data
├── strategy_selector.py        # Selects analysis strategies
└── analysis_output.py          # Generates audit-friendly output
```

### Configuration
```
ml_backend/config.py           # Settings and configuration
ml_backend/.env               # Environment variables (GROQ_API_KEY)
ml_backend/requirements.txt   # Python dependencies
ml_backend/main.py           # FastAPI application entry point
```

## Frontend Files (React/TypeScript)

### Pages
```
src/pages/Monitoring.tsx
└── Main operations monitoring interface
    - Monitoring request form
    - Results display
    - AI analysis integration
```

### Components
```
src/components/AIAnalysisDialog.tsx
└── Modal dialog for AI analysis display
    - Detailed analysis section
    - Root cause display
    - Impact assessment
    - Recommended actions list
    - Monitoring plan
```

### Services
```
src/services/api.ts
└── API client and type definitions
    - getMonitoringIntents()
    - submitMonitoringRequest()
    - getMonitoringStatus()
    - getMonitoringResults()
    - analyzeFinding()
    
    Types:
    - MonitoringIntent
    - MonitoringRequest
    - MonitoringResults
    - FindingAnalysisResponse
    - ImpactAssessment
    - DetailedActionItem
```

### UI Components (Shadcn/ui)
```
src/components/ui/
├── dialog.tsx          # Dialog component
├── button.tsx          # Button component
├── card.tsx           # Card components
├── badge.tsx          # Badge component
├── select.tsx         # Select dropdown
└── alert.tsx          # Alert component
```

## Middleware (Express/Node.js)

### Server
```
server/index.js
└── Express server with proxy
    - Serves frontend static files
    - Proxies /api/monitor/* to ML backend
    - Port 3001
```

### Configuration
```
server/package.json    # Node dependencies
server/.env           # Server configuration
```

## Documentation
```
operations_monitoring/
├── README.md          # Complete documentation
└── FILE_REFERENCE.md  # This file
```

## Quick File Copy Reference

If you need to copy files to a new location, here are the essential files:

### Backend (Must Copy)
1. `ml_backend/api/routes/monitor.py` (790 lines)
2. `ml_backend/services/groq_service.py` (120 lines)
3. `ml_backend/policy/intent_resolver.py`
4. `ml_backend/policy/dataset_orchestrator.py`
5. `ml_backend/policy/strategy_selector.py`
6. `ml_backend/policy/analysis_output.py`
7. `ml_backend/config.py`
8. `ml_backend/.env`
9. `ml_backend/main.py`

### Frontend (Must Copy)
1. `src/pages/Monitoring.tsx` (440 lines)
2. `src/components/AIAnalysisDialog.tsx` (250 lines)
3. `src/services/api.ts` (368 lines)
4. All files in `src/components/ui/` (Shadcn components)

### Middleware (Must Copy)
1. `server/index.js` (128 lines)
2. `server/package.json`

## File Sizes (Approximate)

| File | Lines of Code | Size |
|------|--------------|------|
| monitor.py | 790 | ~35 KB |
| groq_service.py | 120 | ~5 KB |
| Monitoring.tsx | 440 | ~18 KB |
| AIAnalysisDialog.tsx | 250 | ~10 KB |
| api.ts | 368 | ~13 KB |
| index.js | 128 | ~5 KB |

## Dependencies

### Python (requirements.txt)
```
fastapi>=0.109.2
uvicorn[standard]>=0.27.1
pydantic-settings
pandas
numpy
scikit-learn
requests
pyyaml
groq
```

### Node.js (package.json)
```json
{
  "express": "^4.18.2",
  "axios": "^1.13.2",
  "cors": "^2.8.5",
  "dotenv": "^16.3.1"
}
```

### Frontend (React)
```json
{
  "react": "^18.x",
  "react-dom": "^18.x",
  "@radix-ui/react-dialog": "^1.1.14",
  "lucide-react": "latest",
  "tailwindcss": "latest"
}
```

## Git Structure
```
.
├── ml_backend/
│   ├── api/routes/monitor.py
│   ├── services/groq_service.py
│   ├── policy/
│   ├── config.py
│   └── .env
├── src/
│   ├── pages/Monitoring.tsx
│   ├── components/AIAnalysisDialog.tsx
│   └── services/api.ts
├── server/
│   └── index.js
└── operations_monitoring/
    ├── README.md
    └── FILE_REFERENCE.md
```

## Key Integration Points

### 1. Backend → Groq API
```python
# ml_backend/services/groq_service.py
groq_service.analyze_monitoring_data(context, flagged_records)
groq_service.analyze_finding(finding, flagged_records, context)
```

### 2. Frontend → Backend
```typescript
// src/services/api.ts
const ML_API_BASE = 'http://localhost:8000';
fetch(`${ML_API_BASE}/api/monitor/...`)
```

### 3. Express → ML Backend
```javascript
// server/index.js
const ML_BACKEND_URL = 'http://localhost:8000';
axios.post(`${ML_BACKEND_URL}/api/monitor/...`)
```

## Environment Variables Required

### ml_backend/.env
```env
GROQ_API_KEY=your_groq_api_key_here
DATA_GOV_API_KEY=your_data_gov_key
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

### server/.env (optional)
```env
PORT=3001
ML_BACKEND_URL=http://localhost:8000
```

## Port Configuration

| Service | Port | Protocol |
|---------|------|----------|
| React Frontend | 8080 | HTTP |
| Express Server | 3001 | HTTP |
| ML Backend | 8000 | HTTP |

---

**Note**: This module is self-contained and can work independently if all three servers are running.
