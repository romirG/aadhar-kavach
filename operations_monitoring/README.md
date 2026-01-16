# Operations Monitoring System - Complete Module

## 📁 File Structure

This folder contains all files required for the Operations Monitoring system with Groq AI integration.

### Backend Files (ML Backend - Python)

**Location**: `ml_backend/`

1. **API Route**: `ml_backend/api/routes/monitor.py`
   - Main monitoring API endpoints
   - Handles job processing and status tracking
   - Integrates Groq AI analysis
   - Endpoints:
     - `GET /api/monitor/intents` - Get available monitoring options
     - `POST /api/monitor` - Submit monitoring request
     - `GET /api/monitor/status/{job_id}` - Check job status
     - `GET /api/monitor/results/{job_id}` - Get results
     - `POST /api/monitor/analyze-finding` - Get AI analysis for specific finding

2. **Groq Service**: `ml_backend/services/groq_service.py`
   - Groq API integration
   - Two main methods:
     - `analyze_monitoring_data()` - Overall monitoring analysis
     - `analyze_finding()` - Deep-dive analysis for individual findings

3. **Policy Engine**: `ml_backend/policy/`
   - `intent_resolver.py` - Resolves user intents to ML strategies
   - `dataset_orchestrator.py` - Manages data fetching
   - `strategy_selector.py` - Selects analysis strategies
   - `analysis_output.py` - Generates audit-friendly output

4. **Configuration**: `ml_backend/.env`
   - Contains `GROQ_API_KEY` (required for AI analysis)
   - API keys and configuration settings

### Frontend Files (React/TypeScript)

**Location**: `src/`

1. **Page Component**: `src/pages/Monitoring.tsx`
   - Main operations monitoring interface
   - Handles monitoring requests and displays results
   - Integrates AI analysis dialog

2. **AI Dialog Component**: `src/components/AIAnalysisDialog.tsx`
   - Modal dialog for displaying AI analysis
   - Shows detailed analysis, root cause, impact, and recommendations

3. **API Service**: `src/services/api.ts`
   - API client functions
   - Type definitions for monitoring data
   - Functions:
     - `getMonitoringIntents()`
     - `submitMonitoringRequest()`
     - `getMonitoringStatus()`
     - `getMonitoringResults()`
     - `analyzeFinding()`

### Middleware (Express Server)

**Location**: `server/`

1. **Server**: `server/index.js`
   - Proxies frontend requests to ML backend
   - Route: `/api/monitor/*` → `http://localhost:8000/api/monitor/*`

## 🚀 How to Run

### Prerequisites
```bash
# Python 3.12+
# Node.js 18+
# npm
```

### 1. Start ML Backend (Port 8000)
```bash
cd ml_backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start Express Server (Port 3001)
```bash
cd server
npm start
```

### 3. Start React Frontend (Port 8080)
```bash
# From project root
npm run dev
```

### 4. Access Application
Open browser: http://localhost:8080/

Navigate to: **Operations Monitoring** in sidebar

## 🎯 Features

### 1. Intent-Based Monitoring
- **Check Enrollments** - Monitor enrollment operations
- **Review Updates** - Examine demographic updates
- **Verify Biometrics** - Check biometric submissions
- **Comprehensive Check** - Full system audit

### 2. Configurable Parameters
- **Focus Area**: State/region selection
- **Time Period**: Today, Last 7 days, This month
- **Vigilance Level**: Routine, Standard, Enhanced, Maximum

### 3. AI-Powered Analysis (Groq)
- **Overall Analysis**: Generated for every monitoring request
- **Per-Finding Analysis**: Click "AI Analysis" button on any finding
- **Output Includes**:
  - Executive summary
  - Key findings with severity
  - Root cause analysis
  - Impact assessment
  - Recommended actions with:
    - Specific actionable steps
    - Priority levels
    - Responsible parties
    - Timelines
  - Follow-up monitoring plan

### 4. Dynamic Recommendations
- Context-aware actions based on:
  - Monitoring intent
  - Geographic focus
  - Risk level
  - Time period
  - Vigilance setting

## 🔧 Configuration

### Groq API Key Setup

Edit `ml_backend/.env`:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

Get your API key from: https://console.groq.com/

### API Endpoints

**ML Backend (Port 8000)**:
- Base URL: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

**Express Proxy (Port 3001)**:
- Base URL: `http://localhost:3001`
- Monitoring: `http://localhost:3001/api/monitor/*`

**Frontend (Port 8080)**:
- Base URL: `http://localhost:8080`

## 📊 Data Flow

```
User Interface (React)
    ↓
    │ Submit Monitoring Request
    ↓
Express Server (Port 3001)
    ↓
    │ Proxy to ML Backend
    ↓
ML Backend (Port 8000)
    ↓
    ├─→ Intent Resolution
    ├─→ Data Orchestration
    ├─→ Analysis Strategy Selection
    ├─→ ML Analysis Execution
    ├─→ Risk Aggregation
    └─→ Groq AI Analysis ⭐
    ↓
Return Results
    ↓
Display in UI with AI Analysis Button
    ↓
Click "AI Analysis" on Finding
    ↓
Request Deep Analysis from Groq
    ↓
Show Detailed AI Analysis Modal
```

## 🧪 Testing

### Test Monitoring Request
```bash
curl -X POST http://localhost:3001/api/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "check_enrollments",
    "focus_area": "Maharashtra",
    "time_period": "today",
    "vigilance": "standard",
    "record_limit": 1000
  }'
```

### Check Job Status
```bash
curl http://localhost:3001/api/monitor/status/{job_id}
```

### Get Results
```bash
curl http://localhost:3001/api/monitor/results/{job_id}
```

### Analyze Specific Finding
```bash
curl -X POST http://localhost:3001/api/monitor/analyze-finding \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "abc123",
    "finding_index": 0
  }'
```

## 🐛 Troubleshooting

### Issue: Different recommended actions not showing

**Solution**: 
1. Check if Groq API key is configured in `ml_backend/.env`
2. Check backend logs for Groq API calls
3. Verify Groq API key is valid and has credits
4. If Groq fails, fallback actions are now dynamic based on context

### Issue: "ML Backend Unavailable" error

**Solution**: Ensure ML backend is running on port 8000
```bash
cd ml_backend
python -m uvicorn main:app --reload
```

### Issue: AI Analysis dialog shows error

**Solution**: 
1. Verify job has completed
2. Check finding index is valid
3. Review browser console for errors
4. Check backend logs for Groq API issues

## 📝 Logs Location

**Backend Logs**: Terminal running `uvicorn` (port 8000)
- Shows Groq API calls
- Analysis success/failure
- Job processing steps

**Express Logs**: Terminal running `node index.js` (port 3001)
- Shows proxy requests

**Frontend Logs**: Browser Developer Console (F12)
- API call responses
- UI state changes

## 🔒 Security Notes

1. **Never commit** `.env` files with real API keys
2. API keys should be stored in environment variables
3. Frontend makes requests through Express proxy (prevents CORS issues)
4. ML backend validates all inputs

## 📚 Additional Resources

- Groq API Docs: https://console.groq.com/docs/
- FastAPI Docs: https://fastapi.tiangolo.com/
- React Docs: https://react.dev/

## 🎓 Training Data

The system uses aggregated, anonymized data from:
- Public government APIs (data.gov.in)
- State/district level statistics only
- No individual Aadhaar numbers processed
- Privacy-compliant data handling

## 📧 Support

For issues or questions:
1. Check this README
2. Review backend logs
3. Test API endpoints directly
4. Verify all services are running

---

**Last Updated**: January 16, 2026
**Version**: 1.0.0
