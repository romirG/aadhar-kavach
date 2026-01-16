# 🚀 Quick Start Guide - Operations Monitoring

## Start All Servers (3 terminals)

### Terminal 1: ML Backend (Port 8000)
```bash
cd ml_backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2: Express Server (Port 3001)
```bash
cd server
npm start
```

### Terminal 3: React Frontend (Port 8080)
```bash
npm run dev
```

## Access Application
🌐 http://localhost:8080/ → Click "Operations Monitoring"

---

## Key Files Modified

### Backend
- `ml_backend/api/routes/monitor.py` (line 589-656: dynamic actions)
- `ml_backend/services/groq_service.py` (added analyze_finding())

### Frontend  
- `src/pages/Monitoring.tsx` (added AI analysis button)
- `src/components/AIAnalysisDialog.tsx` (NEW FILE)
- `src/services/api.ts` (added analyzeFinding())

---

## Documentation

📁 **Main Docs**: `operations_monitoring/README.md`  
📋 **All Files**: `operations_monitoring/FILE_REFERENCE.md`  
🔌 **API Docs**: `operations_monitoring/docs/API_DOCUMENTATION.md`

---

## Troubleshooting

### "Different actions not showing"
✅ **Fixed**: Dynamic context-aware actions now generated  
✅ **Check**: Backend logs show Groq API calls  
✅ **Verify**: GROQ_API_KEY in `ml_backend/.env`

### "ML Backend Unavailable"
```bash
cd ml_backend
python -m uvicorn main:app --reload
```

### "AI Analysis fails"
- Check Groq API key is valid
- Review backend terminal for errors
- Verify job has completed

---

## Test Commands

```bash
# Check health
curl http://localhost:3001/api/monitor/health

# Submit request
curl -X POST http://localhost:3001/api/monitor \
  -H "Content-Type: application/json" \
  -d '{"intent":"check_enrollments","time_period":"today","vigilance":"standard"}'
```

---

**Need Help?** Check `operations_monitoring/README.md`
