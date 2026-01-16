# Backend Files Location Map

## Actual File Paths

All backend files are located in the `ml_backend/` directory at the project root.

### API Routes
- **File**: `../../ml_backend/api/routes/monitor.py`
- **Purpose**: Main monitoring API endpoints
- **Lines**: ~790 lines
- **Key Functions**:
  - `get_available_intents()` - Returns monitoring options
  - `submit_monitoring_request()` - Creates monitoring job
  - `get_job_status()` - Returns job progress
  - `get_job_results()` - Returns completed results
  - `analyze_finding()` - Deep-dive AI analysis
  - `process_monitoring_job()` - Background processing

### Services
- **File**: `../../ml_backend/services/groq_service.py`
- **Purpose**: Groq AI integration
- **Lines**: ~120 lines
- **Key Classes/Functions**:
  - `GroqService` class
  - `analyze_monitoring_data()` - Overall analysis
  - `analyze_finding()` - Per-finding analysis

### Policy Engine Files
- **intent_resolver.py**: `../../ml_backend/policy/intent_resolver.py`
  - Maps user intents to analysis strategies
  
- **dataset_orchestrator.py**: `../../ml_backend/policy/dataset_orchestrator.py`
  - Fetches and prepares data
  
- **strategy_selector.py**: `../../ml_backend/policy/strategy_selector.py`
  - Recommends analysis strategies
  
- **analysis_output.py**: `../../ml_backend/policy/analysis_output.py`
  - Generates audit-friendly output

### Configuration
- **config.py**: `../../ml_backend/config.py`
  - Application settings and configuration
  
- **.env**: `../../ml_backend/.env`
  - Environment variables (GROQ_API_KEY required)
  
- **main.py**: `../../ml_backend/main.py`
  - FastAPI application entry point

## Dependencies
- **requirements.txt**: `../../ml_backend/requirements.txt`
  - Python package dependencies

## Quick Navigation Commands

```bash
# View main API file
cat ../../ml_backend/api/routes/monitor.py

# View Groq service
cat ../../ml_backend/services/groq_service.py

# Edit environment variables
nano ../../ml_backend/.env

# Start backend server
cd ../../ml_backend && python -m uvicorn main:app --reload
```

## File Change History

### Last Modified
- `monitor.py`: Added dynamic fallback actions, improved logging
- `groq_service.py`: Added `analyze_finding()` method for deep-dive analysis
- `.env`: Added GROQ_API_KEY configuration

### Recent Changes
1. Enhanced context logging in Groq API calls
2. Dynamic action generation based on intent and risk level
3. Improved error handling and traceback logging
4. Added detailed analysis endpoint for individual findings
