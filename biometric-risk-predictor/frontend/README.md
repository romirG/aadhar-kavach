# Biometric Risk Predictor - Frontend

This folder contains the web UI for the Biometric Re-enrollment Risk Predictor feature.

## 📄 Files

- `index.html` - Main dashboard page (self-contained HTML/CSS/JS)
- `assets/` - Static assets (images, icons, fonts)

## 🎨 Features

### Dashboard Components
1. **Analysis Parameters** - State, Age Range, Records, Threshold filters
2. **KPI Cards** - Records Analyzed, High Risk Cases, Avg Risk Score, Model Accuracy
3. **Metrics Row** - Regions Coverage, Risk Distribution, ML Precision
4. **Quick Stats** - Regions, Recall, F1 Score, AUC-ROC, Critical Cases

### Interactive Charts (Chart.js)
1. State Risk Distribution (horizontal bar)
2. Risk Category Distribution (doughnut)
3. Age Bucket Risk Analysis (grouped bar)
4. Monthly Trend Analysis (line)
5. Survival Curves (line)
6. SHAP Feature Importance (horizontal bar)
7. Centre Performance (bar)
8. ROC Curve (line)
9. Precision-Recall Curve (line)
10. Calibration Chart (line)
11. Occupation Risk (bar)
12. Demographic Disparity (bar)
13. Outreach Funnel (bar)
14. Threshold Sensitivity (line)

### AI Features
- Deep Analysis with Groq LLaMA 3.3
- Policy Framework Recommendations

## 🎯 Styling

- **Theme**: Professional white/light grey
- **Colors**: 
  - Primary: `#4f46e5` (Indigo)
  - Success: `#10b981` (Green)
  - Warning: `#f59e0b` (Amber)
  - Danger: `#dc2626` (Red)
- **Font**: System UI stack

## 🚀 Running

### Via Express Server
```bash
cd ../../server
npm start
# Access at http://localhost:3001/risk_analysis.html
```

### Direct File Access
Simply open `index.html` in a web browser.
Note: API calls require the ML backend running on port 8000.

## 🔗 API Connection

The frontend connects to the ML backend at:
```javascript
const ML_API = 'http://localhost:8000';
```

Required backend endpoints:
- `POST /api/risk/analyze` - Run analysis
- `GET /api/risk/states` - Get state list
- `POST /api/risk/ai/deep-analysis` - AI analysis
- `POST /api/risk/ai/policy-framework` - Policy recommendations
