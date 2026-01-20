import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cors from 'cors';
import axios from 'axios';
import path from 'path';
import { fileURLToPath } from 'url';

// Get __dirname equivalent for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import routes
import enrolmentRoutes from './routes/enrolment.js';
import demographicRoutes from './routes/demographic.js';
import biometricRoutes from './routes/biometric.js';
import dashboardRoutes from './routes/dashboard.js';
import aiRoutes from './routes/ai.js';
import hotspotsRoutes from './routes/hotspots.js';
import geoPenetrationRoutes from './routes/geo-penetration.js';
import geoDataRoutes from './routes/geo-data.js';
// Ridwan features
import ridwanGenderRoutes from './routes/ridwan-gender.js';
import ridwanVulnerableRoutes from './routes/ridwan-vulnerable.js';
import ridwanSimulateRoutes from './routes/ridwan-simulate.js';
// Heer enrollment forecaster
import heerForecastRoutes from './heer-forecast/routes/heer-forecast.js';

const app = express();
const PORT = process.env.PORT || 3001;

// Handle ML_BACKEND_URL - Render provides hostname only, local dev uses full URL
let ML_BACKEND_URL = process.env.ML_BACKEND_URL || 'http://localhost:8000';
// If Render provides just hostname (no protocol), add https://
if (ML_BACKEND_URL && !ML_BACKEND_URL.startsWith('http')) {
  ML_BACKEND_URL = `https://${ML_BACKEND_URL}`;
}

// Middleware - Configure CORS for production frontend
app.use(cors({
  origin: [
    'https://aadhar-kavach.vercel.app',    // Vercel frontend (if still used)
    'https://aadhar-kavach-9x96.onrender.com',  // Express server on Render (self)
    'http://localhost:3001',               // Local development
    'http://localhost:5173',               // Vite dev server
    'http://127.0.0.1:3001'
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json());

// Serve static files from public folder
app.use(express.static(path.join(__dirname, 'public')));

// Serve Biometric Risk Predictor frontend from its feature module
app.use('/biometric', express.static(path.join(__dirname, '..', 'biometric-risk-predictor', 'frontend')));

// Redirect risk_analysis.html to new location
app.get('/risk_analysis.html', (req, res) => {
  res.redirect('/biometric/');
});

// Routes
app.use('/api/enrolment', enrolmentRoutes);
app.use('/api/demographic', demographicRoutes);
app.use('/api/biometric', biometricRoutes);
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/ai', aiRoutes);
app.use('/api/hotspots', hotspotsRoutes);

// Test route to verify server is working
app.get('/api/test', (req, res) => {
  console.log('✅ Test route hit!');
  res.json({ status: 'ok', message: 'Server is responding' });
});

app.use('/api/geo-penetration', geoPenetrationRoutes);
app.use('/api/geo-data', geoDataRoutes);

// Ridwan features routes
app.use('/api/ridwan-gender', ridwanGenderRoutes);
app.use('/api/ridwan-vulnerable', ridwanVulnerableRoutes);
app.use('/api/ridwan-simulate', ridwanSimulateRoutes);

// Heer enrollment forecaster routes
app.use('/api/heer-forecast', heerForecastRoutes);

// Enrollment Forecast API Proxy - Forward to ML backend Python ARIMA models
app.use('/api/forecast', async (req, res) => {
  try {
    const targetUrl = `${ML_BACKEND_URL}/api/forecast${req.url}`;
    console.log(`📈 Forecast API: ${req.method} ${targetUrl}`);

    const response = await axios({
      method: req.method,
      url: targetUrl,
      data: req.body,
      headers: {
        'Content-Type': 'application/json'
      },
      timeout: 120000
    });

    res.status(response.status).json(response.data);
  } catch (error) {
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else if (error.code === 'ECONNREFUSED') {
      res.status(503).json({
        error: 'ML Backend Unavailable',
        message: 'The Python ML backend is not running. Start with: cd ml_backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload'
      });
    } else {
      console.error('Forecast Proxy Error:', error.message);
      res.status(500).json({ error: 'Forecast proxy error', message: error.message });
    }
  }
});

// Serve geospatial feature on /geospatial path
app.get('/geospatial', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'geospatial.html'));
});

// Monitoring API Proxy - Forward to ML backend /api/monitor endpoints
app.use('/api/monitor', async (req, res) => {
  try {
    const targetUrl = `${ML_BACKEND_URL}/api/monitor${req.url}`;
    console.log(`🛡️  Monitoring API: ${req.method} ${targetUrl}`);

    const response = await axios({
      method: req.method,
      url: targetUrl,
      data: req.body,
      headers: {
        'Content-Type': 'application/json'
      },
      timeout: 120000
    });

    res.status(response.status).json(response.data);
  } catch (error) {
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else if (error.code === 'ECONNREFUSED') {
      res.status(503).json({
        error: 'ML Backend Unavailable',
        message: 'The ML backend server is not running on port 8000.'
      });
    } else {
      console.error('Monitoring Proxy Error:', error.message);
      res.status(500).json({ error: 'Monitoring proxy error', message: error.message });
    }
  }
});

// ML Backend Proxy - Forward requests to Python FastAPI ML backend
app.use('/api/ml', async (req, res) => {
  try {
    const targetUrl = `${ML_BACKEND_URL}${req.originalUrl}`;
    console.log(`🔀 Proxying to ML Backend: ${req.method} ${targetUrl}`);

    const response = await axios({
      method: req.method,
      url: targetUrl,
      data: req.body,
      headers: {
        'Content-Type': 'application/json',
        ...req.headers
      },
      timeout: 120000 // 2 minute timeout for long-running ML tasks
    });

    res.status(response.status).json(response.data);
  } catch (error) {
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else if (error.code === 'ECONNREFUSED') {
      res.status(503).json({
        error: 'ML Backend Unavailable',
        message: 'The ML backend server is not running. Start it with: cd ml_backend && python -m uvicorn main:app --reload'
      });
    } else {
      console.error('ML Proxy Error:', error.message);
      res.status(500).json({ error: 'ML proxy error', message: error.message });
    }
  }
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// ML Backend health check proxy
app.get('/api/ml-health', async (req, res) => {
  try {
    const response = await axios.get(`${ML_BACKEND_URL}/health`, { timeout: 10000 });
    res.json(response.data);
  } catch (error) {
    res.status(503).json({
      status: 'offline',
      error: 'ML Backend unavailable',
      ml_url: ML_BACKEND_URL
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err.message);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});

const server = app.listen(PORT, () => {
  console.log(`🚀 UIDAI Backend Server running on http://localhost:${PORT}`);
  console.log(`📊 Dashboard API: http://localhost:${PORT}/api/dashboard`);
  console.log(`🤖 AI API: http://localhost:${PORT}/api/ai`);
  console.log(`🧠 ML API (proxied): http://localhost:${PORT}/api/ml → ${ML_BACKEND_URL}`);
  console.log(`🗺️  Hotspots API: http://localhost:${PORT}/api/hotspots`);
  console.log(`📈 Spatial Analysis: http://localhost:${PORT}/api/hotspots/spatial`);
  console.log(`\n✅ Server is listening and ready for requests...`);

  // Prevent process from exiting
  process.stdin.resume();
  console.log('📌 Process keep-alive enabled');
});

server.setTimeout(0); // Disable timeout

server.on('error', (err) => {
  console.error('❌ Server error:', err);
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${PORT} is already in use. Please close other instances.`);
    process.exit(1);
  }
});

process.on('uncaughtException', (err) => {
  console.error('❌ Uncaught Exception:', err);
  console.error('Stack:', err.stack);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled Rejection at:', promise);
  console.error('Reason:', reason);
});

