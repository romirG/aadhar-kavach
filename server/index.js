import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cors from 'cors';

// Import routes
import enrolmentRoutes from './routes/enrolment.js';
import demographicRoutes from './routes/demographic.js';
import biometricRoutes from './routes/biometric.js';
import dashboardRoutes from './routes/dashboard.js';
import aiRoutes from './routes/ai.js';

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/enrolment', enrolmentRoutes);
app.use('/api/demographic', demographicRoutes);
app.use('/api/biometric', biometricRoutes);
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/ai', aiRoutes);

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err.message);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});

app.listen(PORT, () => {
  console.log(`🚀 UIDAI Backend Server running on http://localhost:${PORT}`);
  console.log(`📊 Dashboard API: http://localhost:${PORT}/api/dashboard`);
  console.log(`🤖 AI API: http://localhost:${PORT}/api/ai`);
});
