import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cors from 'cors';

// Import routes
import hotspotsRoutes from './routes/hotspots.js';

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/hotspots', hotspotsRoutes);

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
    console.log(`🗺️  Hotspots API: http://localhost:${PORT}/api/hotspots`);
    console.log(`📊 Spatial Analysis: http://localhost:${PORT}/api/hotspots/spatial`);
    console.log(`🔥 Gi* Hotspots: http://localhost:${PORT}/api/hotspots/gi-star`);
    console.log(`📈 Velocity: http://localhost:${PORT}/api/hotspots/velocity`);
    console.log(`⚠️  Anomalies: http://localhost:${PORT}/api/hotspots/anomalies`);
});
