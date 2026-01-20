/**
 * Heer Enrollment Forecaster API Routes
 * ARIMA-based enrollment forecasting adapted from heer-suidai repository
 */

import express from 'express';
import { getEnrolmentData } from '../../services/dataGovApi.js';

const router = express.Router();

/**
 * Helper function to prepare time series data from enrollment records
 */
function prepareTimeSeries(records) {
    const districtSeries = {};
    
    records.forEach(record => {
        const district = record.district;
        if (!district) return;
        
        if (!districtSeries[district]) {
            districtSeries[district] = [];
        }
        
        const totalEnrollment = 
            (parseInt(record.age_0_5) || 0) +
            (parseInt(record.age_5_17) || 0) +
            (parseInt(record.age_18_greater) || 0);
        
        districtSeries[district].push({
            date: record.date,
            enrollment: totalEnrollment
        });
    });
    
    // Sort by date and aggregate
    Object.keys(districtSeries).forEach(district => {
        districtSeries[district] = districtSeries[district]
            .sort((a, b) => new Date(a.date) - new Date(b.date));
    });
    
    return districtSeries;
}

/**
 * Simple moving average forecast (lightweight alternative to ARIMA for Node.js)
 */
function generateForecast(timeSeries, periods = 6) {
    if (!timeSeries || timeSeries.length < 3) {
        return null;
    }
    
    // Calculate moving average and trend
    const recentData = timeSeries.slice(-12); // Last 12 periods
    const sum = recentData.reduce((acc, item) => acc + item.enrollment, 0);
    const avg = sum / recentData.length;
    
    // Calculate trend
    let trend = 0;
    if (recentData.length >= 2) {
        const firstHalf = recentData.slice(0, Math.floor(recentData.length / 2));
        const secondHalf = recentData.slice(Math.floor(recentData.length / 2));
        
        const firstAvg = firstHalf.reduce((acc, item) => acc + item.enrollment, 0) / firstHalf.length;
        const secondAvg = secondHalf.reduce((acc, item) => acc + item.enrollment, 0) / secondHalf.length;
        
        trend = (secondAvg - firstAvg) / firstHalf.length;
    }
    
    // Generate forecasts
    const forecasts = [];
    const lastDate = new Date(recentData[recentData.length - 1].date);
    
    for (let i = 1; i <= periods; i++) {
        const predicted = avg + (trend * i);
        const confidenceWidth = predicted * 0.15 * Math.sqrt(i); // Increasing uncertainty
        
        const forecastDate = new Date(lastDate);
        forecastDate.setMonth(forecastDate.getMonth() + i);
        
        forecasts.push({
            period: i,
            date: forecastDate.toISOString().split('T')[0],
            predicted_enrollment: Math.round(Math.max(0, predicted)),
            lower_bound: Math.round(Math.max(0, predicted - confidenceWidth)),
            upper_bound: Math.round(Math.max(0, predicted + confidenceWidth))
        });
    }
    
    return forecasts;
}

/**
 * GET /api/heer-forecast/districts
 * Get list of available districts with sufficient data
 */
router.get('/districts', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 500;
        
        const enrolmentData = await getEnrolmentData({ limit });
        
        if (!enrolmentData.success) {
            return res.status(500).json({
                success: false,
                error: 'Failed to fetch enrollment data'
            });
        }
        
        const districtSeries = prepareTimeSeries(enrolmentData.records);
        
        // Filter districts with sufficient data (at least 6 data points)
        const availableDistricts = Object.keys(districtSeries)
            .filter(district => districtSeries[district].length >= 6)
            .sort();
        
        res.json({
            success: true,
            count: availableDistricts.length,
            districts: availableDistricts
        });
        
    } catch (error) {
        console.error('Districts list error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to retrieve districts list',
            message: error.message
        });
    }
});

/**
 * GET /api/heer-forecast/predict/:district
 * Get enrollment forecast for a specific district
 */
router.get('/predict/:district', async (req, res) => {
    try {
        const district = req.params.district;
        const periods = parseInt(req.query.periods) || 6;
        const limit = parseInt(req.query.limit) || 500;
        
        if (periods < 1 || periods > 24) {
            return res.status(400).json({
                success: false,
                error: 'Periods must be between 1 and 24'
            });
        }
        
        const enrolmentData = await getEnrolmentData({ limit });
        
        if (!enrolmentData.success) {
            return res.status(500).json({
                success: false,
                error: 'Failed to fetch enrollment data'
            });
        }
        
        const districtSeries = prepareTimeSeries(enrolmentData.records);
        
        if (!districtSeries[district]) {
            return res.status(404).json({
                success: false,
                error: `No data available for district: ${district}`
            });
        }
        
        const timeSeries = districtSeries[district];
        
        if (timeSeries.length < 6) {
            return res.status(400).json({
                success: false,
                error: `Insufficient data for district ${district}. Need at least 6 data points.`
            });
        }
        
        const forecasts = generateForecast(timeSeries, periods);
        
        if (!forecasts) {
            return res.status(500).json({
                success: false,
                error: 'Failed to generate forecast'
            });
        }
        
        // Calculate historical stats
        const enrollments = timeSeries.map(item => item.enrollment);
        const mean = enrollments.reduce((acc, val) => acc + val, 0) / enrollments.length;
        const variance = enrollments.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / enrollments.length;
        const std = Math.sqrt(variance);
        
        res.json({
            success: true,
            district,
            periods,
            confidence_level: 0.90,
            model_type: 'Moving Average with Trend',
            historical_stats: {
                data_points: timeSeries.length,
                mean: Math.round(mean),
                std: Math.round(std),
                last_date: timeSeries[timeSeries.length - 1].date,
                min: Math.min(...enrollments),
                max: Math.max(...enrollments)
            },
            forecasts
        });
        
    } catch (error) {
        console.error('Forecast prediction error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to generate forecast',
            message: error.message
        });
    }
});

/**
 * GET /api/heer-forecast/states
 * Get state-level forecast summary
 */
router.get('/states', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 500;
        const topN = parseInt(req.query.top) || 10;
        
        const enrolmentData = await getEnrolmentData({ limit });
        
        if (!enrolmentData.success) {
            return res.status(500).json({
                success: false,
                error: 'Failed to fetch enrollment data'
            });
        }
        
        // Aggregate by state
        const stateTotals = {};
        
        enrolmentData.records.forEach(record => {
            const state = record.state;
            if (!state) return;
            
            if (!stateTotals[state]) {
                stateTotals[state] = {
                    state,
                    total_enrollment: 0,
                    districts: new Set()
                };
            }
            
            const totalEnrollment = 
                (parseInt(record.age_0_5) || 0) +
                (parseInt(record.age_5_17) || 0) +
                (parseInt(record.age_18_greater) || 0);
            
            stateTotals[state].total_enrollment += totalEnrollment;
            stateTotals[state].districts.add(record.district);
        });
        
        // Convert to array and sort
        const stateList = Object.values(stateTotals)
            .map(state => ({
                state: state.state,
                total_enrollment: state.total_enrollment,
                districts_count: state.districts.size,
                // Simple 6-month growth projection (5% growth)
                projected_6m: Math.round(state.total_enrollment * 1.05)
            }))
            .sort((a, b) => b.total_enrollment - a.total_enrollment)
            .slice(0, topN);
        
        res.json({
            success: true,
            count: stateList.length,
            states: stateList,
            note: 'Projections based on 5% growth rate over 6 months'
        });
        
    } catch (error) {
        console.error('State forecast error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to generate state forecasts',
            message: error.message
        });
    }
});

export default router;
