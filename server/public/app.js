/**
 * UIDAI Analytics Dashboard - Frontend JavaScript
 */

const API_BASE = 'http://localhost:3001/api';
const ML_API_BASE = 'http://localhost:8000';

// =====================================
// Theme Toggle
// =====================================

function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const themeIcon = document.querySelector('.theme-icon');
    if (themeIcon) {
        themeIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
}

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', initTheme);

// =====================================
// Status Check
// =====================================

async function checkStatus() {
    // Check backend status
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            document.getElementById('backend-status').textContent = 'Online';
            document.querySelector('.status-item:first-child .status-dot').classList.add('online');
            document.querySelector('.status-item:first-child .status-dot').classList.remove('offline');
        }
    } catch (e) {
        document.getElementById('backend-status').textContent = 'Offline';
    }

    // Check ML backend status
    try {
        const response = await fetch(`${ML_API_BASE}/health`);
        if (response.ok) {
            document.getElementById('ml-status').textContent = 'Online';
            document.querySelector('.status-item:last-child .status-dot').classList.add('online');
            document.querySelector('.status-item:last-child .status-dot').classList.remove('offline');
        }
    } catch (e) {
        document.getElementById('ml-status').textContent = 'Offline';
    }
}

// =====================================
// Feature Loaders
// =====================================

async function loadFeature(feature) {
    showLoading();

    try {
        switch (feature) {
            case 'dashboard':
                await loadDashboard();
                break;
            case 'hotspots':
                await loadHotspots();
                break;
            case 'geospatial':
                // Open geospatial feature in same tab
                window.location.href = '/geospatial';
                break;
            case 'anomalies':
                await loadAnomalies();
                break;
            case 'gender':
                await loadGenderTracker();
                break;
            case 'map':
                await loadGenderGapMap();
                break;
            case 'vulnerable':
                await loadVulnerableGroups();
                break;
            case 'simulator':
                await loadImpactSimulator();
                break;
            case 'risk':
                await loadRiskPredictor();
                break;
            case 'forecast':
                await loadForecast();
                break;
            case 'monitoring':
                await loadMonitoring();
                break;
            default:
                showError('Unknown feature');
        }
    } catch (error) {
        showError(error.message);
    }
}

async function loadData(dataType) {
    showLoading();

    try {
        const response = await fetch(`${API_BASE}/${dataType}?limit=50`);
        const data = await response.json();

        if (data.success) {
            showDataTable(dataType, data.records);
        } else {
            showError(data.error || 'Failed to load data');
        }
    } catch (error) {
        showError(`Failed to fetch ${dataType} data: ${error.message}`);
    }
}

// =====================================
// Feature Implementations
// =====================================

async function loadDashboard() {
    setTitle('📈 Dashboard Overview');

    // Fetch all data sources
    const [enrolment, demographic, biometric] = await Promise.all([
        fetch(`${API_BASE}/enrolment?limit=100`).then(r => r.json()).catch(() => ({ records: [] })),
        fetch(`${API_BASE}/demographic?limit=100`).then(r => r.json()).catch(() => ({ records: [] })),
        fetch(`${API_BASE}/biometric?limit=100`).then(r => r.json()).catch(() => ({ records: [] }))
    ]);

    // Calculate stats
    const totalEnrollments = enrolment.records?.reduce((sum, r) =>
        sum + (parseInt(r.age_0_5) || 0) + (parseInt(r.age_5_17) || 0) + (parseInt(r.age_18_greater) || 0), 0) || 0;

    const uniqueStates = new Set(enrolment.records?.map(r => r.state) || []).size;
    const uniqueDistricts = new Set(enrolment.records?.map(r => r.district) || []).size;

    showContent(`
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${formatNumber(totalEnrollments)}</div>
                <div class="stat-label">Total Enrollments</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${uniqueStates}</div>
                <div class="stat-label">States</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${uniqueDistricts}</div>
                <div class="stat-label">Districts</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${enrolment.total || enrolment.records?.length || 0}</div>
                <div class="stat-label">Records Available</div>
            </div>
        </div>
        
        <h3 style="margin: 20px 0 15px; color: #000080;">📊 Data Sources Status</h3>
        <div class="alert alert-info">
            <strong>Enrolment API:</strong> ${enrolment.records?.length || 0} records loaded
            ${enrolment.isMockData ? ' (Mock Data)' : ''}
        </div>
        <div class="alert alert-info">
            <strong>Demographic API:</strong> ${demographic.records?.length || 0} records loaded
        </div>
        <div class="alert alert-info">
            <strong>Biometric API:</strong> ${biometric.records?.length || 0} records loaded
        </div>
    `);
}

async function loadHotspots() {
    setTitle('🗺️ Geographic Hotspots - Gi* Analysis');

    try {
        const response = await fetch(`${API_BASE}/hotspots/gi-star?limit=200`);
        const data = await response.json();

        if (data.success && data.data) {
            const { summary, coldspots, hotspots } = data.data;

            showContent(`
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">${summary?.totalRegions || 0}</div>
                        <div class="stat-label">Regions Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #DC3545;">${summary?.coldspotCount || 0}</div>
                        <div class="stat-label">Coldspots (Low Coverage)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #138808;">${summary?.hotspotCount || 0}</div>
                        <div class="stat-label">Hotspots (High Coverage)</div>
                    </div>
                </div>
                
                <h3 style="margin: 20px 0 15px; color: #DC3545;">🔴 Priority Coldspots (Need Intervention)</h3>
                ${renderHotspotList(coldspots || [], 'coldspot')}
                
                <h3 style="margin: 20px 0 15px; color: #138808;">🟢 Top Performing Hotspots</h3>
                ${renderHotspotList(hotspots || [], 'hotspot')}
            `);
        } else {
            // Fallback to basic data
            const enrolment = await fetch(`${API_BASE}/enrolment?limit=100`).then(r => r.json());
            showContent(`
                <div class="alert alert-warning">
                    <strong>Note:</strong> Hotspot API not available. Showing basic enrollment data.
                </div>
                ${renderBasicStats(enrolment.records || [])}
            `);
        }
    } catch (error) {
        showError(`Failed to load hotspots: ${error.message}`);
    }
}

async function loadAnomalies() {
    setTitle('⚠️ Anomaly Detection');

    try {
        const response = await fetch(`${API_BASE}/hotspots/anomalies?limit=200`);
        const data = await response.json();

        if (data.success && data.data) {
            const { summary, alerts } = data.data;

            showContent(`
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" style="color: #FF9933;">${summary?.totalAnomalies || 0}</div>
                        <div class="stat-label">Total Anomalies</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #DC3545;">${summary?.critical || 0}</div>
                        <div class="stat-label">Critical</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #FF9933;">${summary?.high || 0}</div>
                        <div class="stat-label">High Priority</div>
                    </div>
                </div>
                
                <h3 style="margin: 20px 0 15px;">🚨 Active Alerts</h3>
                ${renderAlerts(alerts || [])}
            `);
        } else {
            showContent(`
                <div class="alert alert-info">
                    <strong>Anomaly Detection:</strong> No anomalies detected or API unavailable.
                </div>
                <p style="margin-top: 15px; color: #6c757d;">
                    The anomaly detection system uses Isolation Forest and ensemble ML models 
                    to identify unusual patterns in enrollment data.
                </p>
            `);
        }
    } catch (error) {
        showError(`Failed to load anomalies: ${error.message}`);
    }
}

async function loadGenderTracker() {
    setTitle('👥 Gender Inclusion Tracker');
    showLoading();

    try {
        // Fetch gender analysis data from ridwan API
        const response = await fetch(`${API_BASE}/ridwan-gender/high-risk?limit=500`);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to fetch gender data');
        }

        const { riskDistribution, districts } = data.data;
        const totalHighRisk = data.data.totalHighRisk || 0;

        // Fetch state-level coverage
        const coverageResponse = await fetch(`${API_BASE}/ridwan-gender/coverage?limit=500`);
        const coverageData = await coverageResponse.json();
        
        const stateAnalysis = coverageData.data?.stateAnalysis || [];
        const summary = coverageData.data?.summary || {};

        showContent(`
            <div class="alert alert-info">
                <strong>🎯 Gender Inclusion Analysis:</strong> 
                Monitoring female Aadhaar enrollment coverage to identify and address the digital gender gap.
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${summary.totalStates || 0}</div>
                    <div class="stat-label">States Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${summary.totalDistricts || 0}</div>
                    <div class="stat-label">Districts Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: #DC3545;">${totalHighRisk}</div>
                    <div class="stat-label">High-Risk Districts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${((summary.avgGenderGap || 0) * 100).toFixed(1)}%</div>
                    <div class="stat-label">Avg Gender Gap</div>
                </div>
            </div>

            <h3 style="margin: 25px 0 15px; color: #DC3545;">⚠️ High-Risk Districts (Gender Gap > 3%)</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>District</th>
                        <th>State</th>
                        <th>Gender Gap</th>
                        <th>Female %</th>
                        <th>Male %</th>
                        <th>Risk</th>
                    </tr>
                </thead>
                <tbody>
                    ${districts.slice(0, 15).map(d => `
                        <tr>
                            <td>${d.district}</td>
                            <td>${d.state}</td>
                            <td style="color: #DC3545; font-weight: bold;">${(d.genderGap * 100).toFixed(1)}%</td>
                            <td>${(d.femaleCoverageRatio * 100).toFixed(1)}%</td>
                            <td>${(d.maleCoverageRatio * 100).toFixed(1)}%</td>
                            <td>
                                <span style="padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;
                                    background: ${d.riskLevel === 'CRITICAL' ? '#ff0000' : d.riskLevel === 'HIGH' ? '#ff9900' : '#ffcc00'}; 
                                    color: #1a1a2e;">
                                    ${d.riskLevel}
                                </span>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>

            <h3 style="margin: 25px 0 15px; color: #000080;">📊 State-wise Gender Coverage</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>State</th>
                        <th>Female Coverage</th>
                        <th>Male Coverage</th>
                        <th>Gender Gap</th>
                    </tr>
                </thead>
                <tbody>
                    ${stateAnalysis.slice(0, 10).map(s => `
                        <tr>
                            <td>${s.state}</td>
                            <td>${(s.femaleCoverageRatio * 100).toFixed(1)}%</td>
                            <td>${(s.maleCoverageRatio * 100).toFixed(1)}%</td>
                            <td style="color: ${s.genderGap > 0.03 ? '#DC3545' : '#138808'};">
                                ${(s.genderGap * 100).toFixed(1)}%
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `);
    } catch (error) {
        showError(`Failed to load gender data: ${error.message}`);
    }
}

async function loadRiskPredictor() {
    setTitle('🔮 Biometric Re-enrollment Risk Predictor');

    try {
        // Check if ML backend is available
        const mlHealth = await fetch(`${ML_API_BASE}/health`).then(r => r.json()).catch(() => null);

        if (mlHealth) {
            showContent(`
                <div class="alert alert-info">
                    <strong>ML Backend Connected!</strong> Risk prediction models are available.
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">3</div>
                        <div class="stat-label">Dataset APIs</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">4</div>
                        <div class="stat-label">ML Models</div>
                    </div>
                </div>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="/risk_analysis.html" style="
                        display: inline-block;
                        background: linear-gradient(135deg, #FF9933, #000080);
                        color: #1a1a2e;
                        text-decoration: none;
                        padding: 15px 40px;
                        font-size: 1.1rem;
                        border-radius: 10px;
                        transition: all 0.3s;
                    ">
                        🧠 Open Full Analysis Tool →
                    </a>
                </div>
                
                <h3 style="margin: 20px 0 15px;">🧠 Available Models</h3>
                <div class="alert alert-info">
                    <strong>Random Forest:</strong> Risk classification based on regional features
                </div>
                <div class="alert alert-info">
                    <strong>XGBoost:</strong> Gradient boosting for high-accuracy predictions
                </div>
                <div class="alert alert-info">
                    <strong>Isolation Forest:</strong> Anomaly-based risk detection
                </div>
                <div class="alert alert-info">
                    <strong>LightGBM:</strong> Gender inclusion gap prediction
                </div>
            `);
        } else {
            showContent(`
                <div class="alert alert-warning">
                    <strong>ML Backend Offline:</strong> Start the ML backend to use risk prediction.
                </div>
                <p style="margin-top: 15px; color: #6c757d;">
                    Run: <code style="background: #dee2e6; padding: 5px 10px; border-radius: 5px;">
                    cd ml_backend && python -m uvicorn main:app --reload
                    </code>
                </p>
            `);
        }
    } catch (error) {
        showError(`Failed to check ML backend: ${error.message}`);
    }
}

async function loadForecast() {
    setTitle('📅 Enrollment Forecast - ARIMA Predictions');

    try {
        // Check if Python ML backend is available
        let backendAvailable = false;
        try {
            const healthCheck = await fetch(`${API_BASE}/forecast/districts`);
            backendAvailable = healthCheck.ok;
        } catch (e) {
            backendAvailable = false;
        }

        if (!backendAvailable) {
            showContent(`
                <div class="alert alert-warning">
                    <strong>⚠️ Python ML Backend Required</strong>
                    <p>The ARIMA enrollment forecaster requires the Python FastAPI backend.</p>
                    <p style="margin-top: 15px;">
                        <strong>Start the backend:</strong><br>
                        <code style="background: #dee2e6; padding: 5px 10px; border-radius: 5px; display: inline-block; margin-top: 5px;">
                        cd ml_backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
                        </code>
                    </p>
                    <p style="margin-top: 15px; color: #6c757d; font-size: 0.9em;">
                        This will enable full ARIMA forecasting with:<br>
                        ✅ Automatic stationarity testing<br>
                        ✅ Confidence intervals from statsmodels<br>
                        ✅ Model training on real data.gov.in data<br>
                        ✅ State-level aggregated forecasts
                    </p>
                </div>
            `);
            return;
        }

        // First get list of available districts
        const districtsResponse = await fetch(`${API_BASE}/forecast/districts`);
        const districtsData = await districtsResponse.json();

        const districts = districtsData.districts || [];
        const trained = districts.length > 0;
        
        showContent(`
            <div class="alert alert-info" style="margin-bottom: 20px;">
                <strong>🔮 ARIMA-Based Enrollment Forecasting</strong>
                <p style="margin: 10px 0 0 0; font-size: 0.9em;">
                    Time-series predictions using statsmodels ARIMA with automatic stationarity testing.
                    ${trained ? 
                        `Select a district to view 6-month enrollment forecasts with confidence intervals.` :
                        `Train models first to enable predictions.`
                    }
                </p>
            </div>

            ${!trained ? `
                <div class="alert alert-warning" style="margin-bottom: 20px;">
                    <strong>⚠️ No Trained Models Found</strong>
                    <p style="margin: 10px 0;">Train ARIMA models on data.gov.in enrollment data first.</p>
                    <button onclick="trainForecastModels()" class="btn-primary" style="margin-top: 10px;">
                        🎯 Train Models (500 records, top 30 districts)
                    </button>
                </div>
            ` : ''}

            ${trained ? `
                <div class="forecast-selector" style="margin-bottom: 20px;">
                    <label for="district-select" style="display: block; margin-bottom: 10px; font-weight: bold;">
                        Select District (${districts.length} available):
                    </label>
                    <div style="display: flex; gap: 10px;">
                        <select id="district-select" style="flex: 1; padding: 10px; border-radius: 4px; border: 1px solid #dee2e6; background: #FFFFFFFFF; color: #1a1a2e;">
                            <option value="">-- Choose a district --</option>
                            ${districts.map(d => `<option value="${d}">${d}</option>`).join('')}
                        </select>
                        <button onclick="runForecast()" class="btn-primary" style="padding: 10px 20px;">
                            Generate Forecast
                        </button>
                        <button onclick="trainForecastModels()" class="btn-secondary" style="padding: 10px 20px;">
                            🔄 Re-train
                        </button>
                    </div>
                </div>
            ` : ''}

            <div id="forecast-results"></div>
        `);
    } catch (error) {
        showError(`Failed to load forecast: ${error.message}`);
    }
}

async function trainForecastModels() {
    const resultsDiv = document.getElementById('forecast-results') || document.body;
    resultsDiv.innerHTML = '<div class="loading">🎯 Training ARIMA models on data.gov.in data...<br><small>This may take 30-60 seconds</small></div>';

    try {
        const response = await fetch(`${API_BASE}/forecast/train?limit=500&max_districts=30`, {
            method: 'POST'
        });
        const data = await response.json();

        if (data.status === 'success') {
            resultsDiv.innerHTML = `
                <div class="alert alert-success">
                    <strong>✅ Training Complete!</strong>
                    <p>Successfully trained ${data.trained_count} district models.</p>
                    <p><small>Model saved to: ${data.model_path}</small></p>
                    <button onclick="loadForecast()" class="btn-primary" style="margin-top: 10px;">
                        View Forecasts
                    </button>
                </div>
            `;
        } else {
            resultsDiv.innerHTML = `
                <div class="alert alert-error">
                    <strong>❌ Training Failed</strong>
                    <p>${data.detail || 'Unknown error during training'}</p>
                </div>
            `;
        }
    } catch (error) {
        resultsDiv.innerHTML = `
            <div class="alert alert-error">
                <strong>❌ Training Error</strong>
                <p>${error.message}</p>
            </div>
        `;
    }
}

async function runForecast() {
    const select = document.getElementById('district-select');
    const district = select.value;
    
    if (!district) {
        alert('Please select a district');
        return;
    }

    const resultsDiv = document.getElementById('forecast-results');
    resultsDiv.innerHTML = '<div class="loading">⏳ Generating ARIMA forecast...</div>';

    try {
        const response = await fetch(`${API_BASE}/forecast/predict/${encodeURIComponent(district)}?periods=6&confidence=0.95`);
        const data = await response.json();

        if (data.detail) {
            resultsDiv.innerHTML = `<div class="alert alert-error">${data.detail}</div>`;
            return;
        }

        const { forecasts, historical_stats, confidence_level } = data;

        // Prepare chart data
        const labels = forecasts.map((f, i) => `Month ${i + 1}`);
        const predicted = forecasts.map(f => f.predicted);
        const lowerBound = forecasts.map(f => f.ci_lower);
        const upperBound = forecasts.map(f => f.ci_upper);

        resultsDiv.innerHTML = `
            <div class="stats-grid" style="margin-bottom: 20px;">
                <div class="stat-card">
                    <div class="stat-value">${formatNumber(historical_stats.data_points)}</div>
                    <div class="stat-label">Historical Data Points</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${formatNumber(Math.round(historical_stats.mean))}</div>
                    <div class="stat-label">Average Enrollment</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${formatNumber(Math.round(historical_stats.max))}</div>
                    <div class="stat-label">Peak Enrollment</div>
                </div>
            </div>

            <div style="background: #F5F5F5; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span><strong>Model:</strong> ARIMA${JSON.stringify(historical_stats.order)}</span>
                    <span><strong>Confidence:</strong> ${(confidence_level * 100).toFixed(0)}%</span>
                    <span><strong>AIC:</strong> ${historical_stats.aic?.toFixed(2)}</span>
                </div>
                <div><strong>Last Date:</strong> ${historical_stats.last_date}</div>
            </div>

            <h3 style="margin: 20px 0 15px;">📈 6-Month Forecast</h3>
            <canvas id="forecast-chart" style="max-height: 400px;"></canvas>

            <h3 style="margin: 30px 0 15px;">📋 Detailed Predictions</h3>
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Period</th>
                            <th>Predicted</th>
                            <th>Lower Bound (95%)</th>
                            <th>Upper Bound (95%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${forecasts.map((f, i) => `
                            <tr>
                                <td>Month ${i + 1}</td>
                                <td><strong>${formatNumber(Math.round(f.predicted))}</strong></td>
                                <td>${formatNumber(Math.round(f.ci_lower))}</td>
                                <td>${formatNumber(Math.round(f.ci_upper))}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;

        // Create Chart.js visualization
        const ctx = document.getElementById('forecast-chart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Predicted Enrollment',
                        data: predicted,
                        borderColor: '#138808',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Upper Bound (90% CI)',
                        data: upperBound,
                        borderColor: '#ff9900',
                        backgroundColor: 'rgba(255, 153, 0, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Lower Bound (90% CI)',
                        data: lowerBound,
                        borderColor: '#ff9900',
                        backgroundColor: 'rgba(255, 153, 0, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: '-1',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    title: {
                        display: true,
                        text: `Enrollment Forecast - ${district}`,
                        color: '#1a1a2e',
                        font: { size: 16 }
                    },
                    legend: {
                        labels: { color: '#1a1a2e' }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + formatNumber(context.parsed.y);
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#6c757d' },
                        grid: { color: '#dee2e6' }
                    },
                    y: {
                        ticks: { 
                            color: '#6c757d',
                            callback: function(value) {
                                return formatNumber(value);
                            }
                        },
                        grid: { color: '#dee2e6' }
                    }
                }
            }
        });

    } catch (error) {
        resultsDiv.innerHTML = `<div class="alert alert-error">Error: ${error.message}</div>`;
    }
}

async function loadMonitoring() {
    setTitle('🛡️ Operations Monitoring');

    try {
        // Check if ML backend and monitoring API are available
        const intentsResponse = await fetch(`${API_BASE}/monitor/intents`).catch(() => null);

        if (intentsResponse && intentsResponse.ok) {
            const intentsData = await intentsResponse.json();
            const intents = intentsData.intents || [];
            const vigilanceLevels = intentsData.vigilance_levels || [];

            showContent(`
                <div class="alert alert-info">
                    <strong>🛡️ Intent-Based Monitoring System</strong><br>
                    AI-powered monitoring for UIDAI auditors. Select what you want to monitor and the system will analyze operations.
                </div>

                <div class="monitoring-controls" style="margin-top: 20px;">
                    <div class="stats-grid">
                        <div class="stat-card" style="text-align: left;">
                            <label style="display: block; margin-bottom: 8px; color: #000080;">Monitoring Type</label>
                            <select id="monitoring-intent" class="monitoring-select">
                                ${intents.map(i => `<option value="${i.id}">${i.display_name}</option>`).join('')}
                            </select>
                        </div>
                        <div class="stat-card" style="text-align: left;">
                            <label style="display: block; margin-bottom: 8px; color: #000080;">Focus Area</label>
                            <select id="monitoring-state" class="monitoring-select">
                                <option value="All India">All India</option>
                                <option value="Andhra Pradesh">Andhra Pradesh</option>
                                <option value="Arunachal Pradesh">Arunachal Pradesh</option>
                                <option value="Assam">Assam</option>
                                <option value="Bihar">Bihar</option>
                                <option value="Chhattisgarh">Chhattisgarh</option>
                                <option value="Goa">Goa</option>
                                <option value="Gujarat">Gujarat</option>
                                <option value="Haryana">Haryana</option>
                                <option value="Himachal Pradesh">Himachal Pradesh</option>
                                <option value="Jharkhand">Jharkhand</option>
                                <option value="Karnataka">Karnataka</option>
                                <option value="Kerala">Kerala</option>
                                <option value="Madhya Pradesh">Madhya Pradesh</option>
                                <option value="Maharashtra">Maharashtra</option>
                                <option value="Manipur">Manipur</option>
                                <option value="Meghalaya">Meghalaya</option>
                                <option value="Mizoram">Mizoram</option>
                                <option value="Nagaland">Nagaland</option>
                                <option value="Odisha">Odisha</option>
                                <option value="Punjab">Punjab</option>
                                <option value="Rajasthan">Rajasthan</option>
                                <option value="Sikkim">Sikkim</option>
                                <option value="Tamil Nadu">Tamil Nadu</option>
                                <option value="Telangana">Telangana</option>
                                <option value="Tripura">Tripura</option>
                                <option value="Uttar Pradesh">Uttar Pradesh</option>
                                <option value="Uttarakhand">Uttarakhand</option>
                                <option value="West Bengal">West Bengal</option>
                                <option value="Delhi">Delhi</option>
                                <option value="Jammu and Kashmir">Jammu and Kashmir</option>
                                <option value="Ladakh">Ladakh</option>
                                <option value="Puducherry">Puducherry</option>
                            </select>
                        </div>
                        <div class="stat-card" style="text-align: left;">
                            <label style="display: block; margin-bottom: 8px; color: #000080;">Time Period</label>
                            <select id="monitoring-period" class="monitoring-select">
                                <option value="today">Today</option>
                                <option value="last_7_days">Last 7 Days</option>
                                <option value="this_month">This Month</option>
                            </select>
                        </div>
                        <div class="stat-card" style="text-align: left;">
                            <label style="display: block; margin-bottom: 8px; color: #000080;">Vigilance Level</label>
                            <select id="monitoring-vigilance" class="monitoring-select">
                                ${vigilanceLevels.map(v => `<option value="${v.id}">${v.name}</option>`).join('')}
                            </select>
                        </div>
                    </div>
                    
                    <button onclick="startMonitoring()" class="btn-monitor">
                        🚀 Start Monitoring
                    </button>
                </div>

                <div id="monitoring-results" style="margin-top: 20px;"></div>
            `);
        } else {
            showContent(`
                <div class="alert alert-warning">
                    <strong>ML Backend Required:</strong> The monitoring system requires the ML backend to be running.
                </div>
                <p style="margin-top: 15px; color: #6c757d;">
                    Start the ML backend with:<br>
                    <code style="background: #dee2e6; padding: 5px 10px; border-radius: 5px; display: inline-block; margin-top: 10px;">
                    cd ml_backend && python main.py
                    </code>
                </p>
            `);
        }
    } catch (error) {
        showError(`Failed to load monitoring: ${error.message}`);
    }
}

async function startMonitoring() {
    const intent = document.getElementById('monitoring-intent').value;
    const state = document.getElementById('monitoring-state').value;
    const period = document.getElementById('monitoring-period').value;
    const vigilance = document.getElementById('monitoring-vigilance').value;

    const resultsDiv = document.getElementById('monitoring-results');
    resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        // Submit monitoring request
        const submitResponse = await fetch(`${API_BASE}/monitor`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                intent,
                focus_area: state === 'All India' ? undefined : state,
                time_period: period,
                vigilance,
                record_limit: 1000
            })
        });

        if (!submitResponse.ok) throw new Error('Failed to submit monitoring request');

        const job = await submitResponse.json();
        const jobId = job.job_id;

        // Poll for results
        let attempts = 0;
        const maxAttempts = 60;

        const pollResults = async () => {
            attempts++;
            const statusResponse = await fetch(`${API_BASE}/monitor/status/${jobId}`);
            const status = await statusResponse.json();

            resultsDiv.innerHTML = `
                <div class="alert alert-info">
                    <strong>Status:</strong> ${status.message}
                    <div style="margin-top: 10px; background: #dee2e6; border-radius: 10px; overflow: hidden;">
                        <div style="width: ${status.progress}%; height: 6px; background: linear-gradient(90deg, #000080, #FF9933);"></div>
                    </div>
                    <span style="font-size: 0.85rem; color: #6c757d;">${status.progress}% complete</span>
                </div>
            `;

            if (status.status === 'completed') {
                const resultsResponse = await fetch(`${API_BASE}/monitor/results/${jobId}`);
                const results = await resultsResponse.json();
                displayMonitoringResults(results);
            } else if (status.status === 'failed') {
                resultsDiv.innerHTML = `<div class="alert alert-critical"><strong>Error:</strong> ${status.message}</div>`;
            } else if (attempts < maxAttempts) {
                setTimeout(pollResults, 1000);
            } else {
                resultsDiv.innerHTML = '<div class="alert alert-warning">Timeout: Analysis is taking too long.</div>';
            }
        };

        pollResults();
    } catch (error) {
        resultsDiv.innerHTML = `<div class="alert alert-critical"><strong>Error:</strong> ${error.message}</div>`;
    }
}

function displayMonitoringResults(results) {
    // IMPORTANT: Add focus_area from form to stored results for AI regeneration
    const stateInput = document.getElementById('monitoring-state');
    const focusArea = stateInput ? stateInput.value : 'All India';
    results.focus_area = focusArea === 'All India' ? 'All India' : focusArea;

    window.lastMonitoringResults = results;
    const riskColor = results.risk.risk_level === 'Low' ? '#138808' :
        results.risk.risk_level === 'Medium' ? '#FF9933' : '#DC3545';

    document.getElementById('monitoring-results').innerHTML = `
        <div class="stats-grid">
            <div class="stat-card" style="border-left: 4px solid ${riskColor};">
                <div class="stat-value" style="color: ${riskColor};">${results.risk.risk_index}</div>
                <div class="stat-label">Risk Index</div>
                <span style="background: ${riskColor}; color: #000; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem;">
                    ${results.risk.risk_level}
                </span>
            </div>
            <div class="stat-card">
                <div class="stat-value">${results.records_analyzed.toLocaleString()}</div>
                <div class="stat-label">Records Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #FF9933;">${results.flagged_for_review}</div>
                <div class="stat-label">Flagged for Review</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #138808;">${results.cleared}</div>
                <div class="stat-label">Cleared</div>
            </div>
        </div>

        <h3 style="margin: 20px 0 15px; color: #000080;">📋 Monitoring Summary</h3>
        <div class="alert alert-info">${results.summary}</div>

        <h3 style="margin: 20px 0 15px; color: #FF9933;">⚠️ Key Observations (${results.findings.length})</h3>
        ${results.findings.slice(0, 5).map((f, i) => {
        const severityColor = f.severity === 'High' ? '#DC3545' : f.severity === 'Medium' ? '#FF9933' : '#000080';
        const bgColor = f.severity === 'High' ? 'rgba(255,68,68,0.15)' : f.severity === 'Medium' ? 'rgba(255,170,0,0.15)' : 'rgba(0,212,255,0.15)';
        const location = f.location || f.state || 'National';
        return `
            <div class="alert" style="border-left: 4px solid ${severityColor}; background: ${bgColor};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <strong style="flex: 1;">${f.title}</strong>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <span style="background: rgba(100,100,255,0.3); color: #6c757d8ff; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem;">📍 ${location}</span>
                        <span style="background: ${severityColor}; color: #000; padding: 2px 10px; border-radius: 10px; font-size: 0.75rem; font-weight: bold;">${f.severity}</span>
                    </div>
                </div>
                <span style="color: #4a4a6a;">${f.description}</span>
                
                <div style="margin-top: 10px;">
                    <button onclick="toggleFindingDetails(${i})" style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #1a1a2e; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8rem;">
                        ▼ Details
                    </button>
                    <div id="finding-details-${i}" style="display: none; margin-top: 10px; background: rgba(0,0,0,0.4); padding: 15px; border-radius: 6px; border-left: 3px solid ${severityColor};">
                        <h4 style="color: #FF9933; margin: 0 0 8px 0; font-size: 0.95rem;">📊 Impact Assessment - ${location}</h4>
                        <p style="margin: 0; color: #4a4a6a; font-size: 0.9rem;">${f.details || 'Pattern detected requiring investigation within the focus area.'}</p>
                    </div>
                </div>
            </div>
        `;
    }).join('')}
        
        ${results.findings.length > 5 ? `
            <button id="show-more-findings-btn" onclick="toggleMoreFindings()" style="background: rgba(255,170,0,0.2); border: 1px solid #FF9933; color: #FF9933; padding: 10px 20px; border-radius: 6px; cursor: pointer; width: 100%; margin: 10px 0;">
                ▼ Show ${results.findings.length - 5} More Findings
            </button>
            <div id="more-findings" style="display: none;">
                ${results.findings.slice(5).map((f, i) => `
                    <div class="alert" style="border-left: 4px solid ${f.severity === 'High' ? '#DC3545' : f.severity === 'Medium' ? '#FF9933' : '#000080'}; background: ${f.severity === 'High' ? 'rgba(255,68,68,0.15)' : f.severity === 'Medium' ? 'rgba(255,170,0,0.15)' : 'rgba(0,212,255,0.15)'}; margin-top: 10px;">
                        <strong>${f.title}</strong> <span style="float: right; color: ${f.severity === 'High' ? '#DC3545' : f.severity === 'Medium' ? '#FF9933' : '#000080'}; font-weight: bold;">${f.severity}</span>
                        <br><span style="color: #4a4a6a;">${f.description}</span>
                    </div>
                `).join('')}
            </div>
        ` : ''}

        <div style="background: linear-gradient(135deg, rgba(139,92,246,0.2) 0%, rgba(59,130,246,0.2) 100%); border: 1px solid rgba(139,92,246,0.4); border-radius: 12px; padding: 20px; margin-top: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <div>
                    <h3 style="margin: 0; color: #a78bfa;">🤖 AI-Powered Analysis</h3>
                    <p style="margin: 5px 0 0; color: #6c757d; font-size: 0.8rem;">Model: <strong style="color: #8b5cf6;">llama-3.3-70b-versatile</strong> via Groq LPU</p>
                </div>
                <button id="regenerate-ai-btn" onclick="regenerateAIAnalysis()" style="background: linear-gradient(135deg, #8b5cf6, #6366f1); border: none; color: #1a1a2e; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-weight: 600; display: flex; align-items: center; gap: 8px; transition: all 0.3s;">
                    <span>✨</span> Generate AI Recommendations
                </button>
            </div>
            <div id="ai-recommendations-container">
                <p style="color: #6c757d; text-align: center; padding: 20px;">Click the button above to generate unique AI-powered recommendations based on current findings.</p>
            </div>
        </div>

        <p style="text-align: center; color: #666; margin-top: 20px; font-size: 0.85rem;">
            Report ID: ${results.report_id} | Generated: ${new Date(results.completed_at).toLocaleString()}
        </p>
    `;
}

// =====================================
// AI Analysis Regeneration
// =====================================

async function regenerateAIAnalysis() {
    const btn = document.getElementById('regenerate-ai-btn');
    const container = document.getElementById('ai-recommendations-container');

    // Get current results
    const results = window.lastMonitoringResults;
    if (!results) {
        container.innerHTML = '<p style="color: #DC3545; text-align: center;">No analysis results available. Run monitoring first.</p>';
        return;
    }

    // Show loading state
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner" style="width: 16px; height: 16px; border-width: 2px;"></span> Generating...';
    container.innerHTML = `
        <div style="text-align: center; padding: 30px;">
            <div class="spinner" style="margin: 0 auto 15px;"></div>
            <p style="color: #a78bfa;">Generating recommendations...</p>
        </div>
    `;

    try {
        const response = await fetch(`${ML_API_BASE}/api/monitor/regenerate-ai`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                focus_area: results.focus_area || 'All India',
                intent: results.intent || 'Operations Monitoring',
                findings: results.findings || [],
                risk_level: results.risk?.risk_level || 'Medium',
                total_analyzed: results.records_analyzed || 0,
                total_flagged: results.flagged_for_review || 0
            })
        });

        const data = await response.json();

        if (data.success) {
            // Display the new recommendations
            container.innerHTML = `
                <div style="margin-bottom: 15px; padding: 10px; background: rgba(139,92,246,0.1); border-radius: 8px;">
                    <p style="margin: 0; color: #a78bfa; font-size: 0.8rem;">
                        ✅ Generated at ${new Date(data.generated_at).toLocaleTimeString()} | Model: <strong>${data.model}</strong>
                    </p>
                </div>
                
                ${data.summary ? `
                    <div style="background: rgba(0,212,255,0.1); border-left: 3px solid #000080; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
                        <h4 style="color: #000080; margin: 0 0 8px;">📋 AI Summary</h4>
                        <p style="color: #4a4a6a; margin: 0; font-size: 0.9rem;">${data.summary}</p>
                    </div>
                ` : ''}
                
                <h4 style="color: #138808; margin: 15px 0 10px;">✅ Recommended Actions (${data.recommended_actions?.length || 0})</h4>
                ${(data.recommended_actions || []).map((a, i) => {
                const categoryColors = {
                    'Audit': { bg: 'rgba(255,99,71,0.2)', border: '#ff6347', accent: '#ff6347' },
                    'Investigation': { bg: 'rgba(255,165,0,0.2)', border: '#ffa500', accent: '#ffa500' },
                    'Training': { bg: 'rgba(50,205,50,0.2)', border: '#32cd32', accent: '#32cd32' },
                    'Infrastructure': { bg: 'rgba(30,144,255,0.2)', border: '#1e90ff', accent: '#1e90ff' },
                    'Policy': { bg: 'rgba(186,85,211,0.2)', border: '#ba55d3', accent: '#ba55d3' }
                };
                const category = a.action_category || 'Policy';
                const colors = categoryColors[category] || categoryColors['Policy'];
                const priorityBg = a.priority === 'Urgent' ? '#DC3545' : a.priority === 'High' ? '#FF9933' : '#138808';
                return `
                    <div style="background: ${colors.bg}; border-left: 4px solid ${colors.border}; padding: 15px; margin-bottom: 12px; border-radius: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                            <div>
                                <h5 style="margin: 0 0 5px; color: ${colors.accent}; font-size: 1rem;">
                                    ${a.action_title || 'Action ' + (i + 1)}
                                </h5>
                                <div style="display: flex; gap: 8px; font-size: 0.7rem;">
                                    <span style="background: ${colors.accent}; color: #000; padding: 2px 8px; border-radius: 8px; font-weight: 600;">
                                        ${category}
                                    </span>
                                    ${a.target_region ? `<span style="background: rgba(100,100,255,0.3); color: #6c757d8ff; padding: 2px 8px; border-radius: 8px;">📍 ${a.target_region}</span>` : ''}
                                </div>
                            </div>
                            <span style="background: ${priorityBg}; color: #000; padding: 3px 10px; border-radius: 10px; font-size: 0.7rem; font-weight: bold;">
                                ${a.priority}
                            </span>
                        </div>
                        <p style="margin: 0; color: #ddd; font-size: 0.85rem; line-height: 1.5;">${a.action}</p>
                    </div>
                `;
            }).join('')}
            `;
        } else {
            container.innerHTML = `<p style="color: #DC3545; text-align: center; padding: 20px;">❌ ${data.error || 'Failed to generate AI analysis'}</p>`;
        }
    } catch (error) {
        container.innerHTML = `<p style="color: #DC3545; text-align: center; padding: 20px;">❌ Error: ${error.message}</p>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>✨</span> Generate AI Recommendations';
    }
}

// =====================================
// UI Helpers
// =====================================

function showLoading() {
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.add('active');
    document.getElementById('results-content').innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
        </div>
    `;
    // Scroll to results section
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function showContent(html) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.add('active');
    document.getElementById('results-content').innerHTML = html;
    // Scroll to results section
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function setTitle(title) {
    document.getElementById('results-title').textContent = title;
}

function showError(message) {
    showContent(`
        <div class="alert alert-critical">
            <strong>Error:</strong> ${message}
        </div>
    `);
}

function clearResults() {
    document.getElementById('results-section').classList.remove('active');
    // Scroll back to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function formatNumber(num) {
    if (num >= 10000000) return (num / 10000000).toFixed(1) + ' Cr';
    if (num >= 100000) return (num / 100000).toFixed(1) + ' L';
    if (num >= 1000) return (num / 1000).toFixed(1) + ' K';
    return num.toString();
}

function showDataTable(dataType, records) {
    if (!records || records.length === 0) {
        showContent('<p class="placeholder">No data available</p>');
        return;
    }

    setTitle(`📋 ${dataType.charAt(0).toUpperCase() + dataType.slice(1)} Data`);

    const headers = Object.keys(records[0]);

    showContent(`
        <p style="margin-bottom: 15px; color: #6c757d;">Showing ${records.length} records</p>
        <table class="data-table">
            <thead>
                <tr>
                    ${headers.slice(0, 7).map(h => `<th>${h}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
                ${records.slice(0, 20).map(r => `
                    <tr>
                        ${headers.slice(0, 7).map(h => `<td>${r[h] || '-'}</td>`).join('')}
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `);
}

function renderHotspotList(items, type) {
    if (!items.length) return '<p style="color: #666;">No data available</p>';

    return items.slice(0, 5).map(item => `
        <div class="alert ${type === 'coldspot' ? 'alert-critical' : 'alert-info'}">
            <strong>${item.region || item.state}</strong>
            ${item.zScore ? `<span style="float: right;">z-score: ${item.zScore.toFixed(2)}</span>` : ''}
            <br><span style="color: #6c757d; font-size: 0.85rem;">
                ${item.classification || (type === 'coldspot' ? 'Low coverage area' : 'High coverage area')}
            </span>
        </div>
    `).join('');
}

function renderAlerts(alerts) {
    if (!alerts.length) return '<p style="color: #666;">No alerts at this time</p>';

    return alerts.slice(0, 10).map(alert => `
        <div class="alert ${alert.severity === 'critical' ? 'alert-critical' : alert.severity === 'high' ? 'alert-warning' : 'alert-info'}">
            <strong>${alert.region}</strong> - ${alert.severity?.toUpperCase() || 'ALERT'}
            <br><span style="color: #6c757d; font-size: 0.85rem;">
                Deviation: ${(alert.percentageDeviation || 0).toFixed(1)}% | ${alert.direction || 'unusual pattern'}
            </span>
        </div>
    `).join('');
}

function renderTrends(regions) {
    if (!regions.length) return '<p style="color: #666;">No trend data available</p>';

    return `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Region</th>
                    <th>Trend</th>
                    <th>Monthly Change</th>
                </tr>
            </thead>
            <tbody>
                ${regions.slice(0, 15).map(r => `
                    <tr>
                        <td>${r.region}</td>
                        <td style="color: ${r.trend === 'increasing' ? '#138808' : r.trend === 'decreasing' ? '#DC3545' : '#6c757d'};">
                            ${r.trend === 'increasing' ? '📈' : r.trend === 'decreasing' ? '📉' : '➡️'} ${r.trend}
                        </td>
                        <td>${(r.monthlyChange || 0).toFixed(1)}%</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function renderBasicStats(records) {
    const stateStats = {};
    records.forEach(r => {
        if (!stateStats[r.state]) stateStats[r.state] = 0;
        stateStats[r.state] += (parseInt(r.age_18_greater) || 0);
    });

    const sorted = Object.entries(stateStats)
        .sort((a, b) => a[1] - b[1])
        .slice(0, 10);

    return `
        <h3 style="margin: 15px 0;">States with Lowest Enrollment (Potential Coldspots)</h3>
        <table class="data-table">
            <thead>
                <tr><th>State</th><th>Adult Enrollments</th></tr>
            </thead>
            <tbody>
                ${sorted.map(([state, count]) => `
                    <tr><td>${state}</td><td>${formatNumber(count)}</td></tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

// =====================================
// Initialize
// =====================================

document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    // Recheck status every 30 seconds
    setInterval(checkStatus, 30000);
});

function renderFlaggedRecords(records) {
    if (!records || records.length === 0) return '';

    // Get headers (exclude internal fields)
    const ignore = ['flagged_reason', 'risk_score'];
    const headers = Object.keys(records[0]).filter(k => !ignore.includes(k));

    // Limit columns
    const cols = headers.slice(0, 6);

    return `
        <h3 style="margin: 20px 0 15px; color: #DC3545;">🚩 Details: Flagged Records (${records.length})</h3>
        <p style="color: #6c757d; margin-bottom: 15px; font-size: 0.9rem;">
            The following records were flagged by the anomaly detection engine. Click "Details" to view full record.
        </p>
        <div style="overflow-x: auto; background: #FFFFFFFFF; border-radius: 8px; padding: 10px;">
            <table class="data-table" style="font-size: 0.9rem;">
                <thead>
                    <tr>
                        <th>Risk Score</th>
                        <th>Reason</th>
                        ${cols.map(h => `<th>${h}</th>`).join('')}
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    ${records.map((r, i) => `
                        <tr>
                            <td><span style="color: #DC3545; font-weight: bold;">${r.risk_score}</span></td>
                            <td>${r.flagged_reason}</td>
                            ${cols.map(h => `<td>${r[h] || '-'}</td>`).join('')}
                            <td>
                                <button onclick="showRecordDetails(${i})" style="background: #000080; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; color: #000; font-weight: bold;">
                                    View
                                </button>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
        
        <!-- Modal for details (hidden by default) -->
        <script>
            window.flaggedRecordsData = ${JSON.stringify(records)};
        </script>
    `;
}

function showRecordDetails(index) {
    const record = window.flaggedRecordsData[index];
    const html = `
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; display: flex; align-items: center; justify-content: center;">
            <div style="background: #F5F5F5; padding: 30px; border-radius: 10px; width: 90%; max-width: 600px; border: 1px solid #dee2e6; position: relative;">
                <button onclick="this.closest('div').parentElement.remove()" style="position: absolute; top: 15px; right: 20px; background: none; border: none; color: #1a1a2e; font-size: 24px; cursor: pointer;">✕</button>
                <h2 style="color: #DC3545; margin-bottom: 20px;">🚩 Record Details</h2>
                
                <div style="background: #dee2e6; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <strong style="color: #6c757d;">Risk Score:</strong>
                        <span style="color: #DC3545; font-weight: bold;">${record.risk_score}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <strong style="color: #6c757d;">Reason:</strong>
                        <span style="color: #1a1a2e;">${record.flagged_reason}</span>
                    </div>
                </div>
                
                <div style="max-height: 400px; overflow-y: auto;">
                    <table style="width: 100%; border-collapse: collapse;">
                        ${Object.entries(record).filter(([k]) => k !== 'flagged_reason' && k !== 'risk_score').map(([k, v]) => `
                            <tr>
                                <td style="padding: 8px; border-bottom: 1px solid #dee2e6; color: #000080;">${k}</td>
                                <td style="padding: 8px; border-bottom: 1px solid #dee2e6; color: #1a1a2e;">${v}</td>
                            </tr>
                        `).join('')}
                    </table>
                </div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', html);
}

function showAnalysisEvidence() {
    const records = window.lastMonitoringResults?.flagged_records;
    if (!records || records.length === 0) {
        alert("No specific data evidence available for this analysis.");
        return;
    }

    // Create modal wrapper around renderFlaggedRecords output
    const content = renderFlaggedRecords(records);

    const html = `
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 999; display: flex; align-items: center; justify-content: center;">
            <div style="background: #151515; padding: 40px; border-radius: 12px; width: 95%; max-width: 1000px; max-height: 90vh; overflow-y: auto; border: 1px solid #dee2e6; position: relative; box-shadow: 0 0 50px rgba(0,0,0,0.5);">
                <button onclick="this.closest('div').parentElement.remove()" style="position: absolute; top: 20px; right: 25px; background: none; border: none; color: #1a1a2e; font-size: 28px; cursor: pointer; opacity: 0.7;">✕</button>
                ${content}
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', html);
}

function toggleFindingDetails(index) {
    const el = document.getElementById(`finding-details-${index}`);
    if (el) {
        el.style.display = el.style.display === 'none' ? 'block' : 'none';
    }
}

function toggleMoreFindings() {
    const el = document.getElementById('more-findings');
    const btn = document.getElementById('show-more-findings-btn');
    if (el && btn) {
        if (el.style.display === 'none') {
            el.style.display = 'block';
            btn.innerHTML = '▲ Hide Additional Findings';
        } else {
            el.style.display = 'none';
            btn.innerHTML = btn.innerHTML.replace('Hide', 'Show');
        }
    }
}

function renderSimpleRecordTable(records) {
    if (!records || records.length === 0) return '';
    const headers = Object.keys(records[0]).filter(k => k !== 'flagged_reason' && k !== 'risk_score').slice(0, 4);

    return `
    <table style="width:100%; font-size: 0.8rem; border-collapse: collapse; margin-top: 5px;">
        <thead>
            <tr style="border-bottom: 1px solid #dee2e6; color: #6c757d;">
                ${headers.map(h => `<th style="text-align:left; padding:4px;">${h}</th>`).join('')}
            </tr>
        </thead>
        <tbody>
            ${records.map(r => `
                <tr>
                    ${headers.map(h => `<td style="padding:4px; color:#4a4a6a;">${r[h]}</td>`).join('')}
                </tr>
            `).join('')}
        </tbody>
    </table>`;
}
// =====================================
// Ridwan Features - Gender & Vulnerable Groups
// =====================================

async function loadGenderGapMap() {
    setTitle('🗺️ Gender Gap Map - India');

    try {
        const response = await fetch(`${API_BASE}/ridwan-gender/coverage?limit=500`);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to fetch data');
        }

        const stateAnalysis = data.data?.stateAnalysis || [];

        // Create a visual table-based map representation
        showContent(`
            <div class="alert alert-info">
                <strong>🗺️ Interactive Gender Gap Map:</strong> 
                Visual representation of gender coverage by state. Click on any state for details.
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 20px 0;">
                ${stateAnalysis.map(s => {
                    const gapPercent = s.genderGap * 100;
                    const color = gapPercent > 5 ? '#DC3545' : gapPercent > 3 ? '#ff9900' : gapPercent > 1 ? '#ffcc00' : '#138808';
                    return `
                        <div onclick="showStateDetails('${s.state}')" 
                            style="background: linear-gradient(135deg, ${color}22 0%, ${color}44 100%);
                            border: 2px solid ${color}; border-radius: 10px; padding: 15px; cursor: pointer;
                            transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.05)'" 
                            onmouseout="this.style.transform='scale(1)'">
                            <div style="font-weight: bold; color: white;">${s.state}</div>
                            <div style="font-size: 1.5rem; color: ${color};">${(s.genderGap * 100).toFixed(1)}%</div>
                            <div style="font-size: 0.8rem; color: #6c757d;">Gender Gap</div>
                            <div style="font-size: 0.7rem; color: #6c757d; margin-top: 5px;">
                                F: ${(s.femaleCoverageRatio * 100).toFixed(0)}% | M: ${(s.maleCoverageRatio * 100).toFixed(0)}%
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>

            <h3 style="margin: 25px 0 15px; color: #000080;">📊 Legend</h3>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <span><span style="display: inline-block; width: 20px; height: 20px; background: #DC3545; border-radius: 5px;"></span> Critical (>5%)</span>
                <span><span style="display: inline-block; width: 20px; height: 20px; background: #ff9900; border-radius: 5px;"></span> High (3-5%)</span>
                <span><span style="display: inline-block; width: 20px; height: 20px; background: #ffcc00; border-radius: 5px;"></span> Moderate (1-3%)</span>
                <span><span style="display: inline-block; width: 20px; height: 20px; background: #138808; border-radius: 5px;"></span> Low (<1%)</span>
            </div>
        `);

    } catch (error) {
        showError(`Failed to load map: ${error.message}`);
    }
}

async function showStateDetails(stateName) {
    showLoading();
    setTitle(`📍 ${stateName} - Gender Analysis`);

    try {
        const response = await fetch(`${API_BASE}/ridwan-gender/coverage?limit=500`);
        const data = await response.json();
        
        const stateData = data.data?.stateAnalysis?.find(s => s.state === stateName);
        const districts = data.data?.districtAnalysis?.filter(d => d.state === stateName) || [];

        if (!stateData) {
            showError('State data not found');
            return;
        }

        showContent(`
            <button onclick="loadGenderGapMap()" style="margin-bottom: 20px; padding: 8px 16px; background: #dee2e6; border: 1px solid #555; border-radius: 5px; color: white; cursor: pointer;">
                ← Back to Map
            </button>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${(stateData.femaleCoverageRatio * 100).toFixed(1)}%</div>
                    <div class="stat-label">Female Coverage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(stateData.maleCoverageRatio * 100).toFixed(1)}%</div>
                    <div class="stat-label">Male Coverage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: ${stateData.genderGap > 0.03 ? '#DC3545' : '#138808'};">
                        ${(stateData.genderGap * 100).toFixed(1)}%
                    </div>
                    <div class="stat-label">Gender Gap</div>
                </div>
            </div>

            <h3 style="margin: 25px 0 15px; color: #000080;">District Analysis</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>District</th>
                        <th>Female %</th>
                        <th>Male %</th>
                        <th>Gap</th>
                    </tr>
                </thead>
                <tbody>
                    ${districts.slice(0, 15).map(d => `
                        <tr>
                            <td>${d.district}</td>
                            <td>${(d.femaleCoverageRatio * 100).toFixed(1)}%</td>
                            <td>${(d.maleCoverageRatio * 100).toFixed(1)}%</td>
                            <td style="color: ${d.genderGap > 0.03 ? '#DC3545' : '#138808'};">
                                ${(d.genderGap * 100).toFixed(1)}%
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `);
    } catch (error) {
        showError(`Failed to load state details: ${error.message}`);
    }
}

async function loadVulnerableGroups() {
    setTitle('👵 Vulnerable Groups Inclusion Tracker');

    try {
        const response = await fetch(`${API_BASE}/ridwan-vulnerable/analysis?limit=500`);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to fetch data');
        }

        const { summary, stateAnalysis, highRiskByGroup } = data.data;

        showContent(`
            <div class="alert alert-info">
                <strong>👵👦👨‍🦽 Multi-Vulnerable Group Analysis:</strong> 
                Tracking inclusion for children (0-5), youth (5-17), adults, elderly (60+), and disabled populations.
            </div>

            <div class="stats-grid">
                <div class="stat-card" style="background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);">
                    <div class="stat-value">${formatNumber(summary.nationalStats.children)}</div>
                    <div class="stat-label">Children (0-5)</div>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #2196F3 0%, #1565C0 100%);">
                    <div class="stat-value">${formatNumber(summary.nationalStats.youth)}</div>
                    <div class="stat-label">Youth (5-17)</div>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #9C27B0 0%, #6A1B9A 100%);">
                    <div class="stat-value">${formatNumber(summary.nationalStats.elderly)}</div>
                    <div class="stat-label">Elderly (60+)</div>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #FF9800 0%, #E65100 100%);">
                    <div class="stat-value">${formatNumber(summary.nationalStats.disabled)}</div>
                    <div class="stat-label">Disabled</div>
                </div>
            </div>

            <div class="stat-card" style="margin: 20px 0; text-align: center;">
                <div class="stat-value" style="font-size: 3rem; color: ${summary.avgInclusionScore > 70 ? '#138808' : summary.avgInclusionScore > 50 ? '#ffcc00' : '#DC3545'};">
                    ${summary.avgInclusionScore.toFixed(0)}
                </div>
                <div class="stat-label">National Inclusion Score (out of 100)</div>
            </div>

            <!-- Charts Section -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 25px 0;">
                <div style="background: rgba(0,0,0,0.3); border-radius: 15px; padding: 20px;">
                    <h3 style="margin: 0 0 15px; color: #000080;">📊 Enrollment by Vulnerable Group</h3>
                    <canvas id="vulnerableBarChart" height="200"></canvas>
                </div>
                <div style="background: rgba(0,0,0,0.3); border-radius: 15px; padding: 20px;">
                    <h3 style="margin: 0 0 15px; color: #ff9900;">🎯 Inclusion Score Radar</h3>
                    <canvas id="inclusionRadarChart" height="200"></canvas>
                </div>
            </div>

            <h3 style="margin: 25px 0 15px; color: #DC3545;">⚠️ High-Risk Districts by Group</h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                <div style="background: rgba(76,175,80,0.1); border: 1px solid #4CAF50; border-radius: 10px; padding: 15px;">
                    <h4 style="color: #4CAF50;">👦 Children (0-5) - At Risk</h4>
                    ${(highRiskByGroup?.children || []).slice(0, 5).map(d => `
                        <div style="padding: 5px 0; border-bottom: 1px solid #dee2e6;">
                            ${d.district}, ${d.state}
                            <span style="float: right; color: #DC3545;">${d.groups?.children?.coverageScore || 0}%</span>
                        </div>
                    `).join('') || '<p style="color: #666;">No high-risk districts</p>'}
                </div>

                <div style="background: rgba(156,39,176,0.1); border: 1px solid #9C27B0; border-radius: 10px; padding: 15px;">
                    <h4 style="color: #9C27B0;">👵 Elderly (60+) - At Risk</h4>
                    ${(highRiskByGroup?.elderly || []).slice(0, 5).map(d => `
                        <div style="padding: 5px 0; border-bottom: 1px solid #dee2e6;">
                            ${d.district}, ${d.state}
                            <span style="float: right; color: #DC3545;">${d.groups?.elderly?.coverageScore || 0}%</span>
                        </div>
                    `).join('') || '<p style="color: #666;">No high-risk districts</p>'}
                </div>

                <div style="background: rgba(255,152,0,0.1); border: 1px solid #FF9800; border-radius: 10px; padding: 15px;">
                    <h4 style="color: #FF9800;">👨‍🦽 Disabled - At Risk</h4>
                    ${(highRiskByGroup?.disabled || []).slice(0, 5).map(d => `
                        <div style="padding: 5px 0; border-bottom: 1px solid #dee2e6;">
                            ${d.district}, ${d.state}
                            <span style="float: right; color: #DC3545;">${d.groups?.disabled?.coverageScore || 0}%</span>
                        </div>
                    `).join('') || '<p style="color: #666;">No high-risk districts</p>'}
                </div>
            </div>

            <h3 style="margin: 25px 0 15px; color: #000080;">📊 State-wise Inclusion Scores</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>State</th>
                        <th>Children</th>
                        <th>Elderly</th>
                        <th>Disabled</th>
                        <th>Overall</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
                    ${stateAnalysis.slice(0, 10).map(s => `
                        <tr>
                            <td>${s.state}</td>
                            <td>${s.groups.children.coverageScore}%</td>
                            <td>${s.groups.elderly.coverageScore}%</td>
                            <td>${s.groups.disabled.coverageScore}%</td>
                            <td style="font-weight: bold;">${s.overallInclusionScore}%</td>
                            <td>
                                <span style="padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;
                                    background: ${s.riskLevel === 'CRITICAL' ? '#ff0000' : s.riskLevel === 'HIGH' ? '#ff9900' : s.riskLevel === 'MODERATE' ? '#ffcc00' : '#138808'}; 
                                    color: ${s.riskLevel === 'LOW' ? '#000' : '#1a1a2e'};">
                                    ${s.riskLevel}
                                </span>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `);

        // Create Bar Chart for Vulnerable Group Enrollment
        const barCtx = document.getElementById('vulnerableBarChart').getContext('2d');
        new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: ['Children (0-5)', 'Youth (5-17)', 'Adults (18+)', 'Elderly (60+)', 'Disabled'],
                datasets: [{
                    label: 'Enrollments',
                    data: [
                        summary.nationalStats.children,
                        summary.nationalStats.youth,
                        summary.nationalStats.adults,
                        summary.nationalStats.elderly,
                        summary.nationalStats.disabled
                    ],
                    backgroundColor: [
                        'rgba(76, 175, 80, 0.8)',
                        'rgba(33, 150, 243, 0.8)',
                        'rgba(0, 212, 255, 0.8)',
                        'rgba(156, 39, 176, 0.8)',
                        'rgba(255, 152, 0, 0.8)'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { ticks: { color: '#6c757d' }, grid: { color: '#dee2e6' } },
                    y: { ticks: { color: '#6c757d' }, grid: { color: '#dee2e6' } }
                }
            }
        });

        // Create Radar Chart for Inclusion Scores (top 5 states)
        const radarCtx = document.getElementById('inclusionRadarChart').getContext('2d');
        const top5States = stateAnalysis.slice(0, 5);
        new Chart(radarCtx, {
            type: 'radar',
            data: {
                labels: ['Children', 'Youth', 'Adults', 'Elderly', 'Disabled'],
                datasets: top5States.map((s, i) => ({
                    label: s.state.substring(0, 8),
                    data: [
                        s.groups.children.coverageScore,
                        s.groups.youth.coverageScore,
                        s.groups.adults.coverageScore,
                        s.groups.elderly.coverageScore,
                        s.groups.disabled.coverageScore
                    ],
                    borderColor: ['#ff6384', '#36a2eb', '#ffce56', '#4bc0c0', '#9966ff'][i],
                    backgroundColor: ['#ff638440', '#36a2eb40', '#ffce5640', '#4bc0c040', '#9966ff40'][i],
                    borderWidth: 2
                }))
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { 
                        position: 'bottom',
                        labels: { color: '#1a1a2e', boxWidth: 12, padding: 10 } 
                    }
                },
                scales: {
                    r: {
                        angleLines: { color: '#dee2e6' },
                        grid: { color: '#dee2e6' },
                        pointLabels: { color: '#6c757d' },
                        ticks: { color: '#6c757d', backdropColor: 'transparent' },
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                }
            }
        });

    } catch (error) {
        showError(`Failed to load vulnerable groups: ${error.message}`);
    }
}

async function loadImpactSimulator() {
    setTitle('💰 Impact Simulator - ROI Calculator');

    showContent(`
        <div class="alert alert-info">
            <strong>💰 Intervention Impact Simulator:</strong> 
            Calculate projected enrollments and welfare benefits for different intervention strategies.
        </div>

        <div style="background: rgba(0,212,255,0.1); border: 1px solid #000080; border-radius: 10px; padding: 20px; margin: 20px 0;">
            <h3 style="color: #000080; margin-top: 0;">🎯 Configure Intervention</h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div>
                    <label style="display: block; margin-bottom: 5px; color: #6c757d;">Intervention Type</label>
                    <select id="sim-intervention" style="width: 100%; padding: 10px; background: #1a1a2e; border: 1px solid #555; border-radius: 5px; color: white;">
                        <option value="CAMP">Women-Only Camps</option>
                        <option value="MOBILE_VAN">Mobile Enrollment Vans</option>
                        <option value="DOORSTEP">Doorstep Service</option>
                        <option value="AWARENESS_CAMPAIGN">Awareness Campaign</option>
                    </select>
                </div>
                <div>
                    <label style="display: block; margin-bottom: 5px; color: #6c757d;">Quantity / Number</label>
                    <input type="number" id="sim-quantity" value="10" min="1" max="100" 
                        style="width: 100%; padding: 10px; background: #1a1a2e; border: 1px solid #555; border-radius: 5px; color: white;">
                </div>
                <div>
                    <label style="display: block; margin-bottom: 5px; color: #6c757d;">Days (for vans)</label>
                    <input type="number" id="sim-days" value="5" min="1" max="30" 
                        style="width: 100%; padding: 10px; background: #1a1a2e; border: 1px solid #555; border-radius: 5px; color: white;">
                </div>
                <div>
                    <label style="display: block; margin-bottom: 5px; color: #6c757d;">Target Group</label>
                    <select id="sim-target" style="width: 100%; padding: 10px; background: #1a1a2e; border: 1px solid #555; border-radius: 5px; color: white;">
                        <option value="WOMEN">Women</option>
                        <option value="ELDERLY">Elderly</option>
                        <option value="CHILDREN">Children</option>
                        <option value="ALL">All Groups</option>
                    </select>
                </div>
            </div>
            
            <button onclick="runSimulation()" 
                style="margin-top: 20px; padding: 12px 30px; background: linear-gradient(135deg, #000080 0%, #0099cc 100%); 
                border: none; border-radius: 8px; color: white; font-size: 1rem; cursor: pointer; width: 100%;">
                🚀 Calculate Impact
            </button>
        </div>

        <div id="sim-results" style="display: none;">
            <!-- Results will be displayed here -->
        </div>

        <h3 style="margin: 25px 0 15px; color: #ffcc00;">🎨 Quick Presets</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">
            <button onclick="runPreset('camp5')" class="preset-btn" style="padding: 15px; background: #1a1a2e; border: 1px solid #555; border-radius: 8px; color: white; cursor: pointer; text-align: left;">
                <strong>Small District</strong><br>
                <span style="color: #6c757d; font-size: 0.9rem;">5 camps, women focus</span>
            </button>
            <button onclick="runPreset('mixed')" class="preset-btn" style="padding: 15px; background: #1a1a2e; border: 1px solid #555; border-radius: 8px; color: white; cursor: pointer; text-align: left;">
                <strong>Medium District</strong><br>
                <span style="color: #6c757d; font-size: 0.9rem;">10 camps + mobile vans</span>
            </button>
            <button onclick="runPreset('intensive')" class="preset-btn" style="padding: 15px; background: #1a1a2e; border: 1px solid #555; border-radius: 8px; color: white; cursor: pointer; text-align: left;">
                <strong>Large District</strong><br>
                <span style="color: #6c757d; font-size: 0.9rem;">20 camps, intensive drive</span>
            </button>
        </div>
    `);
}

async function runSimulation() {
    const intervention = document.getElementById('sim-intervention').value;
    const quantity = parseInt(document.getElementById('sim-quantity').value);
    const days = parseInt(document.getElementById('sim-days').value);
    const targetGroup = document.getElementById('sim-target').value;

    const resultsDiv = document.getElementById('sim-results');
    resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    resultsDiv.style.display = 'block';

    try {
        const response = await fetch(`${API_BASE}/ridwan-simulate/intervention`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ intervention, quantity, days, targetGroup })
        });
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Simulation failed');
        }

        const result = data.data;
        const benefits = result.benefitsUnlocked;

        resultsDiv.innerHTML = `
            <h3 style="margin: 20px 0 15px; color: #138808;">📊 Simulation Results</h3>
            
            <div class="stats-grid">
                <div class="stat-card" style="background: linear-gradient(135deg, #000080 0%, #0099cc 100%);">
                    <div class="stat-value">${formatNumber(result.projections.totalReach)}</div>
                    <div class="stat-label">People Reached</div>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #138808 0%, #00cc6a 100%);">
                    <div class="stat-value">${formatNumber(result.projections.projectedEnrollments)}</div>
                    <div class="stat-label">Projected Enrollments</div>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #ff9900 0%, #cc7700 100%);">
                    <div class="stat-value">₹${formatNumber(result.projections.totalCost)}</div>
                    <div class="stat-label">Total Cost</div>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #9C27B0 0%, #6A1B9A 100%);">
                    <div class="stat-value">${benefits.totalAnnualFormatted}</div>
                    <div class="stat-label">Annual Benefits Unlocked</div>
                </div>
            </div>

            <div class="stats-grid" style="margin-top: 15px;">
                <div class="stat-card">
                    <div class="stat-value" style="color: ${parseFloat(result.roi.value) > 100 ? '#138808' : '#ffcc00'};">
                        ${result.roi.value}
                    </div>
                    <div class="stat-label">Return on Investment</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${result.roi.breakEvenMonths}</div>
                    <div class="stat-label">Months to Break Even</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">₹${result.projections.costPerEnrollment}</div>
                    <div class="stat-label">Cost per Enrollment</div>
                </div>
            </div>

            <div class="alert ${result.roi.recommendation === 'HIGHLY RECOMMENDED' ? 'alert-info' : 'alert-warning'}" style="margin-top: 20px;">
                <strong>💡 Recommendation:</strong> ${result.roi.recommendation}
            </div>

            <h4 style="margin: 20px 0 10px; color: #000080;">💰 Welfare Benefits Breakdown</h4>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Benefit</th>
                        <th>Beneficiaries</th>
                        <th>Annual Value</th>
                    </tr>
                </thead>
                <tbody>
                    ${benefits.benefits.map(b => `
                        <tr>
                            <td>${b.name}</td>
                            <td>${formatNumber(b.beneficiaries)} (${b.eligiblePercent}%)</td>
                            <td>₹${formatNumber(b.annualValue)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;

    } catch (error) {
        resultsDiv.innerHTML = `
            <div class="alert alert-critical">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    }
}

async function runPreset(preset) {
    let intervention, quantity, days, targetGroup;

    switch (preset) {
        case 'camp5':
            intervention = 'CAMP';
            quantity = 5;
            days = 1;
            targetGroup = 'WOMEN';
            break;
        case 'mixed':
            intervention = 'CAMP';
            quantity = 10;
            days = 5;
            targetGroup = 'ALL';
            break;
        case 'intensive':
            intervention = 'CAMP';
            quantity = 20;
            days = 10;
            targetGroup = 'WOMEN';
            break;
    }

    // Update form values
    document.getElementById('sim-intervention').value = intervention;
    document.getElementById('sim-quantity').value = quantity;
    document.getElementById('sim-days').value = days;
    document.getElementById('sim-target').value = targetGroup;

    // Run simulation
    await runSimulation();
}