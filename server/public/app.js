/**
 * UIDAI Analytics Dashboard - Frontend JavaScript
 */

const API_BASE = 'http://localhost:3001/api';
const ML_API_BASE = 'http://localhost:8000';

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
        
        <h3 style="margin: 20px 0 15px; color: #00d4ff;">📊 Data Sources Status</h3>
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
                        <div class="stat-value" style="color: #ff4444;">${summary?.coldspotCount || 0}</div>
                        <div class="stat-label">Coldspots (Low Coverage)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #00ff88;">${summary?.hotspotCount || 0}</div>
                        <div class="stat-label">Hotspots (High Coverage)</div>
                    </div>
                </div>
                
                <h3 style="margin: 20px 0 15px; color: #ff6666;">🔴 Priority Coldspots (Need Intervention)</h3>
                ${renderHotspotList(coldspots || [], 'coldspot')}
                
                <h3 style="margin: 20px 0 15px; color: #00ff88;">🟢 Top Performing Hotspots</h3>
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
                        <div class="stat-value" style="color: #ffaa00;">${summary?.totalAnomalies || 0}</div>
                        <div class="stat-label">Total Anomalies</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #ff4444;">${summary?.critical || 0}</div>
                        <div class="stat-label">Critical</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #ffaa00;">${summary?.high || 0}</div>
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
                <p style="margin-top: 15px; color: #aaa;">
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

    try {
        const enrolment = await fetch(`${API_BASE}/enrolment?limit=200`).then(r => r.json());
        const records = enrolment.records || [];

        // Calculate gender stats by state (simulated since real data may not have gender)
        const stateStats = {};
        records.forEach(r => {
            if (!stateStats[r.state]) {
                stateStats[r.state] = { total: 0, records: 0 };
            }
            stateStats[r.state].total += (parseInt(r.age_18_greater) || 0);
            stateStats[r.state].records++;
        });

        const stateList = Object.entries(stateStats)
            .map(([state, data]) => ({ state, ...data }))
            .sort((a, b) => b.total - a.total)
            .slice(0, 15);

        showContent(`
            <div class="alert alert-info">
                <strong>Gender Inclusion Analysis:</strong> 
                Monitoring female Aadhaar enrollment coverage across districts.
            </div>
            
            <h3 style="margin: 20px 0 15px;">📊 State-wise Enrollment Summary</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>State</th>
                        <th>Adult Enrollments (18+)</th>
                        <th>Records</th>
                    </tr>
                </thead>
                <tbody>
                    ${stateList.map(s => `
                        <tr>
                            <td>${s.state}</td>
                            <td>${formatNumber(s.total)}</td>
                            <td>${s.records}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
            
            <div class="alert alert-warning" style="margin-top: 20px;">
                <strong>Note:</strong> Full gender disaggregated data requires ML backend connection.
            </div>
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
                        background: linear-gradient(135deg, #7b2ff7, #00d4ff);
                        color: #fff;
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
                <p style="margin-top: 15px; color: #aaa;">
                    Run: <code style="background: #333; padding: 5px 10px; border-radius: 5px;">
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
    setTitle('📅 Enrollment Forecast');

    try {
        const response = await fetch(`${API_BASE}/hotspots/trends?limit=500`);
        const data = await response.json();

        if (data.success && data.data) {
            const { regionalTrends } = data.data;

            showContent(`
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" style="color: #00ff88;">${regionalTrends?.summary?.increasing || 0}</div>
                        <div class="stat-label">Increasing Trend</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${regionalTrends?.summary?.stable || 0}</div>
                        <div class="stat-label">Stable</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: #ff4444;">${regionalTrends?.summary?.decreasing || 0}</div>
                        <div class="stat-label">Decreasing Trend</div>
                    </div>
                </div>
                
                <h3 style="margin: 20px 0 15px;">📈 Regional Trends</h3>
                ${renderTrends(regionalTrends?.regions || [])}
            `);
        } else {
            showContent(`
                <div class="alert alert-info">
                    <strong>Enrollment Forecast:</strong> Time-series analysis for enrollment predictions.
                </div>
                <p style="margin-top: 15px; color: #aaa;">
                    This feature uses seasonal decomposition to predict future enrollment patterns.
                </p>
            `);
        }
    } catch (error) {
        showError(`Failed to load forecast: ${error.message}`);
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
                            <label style="display: block; margin-bottom: 8px; color: #00d4ff;">Monitoring Type</label>
                            <select id="monitoring-intent" class="monitoring-select">
                                ${intents.map(i => `<option value="${i.id}">${i.display_name}</option>`).join('')}
                            </select>
                        </div>
                        <div class="stat-card" style="text-align: left;">
                            <label style="display: block; margin-bottom: 8px; color: #00d4ff;">Focus Area</label>
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
                            <label style="display: block; margin-bottom: 8px; color: #00d4ff;">Time Period</label>
                            <select id="monitoring-period" class="monitoring-select">
                                <option value="today">Today</option>
                                <option value="last_7_days">Last 7 Days</option>
                                <option value="this_month">This Month</option>
                            </select>
                        </div>
                        <div class="stat-card" style="text-align: left;">
                            <label style="display: block; margin-bottom: 8px; color: #00d4ff;">Vigilance Level</label>
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
                <p style="margin-top: 15px; color: #aaa;">
                    Start the ML backend with:<br>
                    <code style="background: #333; padding: 5px 10px; border-radius: 5px; display: inline-block; margin-top: 10px;">
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
                    <div style="margin-top: 10px; background: #333; border-radius: 10px; overflow: hidden;">
                        <div style="width: ${status.progress}%; height: 6px; background: linear-gradient(90deg, #00d4ff, #7b2ff7);"></div>
                    </div>
                    <span style="font-size: 0.85rem; color: #888;">${status.progress}% complete</span>
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
    const riskColor = results.risk.risk_level === 'Low' ? '#00ff88' :
        results.risk.risk_level === 'Medium' ? '#ffaa00' : '#ff4444';

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
                <div class="stat-value" style="color: #ffaa00;">${results.flagged_for_review}</div>
                <div class="stat-label">Flagged for Review</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #00ff88;">${results.cleared}</div>
                <div class="stat-label">Cleared</div>
            </div>
        </div>

        <h3 style="margin: 20px 0 15px; color: #00d4ff;">📋 Monitoring Summary</h3>
        <div class="alert alert-info">${results.summary}</div>

        <h3 style="margin: 20px 0 15px; color: #ffaa00;">⚠️ Key Observations (${results.findings.length})</h3>
        ${results.findings.slice(0, 5).map((f, i) => {
        const severityColor = f.severity === 'High' ? '#ff4444' : f.severity === 'Medium' ? '#ffaa00' : '#00d4ff';
        const bgColor = f.severity === 'High' ? 'rgba(255,68,68,0.15)' : f.severity === 'Medium' ? 'rgba(255,170,0,0.15)' : 'rgba(0,212,255,0.15)';
        const location = f.location || f.state || 'National';
        return `
            <div class="alert" style="border-left: 4px solid ${severityColor}; background: ${bgColor};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <strong style="flex: 1;">${f.title}</strong>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <span style="background: rgba(100,100,255,0.3); color: #8888ff; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem;">📍 ${location}</span>
                        <span style="background: ${severityColor}; color: #000; padding: 2px 10px; border-radius: 10px; font-size: 0.75rem; font-weight: bold;">${f.severity}</span>
                    </div>
                </div>
                <span style="color: #ccc;">${f.description}</span>
                
                <div style="margin-top: 10px;">
                    <button onclick="toggleFindingDetails(${i})" style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8rem;">
                        ▼ Details
                    </button>
                    <div id="finding-details-${i}" style="display: none; margin-top: 10px; background: rgba(0,0,0,0.4); padding: 15px; border-radius: 6px; border-left: 3px solid ${severityColor};">
                        <h4 style="color: #ffaa00; margin: 0 0 8px 0; font-size: 0.95rem;">📊 Impact Assessment - ${location}</h4>
                        <p style="margin: 0; color: #ccc; font-size: 0.9rem;">${f.details || 'Pattern detected requiring investigation within the focus area.'}</p>
                    </div>
                </div>
            </div>
        `;
    }).join('')}
        
        ${results.findings.length > 5 ? `
            <button id="show-more-findings-btn" onclick="toggleMoreFindings()" style="background: rgba(255,170,0,0.2); border: 1px solid #ffaa00; color: #ffaa00; padding: 10px 20px; border-radius: 6px; cursor: pointer; width: 100%; margin: 10px 0;">
                ▼ Show ${results.findings.length - 5} More Findings
            </button>
            <div id="more-findings" style="display: none;">
                ${results.findings.slice(5).map((f, i) => `
                    <div class="alert" style="border-left: 4px solid ${f.severity === 'High' ? '#ff4444' : f.severity === 'Medium' ? '#ffaa00' : '#00d4ff'}; background: ${f.severity === 'High' ? 'rgba(255,68,68,0.15)' : f.severity === 'Medium' ? 'rgba(255,170,0,0.15)' : 'rgba(0,212,255,0.15)'}; margin-top: 10px;">
                        <strong>${f.title}</strong> <span style="float: right; color: ${f.severity === 'High' ? '#ff4444' : f.severity === 'Medium' ? '#ffaa00' : '#00d4ff'}; font-weight: bold;">${f.severity}</span>
                        <br><span style="color: #ccc;">${f.description}</span>
                    </div>
                `).join('')}
            </div>
        ` : ''}

        <div style="background: linear-gradient(135deg, rgba(139,92,246,0.2) 0%, rgba(59,130,246,0.2) 100%); border: 1px solid rgba(139,92,246,0.4); border-radius: 12px; padding: 20px; margin-top: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <div>
                    <h3 style="margin: 0; color: #a78bfa;">🤖 AI-Powered Analysis</h3>
                    <p style="margin: 5px 0 0; color: #888; font-size: 0.8rem;">Model: <strong style="color: #8b5cf6;">llama-3.3-70b-versatile</strong> via Groq LPU</p>
                </div>
                <button id="regenerate-ai-btn" onclick="regenerateAIAnalysis()" style="background: linear-gradient(135deg, #8b5cf6, #6366f1); border: none; color: #fff; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-weight: 600; display: flex; align-items: center; gap: 8px; transition: all 0.3s;">
                    <span>✨</span> Generate AI Recommendations
                </button>
            </div>
            <div id="ai-recommendations-container">
                <p style="color: #888; text-align: center; padding: 20px;">Click the button above to generate unique AI-powered recommendations based on current findings.</p>
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
        container.innerHTML = '<p style="color: #ff4444; text-align: center;">No analysis results available. Run monitoring first.</p>';
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
                    <div style="background: rgba(0,212,255,0.1); border-left: 3px solid #00d4ff; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
                        <h4 style="color: #00d4ff; margin: 0 0 8px;">📋 AI Summary</h4>
                        <p style="color: #ccc; margin: 0; font-size: 0.9rem;">${data.summary}</p>
                    </div>
                ` : ''}
                
                <h4 style="color: #00ff88; margin: 15px 0 10px;">✅ Recommended Actions (${data.recommended_actions?.length || 0})</h4>
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
                const priorityBg = a.priority === 'Urgent' ? '#ff4444' : a.priority === 'High' ? '#ffaa00' : '#00ff88';
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
                                    ${a.target_region ? `<span style="background: rgba(100,100,255,0.3); color: #8888ff; padding: 2px 8px; border-radius: 8px;">📍 ${a.target_region}</span>` : ''}
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
            container.innerHTML = `<p style="color: #ff4444; text-align: center; padding: 20px;">❌ ${data.error || 'Failed to generate AI analysis'}</p>`;
        }
    } catch (error) {
        container.innerHTML = `<p style="color: #ff4444; text-align: center; padding: 20px;">❌ Error: ${error.message}</p>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>✨</span> Generate AI Recommendations';
    }
}

// =====================================
// UI Helpers
// =====================================

function showLoading() {
    document.getElementById('results-section').classList.add('active');
    document.getElementById('results-content').innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
        </div>
    `;
}

function showContent(html) {
    document.getElementById('results-section').classList.add('active');
    document.getElementById('results-content').innerHTML = html;
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
        <p style="margin-bottom: 15px; color: #aaa;">Showing ${records.length} records</p>
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
            <br><span style="color: #aaa; font-size: 0.85rem;">
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
            <br><span style="color: #aaa; font-size: 0.85rem;">
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
                        <td style="color: ${r.trend === 'increasing' ? '#00ff88' : r.trend === 'decreasing' ? '#ff4444' : '#aaa'};">
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
        <h3 style="margin: 20px 0 15px; color: #ff4444;">🚩 Details: Flagged Records (${records.length})</h3>
        <p style="color: #aaa; margin-bottom: 15px; font-size: 0.9rem;">
            The following records were flagged by the anomaly detection engine. Click "Details" to view full record.
        </p>
        <div style="overflow-x: auto; background: #222; border-radius: 8px; padding: 10px;">
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
                            <td><span style="color: #ff4444; font-weight: bold;">${r.risk_score}</span></td>
                            <td>${r.flagged_reason}</td>
                            ${cols.map(h => `<td>${r[h] || '-'}</td>`).join('')}
                            <td>
                                <button onclick="showRecordDetails(${i})" style="background: #00d4ff; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; color: #000; font-weight: bold;">
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
            <div style="background: #1a1a1a; padding: 30px; border-radius: 10px; width: 90%; max-width: 600px; border: 1px solid #444; position: relative;">
                <button onclick="this.closest('div').parentElement.remove()" style="position: absolute; top: 15px; right: 20px; background: none; border: none; color: #fff; font-size: 24px; cursor: pointer;">✕</button>
                <h2 style="color: #ff4444; margin-bottom: 20px;">🚩 Record Details</h2>
                
                <div style="background: #333; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <strong style="color: #aaa;">Risk Score:</strong>
                        <span style="color: #ff4444; font-weight: bold;">${record.risk_score}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <strong style="color: #aaa;">Reason:</strong>
                        <span style="color: #fff;">${record.flagged_reason}</span>
                    </div>
                </div>
                
                <div style="max-height: 400px; overflow-y: auto;">
                    <table style="width: 100%; border-collapse: collapse;">
                        ${Object.entries(record).filter(([k]) => k !== 'flagged_reason' && k !== 'risk_score').map(([k, v]) => `
                            <tr>
                                <td style="padding: 8px; border-bottom: 1px solid #333; color: #00d4ff;">${k}</td>
                                <td style="padding: 8px; border-bottom: 1px solid #333; color: #fff;">${v}</td>
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
            <div style="background: #151515; padding: 40px; border-radius: 12px; width: 95%; max-width: 1000px; max-height: 90vh; overflow-y: auto; border: 1px solid #333; position: relative; box-shadow: 0 0 50px rgba(0,0,0,0.5);">
                <button onclick="this.closest('div').parentElement.remove()" style="position: absolute; top: 20px; right: 25px; background: none; border: none; color: #fff; font-size: 28px; cursor: pointer; opacity: 0.7;">✕</button>
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
            <tr style="border-bottom: 1px solid #444; color: #888;">
                ${headers.map(h => `<th style="text-align:left; padding:4px;">${h}</th>`).join('')}
            </tr>
        </thead>
        <tbody>
            ${records.map(r => `
                <tr>
                    ${headers.map(h => `<td style="padding:4px; color:#ccc;">${r[h]}</td>`).join('')}
                </tr>
            `).join('')}
        </tbody>
    </table>`;
}
