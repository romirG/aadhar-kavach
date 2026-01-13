/**
 * API Service for UIDAI Dashboard
 * Connects to Express backend (port 3001) and ML backend (via proxy)
 */

const API_BASE = 'http://localhost:3001/api';

// ============== Data Fetching Functions ==============

export async function fetchEnrolmentData(options: { limit?: number; offset?: number; state?: string } = {}) {
    const { limit = 1000, offset = 0, state } = options;
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (state) params.append('state', state);

    const response = await fetch(`${API_BASE}/enrolment?${params}`);
    if (!response.ok) throw new Error('Failed to fetch enrolment data');
    return response.json();
}

export async function fetchDemographicData(options: { limit?: number; offset?: number } = {}) {
    const { limit = 1000, offset = 0 } = options;
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });

    const response = await fetch(`${API_BASE}/demographic?${params}`);
    if (!response.ok) throw new Error('Failed to fetch demographic data');
    return response.json();
}

export async function fetchBiometricData(options: { limit?: number; offset?: number } = {}) {
    const { limit = 1000, offset = 0 } = options;
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });

    const response = await fetch(`${API_BASE}/biometric?${params}`);
    if (!response.ok) throw new Error('Failed to fetch biometric data');
    return response.json();
}

// ============== ML Backend Functions ==============

export async function fetchMLDatasets() {
    const response = await fetch(`${API_BASE}/ml/datasets`);
    if (!response.ok) throw new Error('Failed to fetch ML datasets');
    return response.json();
}

export async function startMLAnalysis(datasetId: string, limit: number = 500) {
    const response = await fetch(`${API_BASE}/ml/datasets/${datasetId}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit })
    });
    if (!response.ok) throw new Error('Failed to start ML analysis');
    return response.json();
}

export async function getAnalysisStatus(jobId: string) {
    const response = await fetch(`${API_BASE}/ml/analysis/${jobId}/status`);
    if (!response.ok) throw new Error('Failed to get analysis status');
    return response.json();
}

export async function getAnalysisResults(jobId: string) {
    const response = await fetch(`${API_BASE}/ml/analysis/${jobId}/results`);
    if (!response.ok) throw new Error('Failed to get analysis results');
    return response.json();
}

export async function getAnalysisVisualizations(jobId: string) {
    const response = await fetch(`${API_BASE}/ml/analysis/${jobId}/visualizations`);
    if (!response.ok) throw new Error('Failed to get visualizations');
    return response.json();
}

export async function getAuditorSummary(jobId: string) {
    const response = await fetch(`${API_BASE}/ml/analysis/${jobId}/summary`);
    if (!response.ok) throw new Error('Failed to get auditor summary');
    return response.json();
}

// ============== AI Recommendation Functions ==============

export async function getAIRecommendation(prompt: string, data?: unknown) {
    const response = await fetch(`${API_BASE}/ai/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, data })
    });
    if (!response.ok) throw new Error('Failed to get AI recommendation');
    return response.json();
}
