/**
 * API Service for UIDAI Dashboard
 * Connects to Express backend (port 3001) and ML backend (port 8000)
 */

const API_BASE = 'http://localhost:3001/api';
const ML_API_BASE = 'http://localhost:8000';

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

// ============== Intent-Based Monitoring API (NEW) ==============

export interface MonitoringIntent {
    id: string;
    display_name: string;
    description: string;
}

export interface VigilanceLevel {
    id: string;
    name: string;
    description: string;
}

export interface MonitoringRequest {
    intent: string;
    focus_area?: string;
    time_period: 'today' | 'last_7_days' | 'this_month';
    vigilance: 'routine' | 'standard' | 'enhanced' | 'maximum';
    record_limit?: number;
}

export interface MonitoringJobResponse {
    job_id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    message: string;
    estimated_time: string;
}

export interface RiskSummary {
    risk_index: number;
    risk_level: string;
    confidence: string;
}

export interface Finding {
    title: string;
    description: string;
    severity: string;
    location?: string;
}

export interface ActionItem {
    action: string;
    priority: string;
}

export interface MonitoringResults {
    job_id: string;
    status: string;
    summary: string;
    risk: RiskSummary;
    findings: Finding[];
    recommended_actions: ActionItem[];
    records_analyzed: number;
    flagged_for_review: number;
    cleared: number;
    analysis_scope: string;
    time_period: string;
    completed_at: string;
    report_id: string;
}

export interface StatusResponse {
    job_id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress: number;
    message: string;
    started_at?: string;
    completed_at?: string;
}

/**
 * Get available monitoring intents
 */
export async function getMonitoringIntents(): Promise<{ intents: MonitoringIntent[]; vigilance_levels: VigilanceLevel[] }> {
    const response = await fetch(`${ML_API_BASE}/api/monitor/intents`);
    if (!response.ok) throw new Error('Failed to fetch monitoring intents');
    return response.json();
}

/**
 * Submit a monitoring request
 */
export async function submitMonitoringRequest(request: MonitoringRequest): Promise<MonitoringJobResponse> {
    const response = await fetch(`${ML_API_BASE}/api/monitor`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
    });
    if (!response.ok) throw new Error('Failed to submit monitoring request');
    return response.json();
}

/**
 * Get monitoring job status
 */
export async function getMonitoringStatus(jobId: string): Promise<StatusResponse> {
    const response = await fetch(`${ML_API_BASE}/api/monitor/status/${jobId}`);
    if (!response.ok) throw new Error('Failed to get monitoring status');
    return response.json();
}

/**
 * Get monitoring results
 */
export async function getMonitoringResults(jobId: string): Promise<MonitoringResults> {
    const response = await fetch(`${ML_API_BASE}/api/monitor/results/${jobId}`);
    if (!response.ok) {
        if (response.status === 202) {
            throw new Error('Results not ready yet');
        }
        throw new Error('Failed to get monitoring results');
    }
    return response.json();
}

