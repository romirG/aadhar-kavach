/**
 * API Service for connecting frontend to backend hotspot detection APIs
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:3001/api';

interface APIResponse<T> {
    success: boolean;
    data?: T;
    error?: string;
}

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<APIResponse<T>> {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options?.headers,
            },
            ...options,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        return { success: true, data };
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        return {
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error'
        };
    }
}

// =====================================
// Hotspot Detection API
// =====================================

export interface SpatialAnalysisResult {
    analysis: string;
    result: {
        moransI: number;
        expectedI: number;
        zScore: number;
        pValue: number;
        isSignificant: boolean;
        interpretation: string;
    };
    recordsAnalyzed: number;
    statesAnalyzed: number;
}

export interface GiStarResult {
    region: string;
    state: string;
    district?: string;
    totalEnrollments: number;
    zScore: number;
    pValue: number;
    classification: string;
    isHotspot: boolean;
    isColdspot: boolean;
}

export interface GiStarResponse {
    analysis: string;
    summary: {
        totalRegions: number;
        hotspotCount: number;
        coldspotCount: number;
        notSignificant: number;
    };
    hotspots: GiStarResult[];
    coldspots: GiStarResult[];
    allRegions: GiStarResult[];
}

export interface VelocityResult {
    region: string;
    velocity: number;
    currentVelocity: number;
    acceleration: number;
    trend: 'accelerating_growth' | 'decelerating_growth' | 'accelerating_decline' | 'decelerating_decline' | 'stable';
    dataPoints: number;
}

export interface VelocityResponse {
    analysis: string;
    summary: {
        totalRegions: number;
        accelerating: number;
        decelerating: number;
        stable: number;
    };
    concerningRegions: VelocityResult[];
    topPerformers: VelocityResult[];
    allRegions: VelocityResult[];
}

export interface AnomalyAlert {
    index: number;
    date: string;
    region: string;
    observedValue: number;
    expectedValue: number;
    deviation: number;
    zScore: number;
    direction: 'above_expected' | 'below_expected';
    severity: 'critical' | 'high' | 'medium';
    percentageDeviation: number;
}

export interface AnomaliesResponse {
    analysis: string;
    summary: {
        totalAnomalies: number;
        regionsWithAnomalies: number;
        threshold: number;
        critical: number;
        high: number;
        medium: number;
    };
    alerts: AnomalyAlert[];
}

export interface TrendsResponse {
    analysis: string;
    decomposition: {
        dates: string[];
        values: number[];
        trend: number[];
        seasonal: number[];
        residual: number[];
        hasSeasonality: boolean;
        seasonalPattern: number[];
    };
    regionalTrends: {
        regions: Array<{
            region: string;
            trend: 'increasing' | 'decreasing' | 'stable';
            monthlyChange: number;
            rSquared: number;
        }>;
        summary: {
            increasing: number;
            stable: number;
            decreasing: number;
        };
    };
}

export interface InterventionRecommendation {
    priority: number;
    region: string;
    coverage: string;
    action: string[];
    urgency: 'critical' | 'high' | 'medium';
}

export interface InterventionResponse {
    analysis: string;
    summary: {
        totalRegionsAnalyzed: number;
        regionsNeedingIntervention: number;
        coverageThreshold: number;
    };
    prioritizedRegions: GiStarResult[];
    actionableRecommendations: InterventionRecommendation[];
}

/**
 * Hotspot Detection API methods
 */
export const hotspotApi = {
    /**
     * Get spatial clustering analysis using Moran's I
     */
    getSpatialAnalysis: (limit = 100) =>
        fetchAPI<SpatialAnalysisResult>(`/hotspots/spatial?limit=${limit}`),

    /**
     * Get Getis-Ord Gi* hotspot scores by region
     */
    getGiStarScores: (limit = 200, groupBy: 'state' | 'district' = 'state') =>
        fetchAPI<GiStarResponse>(`/hotspots/gi-star?limit=${limit}&groupBy=${groupBy}`),

    /**
     * Get enrollment velocity by region
     */
    getVelocity: (limit = 500, groupBy: 'state' | 'district' = 'state') =>
        fetchAPI<VelocityResponse>(`/hotspots/velocity?limit=${limit}&groupBy=${groupBy}`),

    /**
     * Get automated anomaly alerts
     */
    getAnomalies: (limit = 300, threshold = 2) =>
        fetchAPI<AnomaliesResponse>(`/hotspots/anomalies?limit=${limit}&threshold=${threshold}`),

    /**
     * Get seasonal trend decomposition
     */
    getTrends: (limit = 500, state?: string) => {
        const url = state
            ? `/hotspots/trends?limit=${limit}&state=${encodeURIComponent(state)}`
            : `/hotspots/trends?limit=${limit}`;
        return fetchAPI<TrendsResponse>(url);
    },

    /**
     * Get intervention priority list with recommendations
     */
    getInterventions: (limit = 200, coverageThreshold = 85) =>
        fetchAPI<InterventionResponse>(`/hotspots/intervention?limit=${limit}&coverageThreshold=${coverageThreshold}`),
};

// =====================================
// AI Recommendations API
// =====================================

export interface AIRecommendation {
    success: boolean;
    recommendation: string;
    error?: string;
}

export const aiApi = {
    /**
     * Get AI-powered analysis for hotspot data
     */
    analyzeHotspots: (data: unknown) =>
        fetchAPI<AIRecommendation>('/ai/analyze', {
            method: 'POST',
            body: JSON.stringify({ type: 'hotspots', data }),
        }),

    /**
     * Get custom AI recommendation
     */
    getRecommendation: (prompt: string, data?: unknown) =>
        fetchAPI<AIRecommendation>('/ai/recommend', {
            method: 'POST',
            body: JSON.stringify({ prompt, data }),
        }),
};

// =====================================
// Health Check
// =====================================

export const healthApi = {
    check: () => fetchAPI<{ status: string; timestamp: string }>('/health'),
};

// =====================================
// Intent-Based Monitoring API (ML Backend)
// =====================================

const ML_API_BASE = 'http://localhost:8000';

export interface MonitoringIntent {
    id: string;
    display_name: string;
    description: string;
    signals: string[];
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
    status: string;
    message: string;
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
    time_period: string;
    analysis_scope: string;
    report_id: string;
    completed_at: string;
}

export interface StatusResponse {
    job_id: string;
    status: string;
    progress: number;
    message: string;
}

export async function getMonitoringIntents(): Promise<{ intents: MonitoringIntent[]; vigilance_levels: VigilanceLevel[] }> {
    const response = await fetch(`${ML_API_BASE}/api/monitor/intents`);
    if (!response.ok) throw new Error('Failed to fetch monitoring intents');
    return response.json();
}

export async function submitMonitoringRequest(request: MonitoringRequest): Promise<MonitoringJobResponse> {
    const response = await fetch(`${ML_API_BASE}/api/monitor`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
    });
    if (!response.ok) throw new Error('Failed to submit monitoring request');
    return response.json();
}

export async function getMonitoringStatus(jobId: string): Promise<StatusResponse> {
    const response = await fetch(`${ML_API_BASE}/api/monitor/status/${jobId}`);
    if (!response.ok) throw new Error('Failed to get monitoring status');
    return response.json();
}

export async function getMonitoringResults(jobId: string): Promise<MonitoringResults> {
    const response = await fetch(`${ML_API_BASE}/api/monitor/results/${jobId}`);
    if (!response.ok) throw new Error('Failed to get monitoring results');
    return response.json();
}

export default {
    hotspot: hotspotApi,
    ai: aiApi,
    health: healthApi,
};
