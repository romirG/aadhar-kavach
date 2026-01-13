/**
<<<<<<< Updated upstream
 * LeafletMap Component v2
 * 
 * Interactive India choropleth map with:
 * - Diverging color scale (Crimson → Orange → Yellow → Blue → Deep Blue)
 * - Z-score based styling
 * - 6-hour auto-refresh
=======
 * LeafletMap Component v3
 * 
 * Interactive India choropleth map with:
 * - Live data fetch from /api/v1/details on click
 * - Diverging color scale (Crimson → Orange → Yellow → Blue → Deep Blue)
 * - Velocity, coverage, and SARIMA forecast display
>>>>>>> Stashed changes
 * - 0.7 opacity for background visibility
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Types
<<<<<<< Updated upstream
=======
interface VelocityData {
    current_enrolments: number;
    previous_enrolments: number;
    raw_change: number;
    normalized_intensity: number;
    intensity_per_100k: number;
    formula: string;
}

interface CoverageData {
    percentage: number;
    enrolled: number;
    population: number;
    remaining: number;
}

interface SpatialData {
    z_score: number;
    p_value: number;
    classification: string;
    color: string;
    confidence: string;
}

interface ForecastPoint {
    month: string;
    predicted: number;
    ci_lower: number;
    ci_upper: number;
    growth_rate: number;
}

interface ForecastData {
    model: string;
    predicted_6m_growth_percent: number;
    monthly_projections: ForecastPoint[];
}

interface InterventionData {
    priority: string;
    action: string;
    rationale: string;
}

interface DistrictDetails {
    success: boolean;
    district_id: string;
    district: string;
    state: string;
    last_updated: string;
    velocity: VelocityData;
    coverage: CoverageData;
    spatial: SpatialData;
    forecast: ForecastData;
    intervention: InterventionData;
    updates: {
        biometric: number;
        demographic: number;
    };
}

>>>>>>> Stashed changes
interface DistrictProperties {
    district: string;
    state: string;
    enrolments: number;
    population: number;
    coverage: number;
    intensity: number;
    intensity_per_100k: number;
    z_score: number;
    p_value: number;
    classification: string;
    color: string;
    opacity?: number;
<<<<<<< Updated upstream
    biometric_updates?: number;
    demographic_updates?: number;
    last_updated?: string;
    // Hex grid properties
    h3_index?: string;
    center_lat?: number;
    center_lon?: number;
}

interface GeoJSONFeature {
    type: 'Feature';
    properties: DistrictProperties;
    geometry: {
        type: string;
        coordinates: number[][][];
    };
=======
>>>>>>> Stashed changes
}

interface GeoJSONData {
    type: 'FeatureCollection';
<<<<<<< Updated upstream
    features: GeoJSONFeature[];
=======
    features: any[];
>>>>>>> Stashed changes
}

interface LegendItem {
    label: string;
    color: string;
    description: string;
}

interface LeafletMapProps {
    geojson: GeoJSONData | null;
    legend?: LegendItem[];
<<<<<<< Updated upstream
    onDistrictClick?: (properties: DistrictProperties) => void;
    height?: string;
    autoRefreshInterval?: number; // in milliseconds
    onRefreshRequest?: () => void;
}

/**
 * Get color from diverging palette based on Z-score
 * 
 * Palette:
 * - Crimson Red (Z > 2.58): Significant Growth Hotspot (99% confidence)
 * - Orange (1.96 < Z < 2.58): Emerging Trend
 * - Gold/Yellow (baseline): In-Sync with population growth
 * - Royal Blue (Z < -1.96): Declining
 * - Deep Blue (Z < -2.58): Cold Spot / Digital Exclusion Zone
 */
=======
    onDistrictSelect?: (details: DistrictDetails) => void;
    height?: string;
    onRefreshRequest?: () => void;
}

const API_BASE = 'http://localhost:3002';

// Color function for Z-scores
>>>>>>> Stashed changes
const getZScoreColor = (zScore: number): string => {
    if (zScore > 2.58) return '#DC143C';  // Crimson
    if (zScore > 1.96) return '#FF8C00';  // Dark Orange
    if (zScore > 0) return '#FFD700';     // Gold
    if (zScore > -1.96) return '#87CEEB'; // Sky Blue
    if (zScore > -2.58) return '#4169E1'; // Royal Blue
    return '#00008B';                      // Dark Blue
};

<<<<<<< Updated upstream
/**
 * Get classification label from Z-score
 */
const getClassificationLabel = (zScore: number): string => {
    if (zScore > 2.58) return 'Significant Hotspot (99% conf)';
    if (zScore > 1.96) return 'Emerging Trend (95% conf)';
    if (zScore > -1.96) return 'In-Sync';
    if (zScore > -2.58) return 'Declining';
    return 'Cold Spot (Exclusion Zone)';
};

=======
>>>>>>> Stashed changes
// Format numbers for display
const formatNumber = (num: number): string => {
    if (num >= 10000000) return `${(num / 10000000).toFixed(2)} Cr`;
    if (num >= 100000) return `${(num / 100000).toFixed(2)} L`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)} K`;
    return num.toLocaleString();
};

export default function LeafletMap({
    geojson,
    legend,
<<<<<<< Updated upstream
    onDistrictClick,
    height = '600px',
    autoRefreshInterval = 6 * 60 * 60 * 1000, // 6 hours
=======
    onDistrictSelect,
    height = '600px',
>>>>>>> Stashed changes
    onRefreshRequest
}: LeafletMapProps) {
    const mapRef = useRef<HTMLDivElement>(null);
    const mapInstanceRef = useRef<L.Map | null>(null);
    const geojsonLayerRef = useRef<L.GeoJSON | null>(null);
<<<<<<< Updated upstream
    const [selectedFeature, setSelectedFeature] = useState<DistrictProperties | null>(null);
    const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

    // Auto-refresh timer
    useEffect(() => {
        if (!onRefreshRequest || autoRefreshInterval <= 0) return;

        const timer = setInterval(() => {
            console.log('Auto-refreshing map data...');
            onRefreshRequest();
            setLastRefresh(new Date());
        }, autoRefreshInterval);

        return () => clearInterval(timer);
    }, [autoRefreshInterval, onRefreshRequest]);
=======
    const [selectedDetails, setSelectedDetails] = useState<DistrictDetails | null>(null);
    const [isLoadingDetails, setIsLoadingDetails] = useState(false);
    const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

    // Fetch district details from API
    const fetchDistrictDetails = useCallback(async (districtId: string) => {
        setIsLoadingDetails(true);
        try {
            const response = await fetch(`${API_BASE}/api/v1/details/${encodeURIComponent(districtId)}`);
            if (!response.ok) throw new Error('Failed to fetch details');

            const data: DistrictDetails = await response.json();
            if (data.success) {
                setSelectedDetails(data);
                onDistrictSelect?.(data);
            }
        } catch (error) {
            console.error('Error fetching district details:', error);
        } finally {
            setIsLoadingDetails(false);
        }
    }, [onDistrictSelect]);
>>>>>>> Stashed changes

    // Initialize map
    useEffect(() => {
        if (!mapRef.current || mapInstanceRef.current) return;

<<<<<<< Updated upstream
        // Create map centered on India
        const map = L.map(mapRef.current, {
            center: [22.5, 82.5],  // Center of India
=======
        const map = L.map(mapRef.current, {
            center: [22.5, 82.5],
>>>>>>> Stashed changes
            zoom: 5,
            minZoom: 4,
            maxZoom: 12,
            scrollWheelZoom: true
        });

<<<<<<< Updated upstream
        // Add dark-style tile layer for better contrast
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
=======
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OSM &copy; CARTO',
>>>>>>> Stashed changes
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(map);

        mapInstanceRef.current = map;
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
        return () => {
            if (mapInstanceRef.current) {
                mapInstanceRef.current.remove();
                mapInstanceRef.current = null;
            }
        };
    }, []);

<<<<<<< Updated upstream
    // Update GeoJSON layer when data changes
    useEffect(() => {
        if (!mapInstanceRef.current || !geojson) return;

        // Remove existing layer
=======
    // Update GeoJSON layer
    useEffect(() => {
        if (!mapInstanceRef.current || !geojson) return;

>>>>>>> Stashed changes
        if (geojsonLayerRef.current) {
            mapInstanceRef.current.removeLayer(geojsonLayerRef.current);
        }

<<<<<<< Updated upstream
        // Style function with diverging colors
        const style = (feature: any) => {
            const props = feature.properties;
            const zScore = props.z_score || 0;

            // Use provided color or calculate from Z-score
            const fillColor = props.color || getZScoreColor(zScore);
            const opacity = props.opacity || 0.7;
=======
        const style = (feature: any) => {
            const props = feature.properties;
            const zScore = props.z_score || 0;
            const fillColor = props.color || getZScoreColor(zScore);
>>>>>>> Stashed changes

            return {
                fillColor,
                weight: 1,
                opacity: 1,
<<<<<<< Updated upstream
                color: '#ffffff',  // White borders
                fillOpacity: opacity
            };
        };

        // Highlight on hover
        const highlightFeature = (e: L.LeafletMouseEvent) => {
            const layer = e.target;
            layer.setStyle({
                weight: 3,
                color: '#333',
                fillOpacity: 0.85
            });
            layer.bringToFront();
        };

        // Reset on mouseout
=======
                color: '#ffffff',
                fillOpacity: props.opacity || 0.7
            };
        };

        const highlightFeature = (e: L.LeafletMouseEvent) => {
            const layer = e.target;
            layer.setStyle({ weight: 3, color: '#333', fillOpacity: 0.85 });
            layer.bringToFront();
        };

>>>>>>> Stashed changes
        const resetHighlight = (e: L.LeafletMouseEvent) => {
            if (geojsonLayerRef.current) {
                geojsonLayerRef.current.resetStyle(e.target);
            }
        };

<<<<<<< Updated upstream
        // Click handler
        const onFeatureClick = (e: L.LeafletMouseEvent) => {
            const properties = e.target.feature.properties as DistrictProperties;
            setSelectedFeature(properties);
            onDistrictClick?.(properties);
        };

        // Enhanced popup content
        const onEachFeature = (feature: any, layer: L.Layer) => {
            const props = feature.properties as DistrictProperties;
            const zScore = props.z_score || 0;
            const classification = getClassificationLabel(zScore);
            const color = props.color || getZScoreColor(zScore);

            const popupContent = `
        <div style="min-width: 280px; font-family: system-ui, -apple-system, sans-serif;">
          <div style="
            background: linear-gradient(135deg, ${color}20, ${color}10);
            padding: 12px;
            margin: -12px -12px 12px -12px;
            border-bottom: 3px solid ${color};
          ">
            <h3 style="margin: 0; color: #1a1a1a; font-size: 18px; font-weight: 600;">
              ${props.district || props.state || 'Unknown'}
            </h3>
            <p style="margin: 4px 0 0 0; color: #666; font-size: 13px;">${props.state || ''}</p>
          </div>
          
          <div style="padding: 8px 0;">
            <div style="
              display: inline-block;
              background: ${color};
              color: white;
              padding: 4px 12px;
              border-radius: 20px;
              font-size: 12px;
              font-weight: 500;
              margin-bottom: 12px;
            ">${classification}</div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px;">
              <div style="background: #f5f5f5; padding: 8px; border-radius: 6px;">
                <div style="color: #888; font-size: 11px;">Enrolments</div>
                <div style="font-weight: 600; color: #333;">${formatNumber(props.enrolments || 0)}</div>
              </div>
              <div style="background: #f5f5f5; padding: 8px; border-radius: 6px;">
                <div style="color: #888; font-size: 11px;">Coverage</div>
                <div style="font-weight: 600; color: #333;">${props.coverage?.toFixed(1) || 'N/A'}%</div>
              </div>
              <div style="background: #f5f5f5; padding: 8px; border-radius: 6px;">
                <div style="color: #888; font-size: 11px;">Velocity</div>
                <div style="font-weight: 600; color: #333;">${props.intensity_per_100k?.toFixed(1) || 'N/A'}/100K</div>
              </div>
              <div style="background: #f5f5f5; padding: 8px; border-radius: 6px;">
                <div style="color: #888; font-size: 11px;">Z-Score</div>
                <div style="font-weight: 600; color: ${color};">${zScore.toFixed(3)}</div>
              </div>
            </div>
            
            ${props.biometric_updates ? `
              <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;">
                <div style="font-size: 12px; color: #666;">
                  <span>📊 Bio Updates: ${formatNumber(props.biometric_updates)}</span>
                  <span style="margin-left: 12px;">📝 Demo Updates: ${formatNumber(props.demographic_updates || 0)}</span>
                </div>
              </div>
            ` : ''}
          </div>
        </div>
      `;

            layer.bindPopup(popupContent, {
                maxWidth: 350,
                className: 'custom-popup'
            });
=======
        // Click handler - fetch from /api/v1/details endpoint
        const onFeatureClick = (e: L.LeafletMouseEvent) => {
            const props = e.target.feature.properties;
            const districtId = props.district || props.district_code || `District_${props.id || 1}`;
            fetchDistrictDetails(districtId);
        };

        const onEachFeature = (feature: any, layer: L.Layer) => {
            const props = feature.properties;
            const zScore = props.z_score || 0;
            const color = props.color || getZScoreColor(zScore);

            // Quick popup with basic info (details loaded on click)
            const popupContent = `
        <div style="min-width: 200px; font-family: system-ui;">
          <h3 style="margin: 0 0 8px; font-size: 16px; color: ${color};">
            ${props.district || props.state}
          </h3>
          <p style="margin: 4px 0; font-size: 13px;">
            <strong>Classification:</strong> 
            <span style="color: ${color}">${props.classification || 'N/A'}</span>
          </p>
          <p style="margin: 4px 0; font-size: 13px;">
            <strong>Z-Score:</strong> ${zScore.toFixed(3)}
          </p>
          <p style="margin: 4px 0; font-size: 13px;">
            <strong>Coverage:</strong> ${props.coverage?.toFixed(1) || 'N/A'}%
          </p>
          <p style="margin: 8px 0 0; font-size: 11px; color: #666;">
            Click for detailed analysis →
          </p>
        </div>
      `;

            layer.bindPopup(popupContent, { maxWidth: 300 });
>>>>>>> Stashed changes

            (layer as L.Path).on({
                mouseover: highlightFeature,
                mouseout: resetHighlight,
                click: onFeatureClick
            });
        };

<<<<<<< Updated upstream
        // Create GeoJSON layer
=======
>>>>>>> Stashed changes
        const geojsonLayer = L.geoJSON(geojson as any, {
            style: style as any,
            onEachFeature: onEachFeature as any
        }).addTo(mapInstanceRef.current);

        geojsonLayerRef.current = geojsonLayer;

<<<<<<< Updated upstream
        // Fit bounds to India
=======
>>>>>>> Stashed changes
        try {
            const bounds = geojsonLayer.getBounds();
            if (bounds.isValid()) {
                mapInstanceRef.current.fitBounds(bounds, { padding: [20, 20] });
            }
<<<<<<< Updated upstream
        } catch (e) {
            console.log('Could not fit bounds, using default view');
        }

    }, [geojson, onDistrictClick]);

    // Default legend if not provided
    const displayLegend = legend || [
        { label: 'Significant Hotspot', color: '#DC143C', description: 'Z > 2.58 (99% confidence)' },
        { label: 'Emerging Trend', color: '#FF8C00', description: 'Z > 1.96 (95% confidence)' },
        { label: 'In-Sync', color: '#FFD700', description: 'Baseline growth' },
        { label: 'Declining', color: '#4169E1', description: 'Z < -1.96' },
        { label: 'Cold Spot', color: '#00008B', description: 'Z < -2.58 (Exclusion Zone)' },
    ];

=======
        } catch (e) { }

    }, [geojson, fetchDistrictDetails]);

    const displayLegend = legend || [
        { label: 'Significant Hotspot', color: '#DC143C', description: 'Z > 2.58' },
        { label: 'Emerging Trend', color: '#FF8C00', description: 'Z > 1.96' },
        { label: 'In-Sync', color: '#FFD700', description: 'Baseline' },
        { label: 'Declining', color: '#4169E1', description: 'Z < -1.96' },
        { label: 'Cold Spot', color: '#00008B', description: 'Z < -2.58' },
    ];

    const getPriorityColor = (priority: string) => {
        switch (priority) {
            case 'CRITICAL': return '#DC143C';
            case 'HIGH': return '#FF8C00';
            case 'MONITOR': return '#4169E1';
            default: return '#44AA44';
        }
    };

>>>>>>> Stashed changes
    return (
        <div className="relative">
            <div
                ref={mapRef}
                style={{ height, width: '100%', borderRadius: '12px' }}
                className="shadow-lg"
            />

<<<<<<< Updated upstream
            {/* Legend - Diverging Color Scale */}
            <div className="absolute bottom-4 right-4 bg-white/95 backdrop-blur-sm rounded-xl p-4 shadow-xl z-[1000] min-w-[200px]">
                <h4 className="text-sm font-bold mb-3 text-gray-800">Gi* Classification</h4>
                <div className="space-y-2">
                    {displayLegend.map((item, idx) => (
                        <div key={idx} className="flex items-center gap-3">
                            <div
                                className="w-5 h-5 rounded-md shadow-sm"
                                style={{ background: item.color }}
                            />
                            <div className="flex-1">
                                <div className="text-xs font-medium text-gray-700">{item.label}</div>
                                <div className="text-[10px] text-gray-500">{item.description}</div>
                            </div>
                        </div>
                    ))}
                </div>
                <div className="mt-3 pt-3 border-t border-gray-200">
                    <div className="text-[10px] text-gray-500">
                        Opacity: 70% • Last refresh: {lastRefresh.toLocaleTimeString()}
                    </div>
                </div>
            </div>

            {/* Selected Feature Card */}
            {selectedFeature && (
                <div className="absolute top-4 left-4 bg-white/95 backdrop-blur-sm rounded-xl p-4 shadow-xl z-[1000] max-w-sm">
                    <div className="flex items-center justify-between mb-3">
                        <h4 className="font-bold text-lg text-gray-800">
                            {selectedFeature.district || selectedFeature.state}
                        </h4>
                        <button
                            onClick={() => setSelectedFeature(null)}
                            className="text-gray-400 hover:text-gray-600 p-1"
                        >
                            ✕
                        </button>
                    </div>

                    <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                            <span className="text-gray-500">Z-Score</span>
                            <span
                                className="font-mono font-bold"
                                style={{ color: selectedFeature.color || getZScoreColor(selectedFeature.z_score) }}
                            >
                                {selectedFeature.z_score?.toFixed(3)}
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-500">Classification</span>
                            <span
                                className="px-2 py-0.5 rounded-full text-white text-xs font-medium"
                                style={{ background: selectedFeature.color || getZScoreColor(selectedFeature.z_score) }}
                            >
                                {selectedFeature.classification}
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-500">Coverage</span>
                            <span className="font-medium">{selectedFeature.coverage?.toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-500">Velocity</span>
                            <span className="font-mono">{selectedFeature.intensity_per_100k?.toFixed(2)} /100K</span>
                        </div>
                    </div>
=======
            {/* Legend */}
            <div className="absolute bottom-4 right-4 bg-white/95 backdrop-blur-sm rounded-xl p-4 shadow-xl z-[1000] min-w-[180px]">
                <h4 className="text-sm font-bold mb-3">Gi* Classification</h4>
                <div className="space-y-2">
                    {displayLegend.map((item, idx) => (
                        <div key={idx} className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded" style={{ background: item.color }} />
                            <span className="text-xs">{item.label}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Selected District Details Panel */}
            {(selectedDetails || isLoadingDetails) && (
                <div className="absolute top-4 left-4 bg-white/95 backdrop-blur-sm rounded-xl shadow-xl z-[1000] w-80 max-h-[calc(100%-2rem)] overflow-auto">
                    {isLoadingDetails ? (
                        <div className="p-6 text-center">
                            <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full mx-auto"></div>
                            <p className="mt-2 text-sm text-muted-foreground">Loading details...</p>
                        </div>
                    ) : selectedDetails && (
                        <div className="p-4">
                            {/* Header */}
                            <div className="flex items-center justify-between mb-4">
                                <div>
                                    <h3 className="font-bold text-lg">{selectedDetails.district}</h3>
                                    <p className="text-sm text-muted-foreground">{selectedDetails.state}</p>
                                </div>
                                <button
                                    onClick={() => setSelectedDetails(null)}
                                    className="text-gray-400 hover:text-gray-600 p-1"
                                >✕</button>
                            </div>

                            {/* Classification Badge */}
                            <div
                                className="inline-block px-3 py-1 rounded-full text-white text-sm font-medium mb-4"
                                style={{ background: selectedDetails.spatial.color }}
                            >
                                {selectedDetails.spatial.classification}
                                {selectedDetails.spatial.confidence !== 'N/A' &&
                                    <span className="ml-1 opacity-75">({selectedDetails.spatial.confidence})</span>
                                }
                            </div>

                            {/* Velocity Section */}
                            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                                <h4 className="text-xs font-semibold text-gray-500 uppercase mb-2">
                                    📈 Enrollment Velocity
                                </h4>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    <div>
                                        <span className="text-gray-500">Current</span>
                                        <p className="font-semibold">{formatNumber(selectedDetails.velocity.current_enrolments)}</p>
                                    </div>
                                    <div>
                                        <span className="text-gray-500">Previous</span>
                                        <p className="font-semibold">{formatNumber(selectedDetails.velocity.previous_enrolments)}</p>
                                    </div>
                                    <div className="col-span-2">
                                        <span className="text-gray-500">Intensity</span>
                                        <p className="font-mono font-semibold text-primary">
                                            {selectedDetails.velocity.intensity_per_100k.toFixed(2)} / 100K
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Coverage Section */}
                            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                                <h4 className="text-xs font-semibold text-gray-500 uppercase mb-2">
                                    📊 Coverage
                                </h4>
                                <div className="flex items-center gap-3">
                                    <div className="flex-1">
                                        <div className="w-full bg-gray-200 rounded-full h-3">
                                            <div
                                                className="bg-primary h-3 rounded-full transition-all"
                                                style={{ width: `${Math.min(100, selectedDetails.coverage.percentage)}%` }}
                                            />
                                        </div>
                                    </div>
                                    <span className="font-bold text-lg">{selectedDetails.coverage.percentage.toFixed(1)}%</span>
                                </div>
                                <p className="text-xs text-gray-500 mt-1">
                                    {formatNumber(selectedDetails.coverage.remaining)} remaining
                                </p>
                            </div>

                            {/* Forecast Section */}
                            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                                <h4 className="text-xs font-semibold text-gray-500 uppercase mb-2">
                                    🔮 6-Month Forecast
                                </h4>
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm">Predicted Growth</span>
                                    <span
                                        className="font-bold text-lg"
                                        style={{ color: selectedDetails.forecast.predicted_6m_growth_percent > 0 ? '#22c55e' : '#ef4444' }}
                                    >
                                        {selectedDetails.forecast.predicted_6m_growth_percent > 0 ? '+' : ''}
                                        {selectedDetails.forecast.predicted_6m_growth_percent.toFixed(2)}%
                                    </span>
                                </div>
                                <div className="space-y-1">
                                    {selectedDetails.forecast.monthly_projections.slice(0, 3).map((fp) => (
                                        <div key={fp.month} className="flex justify-between text-xs">
                                            <span className="text-gray-500">{fp.month}</span>
                                            <span className="font-mono">
                                                {formatNumber(fp.predicted)}
                                                <span className="text-gray-400 ml-1">
                                                    ±{formatNumber((fp.ci_upper - fp.ci_lower) / 2)}
                                                </span>
                                            </span>
                                        </div>
                                    ))}
                                </div>
                                <p className="text-xs text-gray-400 mt-1">{selectedDetails.forecast.model}</p>
                            </div>

                            {/* Intervention Section */}
                            <div
                                className="p-3 rounded-lg border-2"
                                style={{ borderColor: getPriorityColor(selectedDetails.intervention.priority) }}
                            >
                                <div className="flex items-center gap-2 mb-1">
                                    <span
                                        className="px-2 py-0.5 rounded text-xs font-bold text-white"
                                        style={{ background: getPriorityColor(selectedDetails.intervention.priority) }}
                                    >
                                        {selectedDetails.intervention.priority}
                                    </span>
                                    <span className="text-xs font-medium">Intervention</span>
                                </div>
                                <p className="text-sm font-medium">{selectedDetails.intervention.action}</p>
                                <p className="text-xs text-gray-500 mt-1">{selectedDetails.intervention.rationale}</p>
                            </div>
                        </div>
                    )}
>>>>>>> Stashed changes
                </div>
            )}
        </div>
    );
}
