/**
 * LeafletMap Component v2
 * 
 * Interactive India choropleth map with:
 * - Diverging color scale (Crimson → Orange → Yellow → Blue → Deep Blue)
 * - Z-score based styling
 * - 6-hour auto-refresh
 * - 0.7 opacity for background visibility
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Types
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
}

interface GeoJSONData {
    type: 'FeatureCollection';
    features: GeoJSONFeature[];
}

interface LegendItem {
    label: string;
    color: string;
    description: string;
}

interface LeafletMapProps {
    geojson: GeoJSONData | null;
    legend?: LegendItem[];
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
const getZScoreColor = (zScore: number): string => {
    if (zScore > 2.58) return '#DC143C';  // Crimson
    if (zScore > 1.96) return '#FF8C00';  // Dark Orange
    if (zScore > 0) return '#FFD700';     // Gold
    if (zScore > -1.96) return '#87CEEB'; // Sky Blue
    if (zScore > -2.58) return '#4169E1'; // Royal Blue
    return '#00008B';                      // Dark Blue
};

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
    onDistrictClick,
    height = '600px',
    autoRefreshInterval = 6 * 60 * 60 * 1000, // 6 hours
    onRefreshRequest
}: LeafletMapProps) {
    const mapRef = useRef<HTMLDivElement>(null);
    const mapInstanceRef = useRef<L.Map | null>(null);
    const geojsonLayerRef = useRef<L.GeoJSON | null>(null);
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

    // Initialize map
    useEffect(() => {
        if (!mapRef.current || mapInstanceRef.current) return;

        // Create map centered on India
        const map = L.map(mapRef.current, {
            center: [22.5, 82.5],  // Center of India
            zoom: 5,
            minZoom: 4,
            maxZoom: 12,
            scrollWheelZoom: true
        });

        // Add dark-style tile layer for better contrast
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(map);

        mapInstanceRef.current = map;

        return () => {
            if (mapInstanceRef.current) {
                mapInstanceRef.current.remove();
                mapInstanceRef.current = null;
            }
        };
    }, []);

    // Update GeoJSON layer when data changes
    useEffect(() => {
        if (!mapInstanceRef.current || !geojson) return;

        // Remove existing layer
        if (geojsonLayerRef.current) {
            mapInstanceRef.current.removeLayer(geojsonLayerRef.current);
        }

        // Style function with diverging colors
        const style = (feature: any) => {
            const props = feature.properties;
            const zScore = props.z_score || 0;

            // Use provided color or calculate from Z-score
            const fillColor = props.color || getZScoreColor(zScore);
            const opacity = props.opacity || 0.7;

            return {
                fillColor,
                weight: 1,
                opacity: 1,
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
        const resetHighlight = (e: L.LeafletMouseEvent) => {
            if (geojsonLayerRef.current) {
                geojsonLayerRef.current.resetStyle(e.target);
            }
        };

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

            (layer as L.Path).on({
                mouseover: highlightFeature,
                mouseout: resetHighlight,
                click: onFeatureClick
            });
        };

        // Create GeoJSON layer
        const geojsonLayer = L.geoJSON(geojson as any, {
            style: style as any,
            onEachFeature: onEachFeature as any
        }).addTo(mapInstanceRef.current);

        geojsonLayerRef.current = geojsonLayer;

        // Fit bounds to India
        try {
            const bounds = geojsonLayer.getBounds();
            if (bounds.isValid()) {
                mapInstanceRef.current.fitBounds(bounds, { padding: [20, 20] });
            }
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

    return (
        <div className="relative">
            <div
                ref={mapRef}
                style={{ height, width: '100%', borderRadius: '12px' }}
                className="shadow-lg"
            />

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
                </div>
            )}
        </div>
    );
}
