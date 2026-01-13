/**
 * LeafletMap Component
 * 
 * Interactive India choropleth map using Leaflet.js
 * Colors districts based on Z-score classification:
 * - Red: Hot Spots (Z > 1.96)
 * - Dark Red: Anomalies (Z > 3)
 * - Blue: Cold Spots (Z < -1.96)
 * - Green: In-Sync (-1.96 < Z < 1.96)
 */

import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Types
interface DistrictProperties {
    district: string;
    state: string;
    enrollment: number;
    population: number;
    coverage: number;
    intensity: number;
    scaled_intensity: number;
    z_score: number;
    p_value: number;
    classification: 'Anomaly' | 'Hot Spot' | 'Cold Spot' | 'In-Sync';
    predicted_growth?: number;
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

interface LeafletMapProps {
    geojson: GeoJSONData | null;
    onDistrictClick?: (properties: DistrictProperties) => void;
    height?: string;
}

// Classification color mapping
const getClassificationColor = (classification: string): string => {
    switch (classification) {
        case 'Anomaly':
            return '#8B0000';  // Dark Red
        case 'Hot Spot':
            return '#FF4444';  // Red
        case 'Cold Spot':
            return '#4444FF';  // Blue
        case 'In-Sync':
            return '#44AA44';  // Green
        default:
            return '#888888';  // Gray
    }
};

// Format numbers for display
const formatNumber = (num: number): string => {
    if (num >= 10000000) return `${(num / 10000000).toFixed(2)}Cr`;
    if (num >= 100000) return `${(num / 100000).toFixed(2)}L`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
};

export default function LeafletMap({
    geojson,
    onDistrictClick,
    height = '500px'
}: LeafletMapProps) {
    const mapRef = useRef<HTMLDivElement>(null);
    const mapInstanceRef = useRef<L.Map | null>(null);
    const geojsonLayerRef = useRef<L.GeoJSON | null>(null);
    const [selectedDistrict, setSelectedDistrict] = useState<DistrictProperties | null>(null);

    // Initialize map
    useEffect(() => {
        if (!mapRef.current || mapInstanceRef.current) return;

        // Create map centered on India
        const map = L.map(mapRef.current, {
            center: [22.5, 82.5],  // Center of India
            zoom: 5,
            minZoom: 4,
            maxZoom: 10,
            scrollWheelZoom: true
        });

        // Add tile layer (OpenStreetMap)
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        }).addTo(map);

        mapInstanceRef.current = map;

        // Cleanup
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

        // Style function for choropleth
        const style = (feature: GeoJSONFeature) => {
            const classification = feature.properties.classification || 'In-Sync';
            return {
                fillColor: getClassificationColor(classification),
                weight: 1,
                opacity: 1,
                color: 'white',
                fillOpacity: 0.7
            };
        };

        // Highlight on hover
        const highlightFeature = (e: L.LeafletMouseEvent) => {
            const layer = e.target;
            layer.setStyle({
                weight: 3,
                color: '#333',
                fillOpacity: 0.9
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
            setSelectedDistrict(properties);
            onDistrictClick?.(properties);
        };

        // onEachFeature function for popups
        const onEachFeature = (feature: GeoJSONFeature, layer: L.Layer) => {
            const props = feature.properties;

            // Create popup content
            const popupContent = `
        <div style="min-width: 200px; font-family: system-ui, -apple-system, sans-serif;">
          <h3 style="margin: 0 0 8px 0; color: #1a1a1a; font-size: 16px; border-bottom: 1px solid #eee; padding-bottom: 8px;">
            ${props.district}
          </h3>
          <div style="font-size: 13px; color: #666;">
            <p style="margin: 4px 0;"><strong>State:</strong> ${props.state}</p>
            <p style="margin: 4px 0;"><strong>Coverage:</strong> ${props.coverage?.toFixed(1) || 'N/A'}%</p>
            <p style="margin: 4px 0;"><strong>Enrollment:</strong> ${formatNumber(props.enrollment || 0)}</p>
            <p style="margin: 4px 0;"><strong>Population:</strong> ${formatNumber(props.population || 0)}</p>
            <hr style="margin: 8px 0; border: none; border-top: 1px solid #eee;">
            <p style="margin: 4px 0;"><strong>Intensity:</strong> ${(props.intensity * 100000).toFixed(2)} per 100K</p>
            <p style="margin: 4px 0;"><strong>Z-Score:</strong> ${props.z_score?.toFixed(3) || 'N/A'}</p>
            <p style="margin: 4px 0;">
              <strong>Classification:</strong> 
              <span style="
                background: ${getClassificationColor(props.classification)};
                color: white;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 12px;
              ">${props.classification || 'N/A'}</span>
            </p>
            ${props.predicted_growth ? `
              <p style="margin: 4px 0;"><strong>Predicted Growth:</strong> ${(props.predicted_growth * 100).toFixed(2)}%</p>
            ` : ''}
          </div>
        </div>
      `;

            layer.bindPopup(popupContent, {
                maxWidth: 300,
                className: 'custom-popup'
            });

            // Add hover and click events
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
            mapInstanceRef.current.fitBounds(geojsonLayer.getBounds());
        } catch (e) {
            // Keep default India center if bounds fail
        }

    }, [geojson, onDistrictClick]);

    return (
        <div className="relative">
            <div
                ref={mapRef}
                style={{ height, width: '100%', borderRadius: '8px' }}
            />

            {/* Legend */}
            <div className="absolute bottom-4 right-4 bg-white/90 backdrop-blur-sm rounded-lg p-3 shadow-lg z-[1000]">
                <h4 className="text-sm font-semibold mb-2">Classification</h4>
                <div className="space-y-1 text-xs">
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded" style={{ background: '#8B0000' }} />
                        <span>Anomaly (Z &gt; 3)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded" style={{ background: '#FF4444' }} />
                        <span>Hot Spot (Z &gt; 1.96)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded" style={{ background: '#44AA44' }} />
                        <span>In-Sync</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded" style={{ background: '#4444FF' }} />
                        <span>Cold Spot (Z &lt; -1.96)</span>
                    </div>
                </div>
            </div>

            {/* Selected District Info */}
            {selectedDistrict && (
                <div className="absolute top-4 left-4 bg-white/90 backdrop-blur-sm rounded-lg p-3 shadow-lg z-[1000] max-w-xs">
                    <div className="flex items-center justify-between mb-2">
                        <h4 className="font-semibold">{selectedDistrict.district}</h4>
                        <button
                            onClick={() => setSelectedDistrict(null)}
                            className="text-gray-400 hover:text-gray-600"
                        >
                            ✕
                        </button>
                    </div>
                    <div className="text-sm space-y-1">
                        <p><span className="text-gray-500">State:</span> {selectedDistrict.state}</p>
                        <p><span className="text-gray-500">Z-Score:</span> {selectedDistrict.z_score?.toFixed(3)}</p>
                        <p>
                            <span
                                className="px-2 py-0.5 rounded text-white text-xs"
                                style={{ background: getClassificationColor(selectedDistrict.classification) }}
                            >
                                {selectedDistrict.classification}
                            </span>
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}
