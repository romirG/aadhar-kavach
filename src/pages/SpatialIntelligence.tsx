/**
 * Spatial Intelligence Dashboard
 * 
 * Phase 1: Spatial-Temporal Intelligence System
 * 
 * Features:
 * - Interactive Leaflet choropleth map of India
 * - Real-time Gi* hotspot analysis
 * - SARIMA forecasts with confidence intervals
 * - Scenario modeling for mobile unit deployment
 */

import { useState, useEffect } from 'react';
import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
    MapPin, TrendingUp, AlertTriangle, Activity,
    RefreshCw, Download, Zap, Target, ThermometerSun,
    Snowflake, Flame, BarChart3
} from 'lucide-react';
import LeafletMap from '@/components/maps/LeafletMap';

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

interface ForecastPoint {
    month: string;
    predicted: number;
    ci_lower: number;
    ci_upper: number;
}

interface StateForecast {
    state: string;
    model: string;
    forecasts: ForecastPoint[];
}

interface SpatialStats {
    morans_i: number;
    morans_p_value: number;
    mean_intensity: number;
    std_intensity: number;
    hotspot_count: number;
    coldspot_count: number;
    anomaly_count: number;
    in_sync_count: number;
}

interface AnalyticsData {
    success: boolean;
    map_geojson: any;
    forecast_data: StateForecast[];
    stats: SpatialStats;
}

const API_BASE = 'http://localhost:3002';

export default function SpatialIntelligence() {
    const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedDistrict, setSelectedDistrict] = useState<DistrictProperties | null>(null);
    const [activeTab, setActiveTab] = useState('map');

    // Fetch analytics data
    const fetchAnalytics = async () => {
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE}/api/v1/analytics`);
            if (!response.ok) throw new Error('Failed to fetch analytics');

            const data = await response.json();
            if (data.success) {
                setAnalyticsData(data);
            } else {
                throw new Error(data.error || 'Unknown error');
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load analytics');
            console.error('Analytics fetch error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchAnalytics();
    }, []);

    const handleDistrictClick = (properties: DistrictProperties) => {
        setSelectedDistrict(properties);
    };

    const stats = analyticsData?.stats;

    // Format number for display
    const formatNumber = (num: number): string => {
        if (num >= 10000000) return `${(num / 10000000).toFixed(2)}Cr`;
        if (num >= 100000) return `${(num / 100000).toFixed(2)}L`;
        if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
        return num.toFixed(2);
    };

    return (
        <DashboardLayout>
            <div className="space-y-6">
                {/* Header */}
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                    <div>
                        <h1 className="text-2xl font-bold flex items-center gap-2">
                            <MapPin className="h-6 w-6 text-primary" />
                            Spatial-Temporal Intelligence System
                        </h1>
                        <p className="text-muted-foreground">
                            Phase 1: Geographic Hotspot Detection with Getis-Ord Gi* and SARIMA Forecasting
                        </p>
                    </div>
                    <div className="flex gap-2">
                        <Button variant="outline" size="sm" onClick={fetchAnalytics} disabled={isLoading}>
                            <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                            Refresh
                        </Button>
                        <Button variant="outline" size="sm">
                            <Download className="mr-2 h-4 w-4" />
                            Export
                        </Button>
                    </div>
                </div>

                {/* Error State */}
                {error && (
                    <Card className="border-red-500/50 bg-red-500/10">
                        <CardContent className="pt-6">
                            <div className="flex items-center gap-2 text-red-500">
                                <AlertTriangle className="h-5 w-5" />
                                <span>{error}</span>
                                <Button variant="outline" size="sm" onClick={fetchAnalytics} className="ml-auto">
                                    Retry
                                </Button>
                            </div>
                        </CardContent>
                    </Card>
                )}

                {/* Stats Cards */}
                <div className="grid gap-4 md:grid-cols-4">
                    {/* Moran's I */}
                    <Card className="border-l-4 border-l-blue-500 bg-gradient-to-br from-blue-500/5 to-transparent">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                                <Activity className="h-4 w-4" />
                                Spatial Autocorrelation
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">
                                {stats?.morans_i?.toFixed(3) || '—'}
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                                Moran's I (p = {stats?.morans_p_value?.toFixed(4) || '—'})
                            </p>
                        </CardContent>
                    </Card>

                    {/* Hotspots */}
                    <Card className="border-l-4 border-l-red-500 bg-gradient-to-br from-red-500/5 to-transparent">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                                <Flame className="h-4 w-4" />
                                Hot Spots
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-red-500">
                                {stats?.hotspot_count || 0}
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                                Z &gt; 1.96 (p &lt; 0.05)
                            </p>
                        </CardContent>
                    </Card>

                    {/* Coldspots */}
                    <Card className="border-l-4 border-l-blue-500 bg-gradient-to-br from-blue-500/5 to-transparent">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                                <Snowflake className="h-4 w-4" />
                                Cold Spots
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-blue-500">
                                {stats?.coldspot_count || 0}
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                                Z &lt; -1.96 (p &lt; 0.05)
                            </p>
                        </CardContent>
                    </Card>

                    {/* Anomalies */}
                    <Card className="border-l-4 border-l-amber-500 bg-gradient-to-br from-amber-500/5 to-transparent">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                                <AlertTriangle className="h-4 w-4" />
                                Anomalies
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-amber-500">
                                {stats?.anomaly_count || 0}
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                                Z &gt; 3 (extreme outliers)
                            </p>
                        </CardContent>
                    </Card>
                </div>

                {/* Tabs */}
                <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
                    <TabsList className="bg-muted/50">
                        <TabsTrigger value="map">
                            <MapPin className="mr-2 h-4 w-4" />
                            India Map
                        </TabsTrigger>
                        <TabsTrigger value="forecast">
                            <TrendingUp className="mr-2 h-4 w-4" />
                            Forecasts
                        </TabsTrigger>
                        <TabsTrigger value="analysis">
                            <BarChart3 className="mr-2 h-4 w-4" />
                            Analysis
                        </TabsTrigger>
                    </TabsList>

                    {/* Map Tab */}
                    <TabsContent value="map" className="space-y-4">
                        <div className="grid gap-6 lg:grid-cols-3">
                            <Card className="lg:col-span-2">
                                <CardHeader>
                                    <CardTitle>India District Choropleth</CardTitle>
                                    <CardDescription>
                                        Getis-Ord Gi* hotspot analysis with Queen contiguity weights
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    {isLoading ? (
                                        <div className="h-[500px] flex items-center justify-center bg-muted/20 rounded-lg">
                                            <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                                        </div>
                                    ) : analyticsData?.map_geojson ? (
                                        <LeafletMap
                                            geojson={analyticsData.map_geojson}
                                            onDistrictClick={handleDistrictClick}
                                            height="500px"
                                        />
                                    ) : (
                                        <div className="h-[500px] flex items-center justify-center bg-muted/20 rounded-lg">
                                            <p className="text-muted-foreground">No map data available</p>
                                        </div>
                                    )}
                                </CardContent>
                            </Card>

                            {/* Selected District Details */}
                            <Card className="bg-gradient-to-br from-primary/5 to-transparent">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <Target className="h-5 w-5" />
                                        District Details
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    {selectedDistrict ? (
                                        <div className="space-y-4">
                                            <div>
                                                <h3 className="text-xl font-bold">{selectedDistrict.district}</h3>
                                                <p className="text-muted-foreground">{selectedDistrict.state}</p>
                                            </div>

                                            <div className="space-y-2">
                                                <div className="flex justify-between">
                                                    <span className="text-muted-foreground">Coverage</span>
                                                    <span className="font-medium">{selectedDistrict.coverage?.toFixed(1)}%</span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-muted-foreground">Enrollment</span>
                                                    <span className="font-medium">{formatNumber(selectedDistrict.enrollment)}</span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-muted-foreground">Population</span>
                                                    <span className="font-medium">{formatNumber(selectedDistrict.population)}</span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-muted-foreground">Intensity</span>
                                                    <span className="font-mono text-sm">
                                                        {(selectedDistrict.intensity * 100000).toFixed(2)} per 100K
                                                    </span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-muted-foreground">Z-Score</span>
                                                    <span className="font-mono">{selectedDistrict.z_score?.toFixed(3)}</span>
                                                </div>
                                                <div className="flex justify-between items-center">
                                                    <span className="text-muted-foreground">Classification</span>
                                                    <Badge
                                                        variant={
                                                            selectedDistrict.classification === 'Hot Spot' ? 'destructive' :
                                                                selectedDistrict.classification === 'Cold Spot' ? 'default' :
                                                                    selectedDistrict.classification === 'Anomaly' ? 'destructive' :
                                                                        'secondary'
                                                        }
                                                    >
                                                        {selectedDistrict.classification}
                                                    </Badge>
                                                </div>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="text-center py-8 text-muted-foreground">
                                            <MapPin className="mx-auto h-12 w-12 opacity-50" />
                                            <p className="mt-2">Click a district on the map to see details</p>
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        </div>
                    </TabsContent>

                    {/* Forecast Tab */}
                    <TabsContent value="forecast" className="space-y-4">
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <TrendingUp className="h-5 w-5" />
                                    SARIMA(1,1,1)(1,1,1,12) Forecasts
                                </CardTitle>
                                <CardDescription>
                                    6-month enrollment projections with 95% confidence intervals
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                {analyticsData?.forecast_data && analyticsData.forecast_data.length > 0 ? (
                                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                                        {analyticsData.forecast_data.map((forecast) => (
                                            <Card key={forecast.state} className="bg-muted/20">
                                                <CardHeader className="pb-2">
                                                    <CardTitle className="text-lg">{forecast.state}</CardTitle>
                                                    <CardDescription className="font-mono text-xs">
                                                        {forecast.model}
                                                    </CardDescription>
                                                </CardHeader>
                                                <CardContent>
                                                    <div className="space-y-2">
                                                        {forecast.forecasts.slice(0, 3).map((fp) => (
                                                            <div key={fp.month} className="flex justify-between text-sm">
                                                                <span className="text-muted-foreground">{fp.month}</span>
                                                                <span className="font-mono">
                                                                    {formatNumber(fp.predicted)}
                                                                    <span className="text-xs text-muted-foreground ml-1">
                                                                        ±{formatNumber((fp.ci_upper - fp.ci_lower) / 2)}
                                                                    </span>
                                                                </span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </CardContent>
                                            </Card>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="text-center py-8 text-muted-foreground">
                                        <TrendingUp className="mx-auto h-12 w-12 opacity-50" />
                                        <p className="mt-2">No forecast data available</p>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    </TabsContent>

                    {/* Analysis Tab */}
                    <TabsContent value="analysis" className="space-y-4">
                        <div className="grid gap-4 md:grid-cols-2">
                            <Card>
                                <CardHeader>
                                    <CardTitle>Spatial Statistics Summary</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        <div className="flex justify-between p-2 bg-muted/30 rounded">
                                            <span>Moran's I</span>
                                            <span className="font-mono">{stats?.morans_i?.toFixed(4) || '—'}</span>
                                        </div>
                                        <div className="flex justify-between p-2 bg-muted/30 rounded">
                                            <span>P-Value</span>
                                            <span className="font-mono">{stats?.morans_p_value?.toFixed(4) || '—'}</span>
                                        </div>
                                        <div className="flex justify-between p-2 bg-muted/30 rounded">
                                            <span>Mean Intensity</span>
                                            <span className="font-mono">{(stats?.mean_intensity || 0).toExponential(4)}</span>
                                        </div>
                                        <div className="flex justify-between p-2 bg-muted/30 rounded">
                                            <span>Std Intensity</span>
                                            <span className="font-mono">{(stats?.std_intensity || 0).toExponential(4)}</span>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card>
                                <CardHeader>
                                    <CardTitle>Classification Distribution</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        <div className="flex justify-between items-center p-2 bg-red-500/10 rounded">
                                            <span className="flex items-center gap-2">
                                                <div className="w-3 h-3 rounded bg-red-500" />
                                                Hot Spots
                                            </span>
                                            <Badge variant="destructive">{stats?.hotspot_count || 0}</Badge>
                                        </div>
                                        <div className="flex justify-between items-center p-2 bg-blue-500/10 rounded">
                                            <span className="flex items-center gap-2">
                                                <div className="w-3 h-3 rounded bg-blue-500" />
                                                Cold Spots
                                            </span>
                                            <Badge>{stats?.coldspot_count || 0}</Badge>
                                        </div>
                                        <div className="flex justify-between items-center p-2 bg-amber-500/10 rounded">
                                            <span className="flex items-center gap-2">
                                                <div className="w-3 h-3 rounded bg-amber-500" />
                                                Anomalies
                                            </span>
                                            <Badge variant="outline">{stats?.anomaly_count || 0}</Badge>
                                        </div>
                                        <div className="flex justify-between items-center p-2 bg-green-500/10 rounded">
                                            <span className="flex items-center gap-2">
                                                <div className="w-3 h-3 rounded bg-green-500" />
                                                In-Sync
                                            </span>
                                            <Badge variant="secondary">{stats?.in_sync_count || 0}</Badge>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>
                    </TabsContent>
                </Tabs>
            </div>
        </DashboardLayout>
    );
}
