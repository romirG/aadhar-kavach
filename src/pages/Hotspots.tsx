import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Download, FileText, Share2, MapPin, AlertCircle, TrendingDown } from 'lucide-react';
import { statesData, districtsData, getHotspotDistricts, formatPercentage } from '@/data/mockData';
import { useState } from 'react';

export default function Hotspots() {
  const [selectedState, setSelectedState] = useState<string>('all');
  const hotspots = getHotspotDistricts();
  
  const filteredHotspots = selectedState === 'all' 
    ? hotspots 
    : hotspots.filter(d => d.stateId === selectedState);

  const getCoverageColor = (coverage: number) => {
    if (coverage >= 95) return 'bg-success';
    if (coverage >= 90) return 'bg-success/70';
    if (coverage >= 85) return 'bg-warning';
    if (coverage >= 80) return 'bg-warning/70';
    return 'bg-destructive';
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Geographic Hotspot Detection</h1>
            <p className="text-muted-foreground">
              AI-powered identification of low-coverage districts requiring intervention
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Export Data
            </Button>
            <Button variant="outline" size="sm">
              <FileText className="mr-2 h-4 w-4" />
              Generate Report
            </Button>
            <Button size="sm">
              <Share2 className="mr-2 h-4 w-4" />
              Share
            </Button>
          </div>
        </div>

        {/* Filters */}
        <Card>
          <CardContent className="p-4">
            <div className="flex flex-wrap gap-4">
              <Select value={selectedState} onValueChange={setSelectedState}>
                <SelectTrigger className="w-48">
                  <SelectValue placeholder="Select State" />
                </SelectTrigger>
                <SelectContent className="bg-popover">
                  <SelectItem value="all">All States</SelectItem>
                  {statesData.map(state => (
                    <SelectItem key={state.id} value={state.id}>{state.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Badge variant="outline" className="h-10 px-4 flex items-center gap-2">
                <AlertCircle className="h-4 w-4 text-destructive" />
                {hotspots.length} Hotspots Identified
              </Badge>
            </div>
          </CardContent>
        </Card>

        {/* Map Placeholder + Stats */}
        <div className="grid gap-6 lg:grid-cols-3">
          <Card className="lg:col-span-2 shadow-card">
            <CardHeader>
              <CardTitle>India Coverage Map</CardTitle>
              <CardDescription>Districts color-coded by enrollment coverage</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex h-80 items-center justify-center rounded-lg border-2 border-dashed border-border bg-muted/30">
                <div className="text-center">
                  <MapPin className="mx-auto h-12 w-12 text-muted-foreground" />
                  <p className="mt-2 text-muted-foreground">Interactive India Map</p>
                  <p className="text-sm text-muted-foreground">Click on states to drill down</p>
                </div>
              </div>
              <div className="mt-4 flex justify-center gap-4 text-sm">
                <span className="flex items-center gap-2"><div className="h-3 w-3 rounded bg-success" /> 95%+</span>
                <span className="flex items-center gap-2"><div className="h-3 w-3 rounded bg-success/70" /> 90-95%</span>
                <span className="flex items-center gap-2"><div className="h-3 w-3 rounded bg-warning" /> 85-90%</span>
                <span className="flex items-center gap-2"><div className="h-3 w-3 rounded bg-warning/70" /> 80-85%</span>
                <span className="flex items-center gap-2"><div className="h-3 w-3 rounded bg-destructive" /> &lt;80%</span>
              </div>
            </CardContent>
          </Card>

          {/* AI Insights */}
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingDown className="h-5 w-5 text-destructive" />
                AI Insights
              </CardTitle>
              <CardDescription>Gemini-powered recommendations</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="rounded-lg bg-destructive/10 p-3">
                <p className="text-sm font-medium text-destructive">Critical: Northeast Gap</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Arunachal Pradesh at 72.1% coverage. Recommend mobile camps in border areas.
                </p>
              </div>
              <div className="rounded-lg bg-warning/10 p-3">
                <p className="text-sm font-medium text-warning">Warning: Bihar Districts</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  5 districts below 80%. Kishanganj needs urgent intervention.
                </p>
              </div>
              <div className="rounded-lg bg-info/10 p-3">
                <p className="text-sm font-medium text-info">Suggestion</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Deploy 15 additional mobile units to tribal regions in Q2.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Hotspot Districts Table */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>Hotspot Districts</CardTitle>
            <CardDescription>Districts requiring immediate attention (coverage &lt;85%)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="pb-3 text-left font-medium">District</th>
                    <th className="pb-3 text-left font-medium">State</th>
                    <th className="pb-3 text-left font-medium">Coverage</th>
                    <th className="pb-3 text-left font-medium">Risk Score</th>
                    <th className="pb-3 text-left font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredHotspots.map((district) => (
                    <tr key={district.id} className="border-b border-border/50">
                      <td className="py-3 font-medium">{district.name}</td>
                      <td className="py-3 text-muted-foreground">{district.stateName}</td>
                      <td className="py-3">
                        <div className="flex items-center gap-2">
                          <div className={`h-2 w-2 rounded-full ${getCoverageColor(district.enrollmentCoverage)}`} />
                          {formatPercentage(district.enrollmentCoverage)}
                        </div>
                      </td>
                      <td className="py-3">
                        <Badge variant={district.riskScore > 4 ? 'destructive' : 'secondary'}>
                          {district.riskScore.toFixed(1)}
                        </Badge>
                      </td>
                      <td className="py-3">
                        <Button variant="ghost" size="sm">View Details</Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
