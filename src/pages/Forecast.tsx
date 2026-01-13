import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Download, RefreshCw, Share2, TrendingUp, ArrowUp, ArrowDown } from 'lucide-react';
import { enrollmentTrends, statesData, formatNumber } from '@/data/mockData';
import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';

export default function Forecast() {
  const [selectedState, setSelectedState] = useState<string>('all');

  const historicalData = enrollmentTrends.filter(d => !d.forecast);
  const forecastData = enrollmentTrends.filter(d => d.forecast);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Enrollment Forecasting</h1>
            <p className="text-muted-foreground">
              ML-powered demand prediction with 6-month horizon
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <RefreshCw className="mr-2 h-4 w-4" />
              Update Forecast
            </Button>
            <Button variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Download CSV
            </Button>
            <Button size="sm">
              <Share2 className="mr-2 h-4 w-4" />
              Share Chart
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
                  <SelectItem value="all">All India</SelectItem>
                  {statesData.map(state => (
                    <SelectItem key={state.id} value={state.id}>{state.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Badge variant="outline" className="h-10 px-4 flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-success" />
                Model: ARIMA + XGBoost Ensemble
              </Badge>
            </div>
          </CardContent>
        </Card>

        {/* KPI Cards */}
        <div className="grid gap-4 md:grid-cols-3">
          <Card className="shadow-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Projected (Next 6 Months)</p>
                  <p className="text-2xl font-bold">89.2 L</p>
                </div>
                <div className="rounded-full bg-success/10 p-3">
                  <ArrowUp className="h-5 w-5 text-success" />
                </div>
              </div>
              <p className="mt-2 text-sm text-success">+8.4% vs last period</p>
            </CardContent>
          </Card>
          <Card className="shadow-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Monthly Growth Rate</p>
                  <p className="text-2xl font-bold">3.2%</p>
                </div>
                <div className="rounded-full bg-info/10 p-3">
                  <TrendingUp className="h-5 w-5 text-info" />
                </div>
              </div>
              <p className="mt-2 text-sm text-muted-foreground">Compound monthly rate</p>
            </CardContent>
          </Card>
          <Card className="shadow-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Forecast Confidence</p>
                  <p className="text-2xl font-bold">94.6%</p>
                </div>
                <Badge className="bg-success text-success-foreground">High</Badge>
              </div>
              <p className="mt-2 text-sm text-muted-foreground">Based on 24-month history</p>
            </CardContent>
          </Card>
        </div>

        {/* Forecast Chart */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>Enrollment Trend & Forecast</CardTitle>
            <CardDescription>Historical data with 6-month prediction and confidence intervals</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={enrollmentTrends}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="month" className="text-xs" tick={{ fontSize: 11 }} />
                <YAxis className="text-xs" tickFormatter={(v) => formatNumber(v)} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'hsl(var(--popover))', 
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number) => formatNumber(value)}
                />
                <Area 
                  type="monotone" 
                  dataKey="forecastUpper" 
                  stackId="1"
                  stroke="none" 
                  fill="hsl(var(--primary))" 
                  fillOpacity={0.1}
                />
                <Area 
                  type="monotone" 
                  dataKey="forecastLower" 
                  stackId="2"
                  stroke="none" 
                  fill="hsl(var(--background))" 
                  fillOpacity={1}
                />
                <Line 
                  type="monotone" 
                  dataKey="enrollments" 
                  stroke="hsl(var(--primary))" 
                  strokeWidth={2}
                  dot={{ fill: 'hsl(var(--primary))' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="forecast" 
                  stroke="hsl(var(--chart-3))" 
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={{ fill: 'hsl(var(--chart-3))' }}
                />
              </AreaChart>
            </ResponsiveContainer>
            <div className="mt-4 flex justify-center gap-6 text-sm">
              <span className="flex items-center gap-2">
                <div className="h-0.5 w-6 bg-primary" /> Historical
              </span>
              <span className="flex items-center gap-2">
                <div className="h-0.5 w-6 bg-accent border-dashed" style={{ borderTop: '2px dashed hsl(var(--chart-3))' }} /> Forecast
              </span>
              <span className="flex items-center gap-2">
                <div className="h-3 w-6 rounded bg-primary/10" /> Confidence Band
              </span>
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
