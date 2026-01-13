import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Download, FileText, Users, AlertCircle, TrendingUp } from 'lucide-react';
import { statesData, formatNumber, formatPercentage } from '@/data/mockData';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function GenderTracker() {
  const genderData = statesData
    .map(s => ({
      name: s.code,
      male: (s.maleEnrolled / s.enrolledPopulation) * 100,
      female: (s.femaleEnrolled / s.enrolledPopulation) * 100,
      gap: s.genderGap,
      state: s.name,
    }))
    .sort((a, b) => b.gap - a.gap)
    .slice(0, 12);

  const highGapStates = statesData.filter(s => s.genderGap > 6);
  const avgGap = statesData.reduce((sum, s) => sum + s.genderGap, 0) / statesData.length;

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Gender Inclusion Tracker</h1>
            <p className="text-muted-foreground">
              Monitoring gender parity in Aadhaar enrollment across states
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <FileText className="mr-2 h-4 w-4" />
              Generate Recommendations
            </Button>
            <Button size="sm">
              <Download className="mr-2 h-4 w-4" />
              Export Analysis
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid gap-4 md:grid-cols-3">
          <Card className="shadow-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">National Gender Gap</p>
                  <p className="text-2xl font-bold">{avgGap.toFixed(1)}%</p>
                </div>
                <div className="rounded-full bg-warning/10 p-3">
                  <Users className="h-5 w-5 text-warning" />
                </div>
              </div>
              <p className="mt-2 text-sm text-muted-foreground">Male-Female enrollment difference</p>
            </CardContent>
          </Card>
          <Card className="border-destructive/50 shadow-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">High Gap States</p>
                  <p className="text-2xl font-bold">{highGapStates.length}</p>
                </div>
                <div className="rounded-full bg-destructive/10 p-3">
                  <AlertCircle className="h-5 w-5 text-destructive" />
                </div>
              </div>
              <p className="mt-2 text-sm text-destructive">States with &gt;6% gender gap</p>
            </CardContent>
          </Card>
          <Card className="shadow-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Best Performer</p>
                  <p className="text-2xl font-bold">Kerala</p>
                </div>
                <div className="rounded-full bg-success/10 p-3">
                  <TrendingUp className="h-5 w-5 text-success" />
                </div>
              </div>
              <p className="mt-2 text-sm text-success">-2% gap (more females enrolled)</p>
            </CardContent>
          </Card>
        </div>

        {/* Gender Distribution Chart */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>Gender Distribution by State</CardTitle>
            <CardDescription>Male vs Female enrollment percentage (sorted by gap)</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={genderData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis type="number" domain={[0, 60]} className="text-xs" />
                <YAxis dataKey="name" type="category" className="text-xs" width={40} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'hsl(var(--popover))', 
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number) => `${value.toFixed(1)}%`}
                />
                <Bar dataKey="male" fill="hsl(var(--chart-1))" name="Male %" radius={[0, 4, 4, 0]} />
                <Bar dataKey="female" fill="hsl(var(--chart-3))" name="Female %" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-4 flex justify-center gap-6 text-sm">
              <span className="flex items-center gap-2"><div className="h-3 w-3 rounded" style={{ backgroundColor: 'hsl(var(--chart-1))' }} /> Male</span>
              <span className="flex items-center gap-2"><div className="h-3 w-3 rounded" style={{ backgroundColor: 'hsl(var(--chart-3))' }} /> Female</span>
            </div>
          </CardContent>
        </Card>

        {/* High Gap States Table */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>States Requiring Intervention</CardTitle>
            <CardDescription>States with gender gap &gt;6% needing targeted programs</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="pb-3 text-left font-medium">State</th>
                    <th className="pb-3 text-left font-medium">Gender Gap</th>
                    <th className="pb-3 text-left font-medium">Male Enrolled</th>
                    <th className="pb-3 text-left font-medium">Female Enrolled</th>
                    <th className="pb-3 text-left font-medium">Recommendation</th>
                  </tr>
                </thead>
                <tbody>
                  {highGapStates.map((state) => (
                    <tr key={state.id} className="border-b border-border/50">
                      <td className="py-3 font-medium">{state.name}</td>
                      <td className="py-3">
                        <Badge variant="destructive">{state.genderGap}%</Badge>
                      </td>
                      <td className="py-3">{formatNumber(state.maleEnrolled)}</td>
                      <td className="py-3">{formatNumber(state.femaleEnrolled)}</td>
                      <td className="py-3">
                        <Button variant="ghost" size="sm">View Plan</Button>
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
