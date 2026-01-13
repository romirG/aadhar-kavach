import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Users, MapPin, TrendingUp, AlertTriangle, ArrowRight, 
  CheckCircle2, Clock, Activity 
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { 
  statesData, getTotalEnrollments, getAverageEnrollmentCoverage, 
  getTotalPendingUpdates, getActiveAlertsCount, formatNumber, formatPercentage,
  anomalyAlerts
} from '@/data/mockData';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';

const kpiCards = [
  {
    title: 'Total Enrollments',
    value: formatNumber(getTotalEnrollments()),
    change: '+2.4%',
    changeType: 'positive' as const,
    icon: Users,
    color: 'text-primary',
    bgColor: 'bg-primary/10',
  },
  {
    title: 'Coverage Rate',
    value: formatPercentage(getAverageEnrollmentCoverage()),
    change: '+0.8%',
    changeType: 'positive' as const,
    icon: CheckCircle2,
    color: 'text-success',
    bgColor: 'bg-success/10',
  },
  {
    title: 'Active Centers',
    value: formatNumber(statesData.reduce((sum, s) => sum + s.activeCenters, 0)),
    change: '+12',
    changeType: 'positive' as const,
    icon: MapPin,
    color: 'text-info',
    bgColor: 'bg-info/10',
  },
  {
    title: 'Pending Updates',
    value: formatNumber(getTotalPendingUpdates()),
    change: '-5.2%',
    changeType: 'negative' as const,
    icon: Clock,
    color: 'text-warning',
    bgColor: 'bg-warning/10',
  },
];

const topStates = statesData
  .sort((a, b) => b.enrolledPopulation - a.enrolledPopulation)
  .slice(0, 6)
  .map(s => ({ name: s.code, enrollments: s.enrolledPopulation / 1000000 }));

const genderData = [
  { name: 'Male', value: 52.3, color: 'hsl(var(--chart-1))' },
  { name: 'Female', value: 47.7, color: 'hsl(var(--chart-3))' },
];

export default function Dashboard() {
  const activeAlerts = getActiveAlertsCount();
  const recentAlerts = anomalyAlerts.slice(0, 3);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Dashboard Overview</h1>
            <p className="text-muted-foreground">
              Real-time Aadhaar enrollment analytics and insights
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Activity className="mr-2 h-4 w-4" />
              Live Data
            </Button>
            <Button size="sm">
              Download Report
            </Button>
          </div>
        </div>

        {/* KPI Cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {kpiCards.map((card) => (
            <Card key={card.title} className="shadow-card hover:shadow-card-hover transition-shadow">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className={`rounded-lg p-2.5 ${card.bgColor}`}>
                    <card.icon className={`h-5 w-5 ${card.color}`} />
                  </div>
                  <Badge 
                    variant={card.changeType === 'positive' ? 'default' : 'secondary'}
                    className={card.changeType === 'positive' ? 'bg-success/20 text-success' : 'bg-destructive/20 text-destructive'}
                  >
                    {card.change}
                  </Badge>
                </div>
                <div className="mt-4">
                  <p className="text-2xl font-bold">{card.value}</p>
                  <p className="text-sm text-muted-foreground">{card.title}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Charts Row */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Top States Chart */}
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle className="text-lg">Top States by Enrollment</CardTitle>
              <CardDescription>Enrolled population in millions</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={topStates}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                  <XAxis dataKey="name" className="text-xs" />
                  <YAxis className="text-xs" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--popover))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="enrollments" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Gender Distribution */}
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle className="text-lg">Gender Distribution</CardTitle>
              <CardDescription>National enrollment by gender</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-center">
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={genderData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {genderData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 flex justify-center gap-6">
                {genderData.map((item) => (
                  <div key={item.name} className="flex items-center gap-2">
                    <div 
                      className="h-3 w-3 rounded-full" 
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-sm">{item.name}: {item.value}%</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Alerts & Quick Actions */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Recent Alerts */}
          <Card className="shadow-card lg:col-span-2">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-lg">Recent Alerts</CardTitle>
                <CardDescription>{activeAlerts} critical/high priority alerts</CardDescription>
              </div>
              <Button variant="outline" size="sm" asChild>
                <Link to="/anomalies">
                  View All <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {recentAlerts.map((alert) => (
                  <div 
                    key={alert.id} 
                    className="flex items-center gap-4 rounded-lg border border-border p-3"
                  >
                    <div className={`rounded-full p-2 ${
                      alert.severity === 'critical' ? 'bg-destructive/20' :
                      alert.severity === 'high' ? 'bg-warning/20' : 'bg-muted'
                    }`}>
                      <AlertTriangle className={`h-4 w-4 ${
                        alert.severity === 'critical' ? 'text-destructive' :
                        alert.severity === 'high' ? 'text-warning' : 'text-muted-foreground'
                      }`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{alert.location}</p>
                      <p className="text-xs text-muted-foreground truncate">{alert.description}</p>
                    </div>
                    <Badge variant={alert.severity === 'critical' ? 'destructive' : 'secondary'}>
                      {alert.severity}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle className="text-lg">Quick Actions</CardTitle>
              <CardDescription>Navigate to key features</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button variant="outline" className="w-full justify-start" asChild>
                <Link to="/hotspots">
                  <MapPin className="mr-2 h-4 w-4 text-destructive" />
                  View Hotspots
                </Link>
              </Button>
              <Button variant="outline" className="w-full justify-start" asChild>
                <Link to="/forecast">
                  <TrendingUp className="mr-2 h-4 w-4 text-success" />
                  Run Forecast
                </Link>
              </Button>
              <Button variant="outline" className="w-full justify-start" asChild>
                <Link to="/anomalies">
                  <AlertTriangle className="mr-2 h-4 w-4 text-warning" />
                  Check Anomalies
                </Link>
              </Button>
              <Button variant="outline" className="w-full justify-start" asChild>
                <Link to="/gender">
                  <Users className="mr-2 h-4 w-4 text-info" />
                  Gender Analysis
                </Link>
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
}
