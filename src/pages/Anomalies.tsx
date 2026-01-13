import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, Flag, X, Eye, Bell, Shield } from 'lucide-react';
import { anomalyAlerts, biometricRisks, formatPercentage } from '@/data/mockData';

export default function Anomalies() {
  const criticalAlerts = anomalyAlerts.filter(a => a.severity === 'critical' || a.severity === 'high');

  const getSeverityStyles = (severity: string) => {
    switch (severity) {
      case 'critical': return { badge: 'bg-destructive text-destructive-foreground', bg: 'bg-destructive/10', text: 'text-destructive' };
      case 'high': return { badge: 'bg-warning text-warning-foreground', bg: 'bg-warning/10', text: 'text-warning' };
      case 'medium': return { badge: 'bg-info text-info-foreground', bg: 'bg-info/10', text: 'text-info' };
      default: return { badge: 'bg-muted text-muted-foreground', bg: 'bg-muted', text: 'text-muted-foreground' };
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Anomaly & Fraud Detection</h1>
            <p className="text-muted-foreground">
              Real-time monitoring of suspicious patterns and operator behavior
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Bell className="mr-2 h-4 w-4" />
              Configure Alerts
            </Button>
            <Button size="sm">
              <Shield className="mr-2 h-4 w-4" />
              Security Report
            </Button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid gap-4 md:grid-cols-4">
          <Card className="border-destructive/50 shadow-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-destructive/10 p-2">
                  <AlertTriangle className="h-5 w-5 text-destructive" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{criticalAlerts.length}</p>
                  <p className="text-sm text-muted-foreground">Critical/High Alerts</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="shadow-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-warning/10 p-2">
                  <Flag className="h-5 w-5 text-warning" />
                </div>
                <div>
                  <p className="text-2xl font-bold">12</p>
                  <p className="text-sm text-muted-foreground">Flagged Operators</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="shadow-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-info/10 p-2">
                  <Eye className="h-5 w-5 text-info" />
                </div>
                <div>
                  <p className="text-2xl font-bold">847</p>
                  <p className="text-sm text-muted-foreground">Under Review</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="shadow-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-success/10 p-2">
                  <Shield className="h-5 w-5 text-success" />
                </div>
                <div>
                  <p className="text-2xl font-bold">99.2%</p>
                  <p className="text-sm text-muted-foreground">System Integrity</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Alerts Table */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>Active Alerts</CardTitle>
            <CardDescription>Sorted by risk score and severity</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {anomalyAlerts.map((alert) => {
                const styles = getSeverityStyles(alert.severity);
                return (
                  <div key={alert.id} className={`rounded-lg border border-border p-4 ${styles.bg}`}>
                    <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                      <div className="flex items-start gap-3">
                        <AlertTriangle className={`h-5 w-5 mt-0.5 ${styles.text}`} />
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{alert.location}</span>
                            <Badge className={styles.badge}>{alert.severity}</Badge>
                            <Badge variant="outline">Score: {alert.riskScore.toFixed(1)}</Badge>
                          </div>
                          <p className="mt-1 text-sm text-muted-foreground">{alert.description}</p>
                          <p className="mt-1 text-xs text-muted-foreground">
                            {alert.state} • {alert.affectedCount} affected • {new Date(alert.timestamp).toLocaleString('en-IN')}
                          </p>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm">
                          <Eye className="mr-1 h-3 w-3" /> Details
                        </Button>
                        <Button variant="outline" size="sm">
                          <Flag className="mr-1 h-3 w-3" /> Flag
                        </Button>
                        <Button variant="ghost" size="sm">
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Biometric Risk Table */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>Biometric Failure Risk Analysis</CardTitle>
            <CardDescription>High-risk demographic segments requiring outreach</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="pb-3 text-left font-medium">Age Group</th>
                    <th className="pb-3 text-left font-medium">State</th>
                    <th className="pb-3 text-left font-medium">Failure Rate</th>
                    <th className="pb-3 text-left font-medium">Common Issue</th>
                    <th className="pb-3 text-left font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {biometricRisks.slice(0, 5).map((risk) => (
                    <tr key={risk.id} className="border-b border-border/50">
                      <td className="py-3 font-medium">{risk.ageGroup}</td>
                      <td className="py-3">{risk.state}</td>
                      <td className="py-3">
                        <Badge variant={risk.failureProbability > 0.3 ? 'destructive' : 'secondary'}>
                          {formatPercentage(risk.failureProbability * 100)}
                        </Badge>
                      </td>
                      <td className="py-3 text-muted-foreground">{risk.commonIssue}</td>
                      <td className="py-3">
                        <Button variant="ghost" size="sm">Plan Outreach</Button>
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
