import { useState } from 'react';
import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, Flag, X, Eye, Bell, Shield, Loader2, Play, Database } from 'lucide-react';
import { useMLDatasets, useStartAnalysis, useAnalysisStatus, useAnalysisResults } from '@/hooks/useData';
import { formatPercentage } from '@/data/dataUtils';

export default function Anomalies() {
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const { data: datasetsData, isLoading: loadingDatasets } = useMLDatasets();
  const startAnalysis = useStartAnalysis();
  const { data: statusData } = useAnalysisStatus(jobId);
  const { data: resultsData, isLoading: loadingResults } = useAnalysisResults(
    statusData?.status === 'completed' ? jobId : null
  );

  const handleStartAnalysis = async (datasetId: string) => {
    setSelectedDataset(datasetId);
    try {
      const result = await startAnalysis.mutateAsync({ datasetId, limit: 500 });
      setJobId(result.job_id);
    } catch (error) {
      console.error('Failed to start analysis:', error);
    }
  };

  const getSeverityStyles = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high': return { badge: 'bg-destructive text-destructive-foreground', bg: 'bg-destructive/10', text: 'text-destructive' };
      case 'medium': return { badge: 'bg-warning text-warning-foreground', bg: 'bg-warning/10', text: 'text-warning' };
      default: return { badge: 'bg-muted text-muted-foreground', bg: 'bg-muted', text: 'text-muted-foreground' };
    }
  };

  const anomalies = resultsData?.anomalies || [];
  const criticalAlerts = anomalies.filter((a: { risk_level: string }) =>
    a.risk_level === 'High' || a.risk_level === 'Medium'
  );

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">ML Anomaly & Fraud Detection</h1>
            <p className="text-muted-foreground">
              AI-powered detection using Isolation Forest, Autoencoder & HDBSCAN
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

        {/* Dataset Selection */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Select Dataset to Analyze
            </CardTitle>
            <CardDescription>Choose a data.gov.in dataset for ML analysis</CardDescription>
          </CardHeader>
          <CardContent>
            {loadingDatasets ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin" />
                <span className="ml-2">Loading datasets...</span>
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-3">
                {datasetsData?.datasets?.map((dataset: { id: string; name: string; description: string }) => (
                  <Card
                    key={dataset.id}
                    className={`cursor-pointer transition-all hover:shadow-md ${selectedDataset === dataset.id ? 'ring-2 ring-primary' : ''
                      }`}
                    onClick={() => setSelectedDataset(dataset.id)}
                  >
                    <CardContent className="p-4">
                      <h3 className="font-semibold">{dataset.name}</h3>
                      <p className="text-sm text-muted-foreground mt-1">{dataset.description}</p>
                      <Button
                        className="mt-3 w-full"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleStartAnalysis(dataset.id);
                        }}
                        disabled={startAnalysis.isPending}
                      >
                        {startAnalysis.isPending && selectedDataset === dataset.id ? (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : (
                          <Play className="mr-2 h-4 w-4" />
                        )}
                        Analyze
                      </Button>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Analysis Status */}
        {jobId && statusData && (
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>Analysis Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-4">
                {statusData.status === 'processing' && (
                  <Loader2 className="h-6 w-6 animate-spin text-primary" />
                )}
                {statusData.status === 'completed' && (
                  <Shield className="h-6 w-6 text-success" />
                )}
                <div>
                  <p className="font-medium capitalize">{statusData.status}</p>
                  <p className="text-sm text-muted-foreground">{statusData.message}</p>
                  {statusData.progress !== undefined && (
                    <div className="mt-2 w-64 bg-muted rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full transition-all"
                        style={{ width: `${statusData.progress}%` }}
                      />
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Stats */}
        {resultsData && (
          <div className="grid gap-4 md:grid-cols-4">
            <Card className="border-destructive/50 shadow-card">
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="rounded-full bg-destructive/10 p-2">
                    <AlertTriangle className="h-5 w-5 text-destructive" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold">{criticalAlerts.length}</p>
                    <p className="text-sm text-muted-foreground">High/Medium Alerts</p>
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
                    <p className="text-2xl font-bold">{resultsData.anomaly_count}</p>
                    <p className="text-sm text-muted-foreground">Total Anomalies</p>
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
                    <p className="text-2xl font-bold">{resultsData.total_records}</p>
                    <p className="text-sm text-muted-foreground">Records Analyzed</p>
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
                    <p className="text-2xl font-bold">{formatPercentage(100 - resultsData.anomaly_percentage)}</p>
                    <p className="text-sm text-muted-foreground">System Integrity</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Models Used */}
        {resultsData?.models_used && (
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>ML Models Used</CardTitle>
              <CardDescription>Ensemble anomaly detection</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {resultsData.models_used.map((model: string) => (
                  <Badge key={model} variant="secondary" className="text-sm py-1 px-3">
                    {model === 'isolation_forest' && '🌲 Isolation Forest'}
                    {model === 'autoencoder' && '🧠 Autoencoder'}
                    {model === 'hdbscan' && '📊 HDBSCAN'}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Alerts Table */}
        {anomalies.length > 0 && (
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>Detected Anomalies</CardTitle>
              <CardDescription>Sorted by risk score</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {anomalies.slice(0, 10).map((alert: {
                  record_id: string;
                  anomaly_score: number;
                  risk_level: string;
                  confidence: number;
                  reasons: string[];
                }) => {
                  const styles = getSeverityStyles(alert.risk_level);
                  return (
                    <div key={alert.record_id} className={`rounded-lg border border-border p-4 ${styles.bg}`}>
                      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                        <div className="flex items-start gap-3">
                          <AlertTriangle className={`h-5 w-5 mt-0.5 ${styles.text}`} />
                          <div>
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="font-medium">Record #{alert.record_id}</span>
                              <Badge className={styles.badge}>{alert.risk_level}</Badge>
                              <Badge variant="outline">Score: {(alert.anomaly_score * 10).toFixed(1)}</Badge>
                              <Badge variant="outline">Confidence: {formatPercentage(alert.confidence * 100)}</Badge>
                            </div>
                            <div className="mt-2 space-y-1">
                              {alert.reasons?.slice(0, 3).map((reason: string, idx: number) => (
                                <p key={idx} className="text-sm text-muted-foreground">{reason}</p>
                              ))}
                            </div>
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
        )}

        {/* Empty State */}
        {!jobId && (
          <Card className="shadow-card">
            <CardContent className="py-12">
              <div className="text-center">
                <Shield className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold">Select a Dataset to Begin</h3>
                <p className="text-muted-foreground mt-2">
                  Choose one of the datasets above and click "Analyze" to run ML-powered fraud detection.
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </DashboardLayout>
  );
}
