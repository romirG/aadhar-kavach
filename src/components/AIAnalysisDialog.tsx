import { useState } from 'react';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
    Loader2,
    Brain,
    AlertCircle,
    Target,
    Users,
    Clock,
    CheckCircle2,
    TrendingUp,
    XCircle
} from "lucide-react";
import { analyzeFinding, type FindingAnalysisResponse } from "@/services/api";

interface AIAnalysisDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    jobId: string;
    findingIndex: number;
    findingTitle: string;
}

export function AIAnalysisDialog({
    open,
    onOpenChange,
    jobId,
    findingIndex,
    findingTitle
}: AIAnalysisDialogProps) {
    const [analysis, setAnalysis] = useState<FindingAnalysisResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Load analysis when dialog opens
    const handleOpenChange = async (isOpen: boolean) => {
        onOpenChange(isOpen);
        
        if (isOpen && !analysis && !loading) {
            setLoading(true);
            setError(null);
            
            try {
                const result = await analyzeFinding(jobId, findingIndex);
                
                if (result.success) {
                    setAnalysis(result);
                } else {
                    setError(result.error || "Analysis failed");
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : "Failed to load analysis");
            } finally {
                setLoading(false);
            }
        }
    };

    const getPriorityColor = (priority: string) => {
        switch (priority.toLowerCase()) {
            case 'immediate':
            case 'urgent':
                return 'bg-destructive text-destructive-foreground';
            case 'high':
                return 'bg-warning text-warning-foreground';
            case 'medium':
                return 'bg-info/80 text-info-foreground';
            case 'low':
                return 'bg-muted text-muted-foreground';
            default:
                return 'bg-muted text-muted-foreground';
        }
    };

    const getSeverityColor = (severity: string) => {
        switch (severity.toLowerCase()) {
            case 'critical':
                return 'bg-destructive/20 text-destructive border-destructive';
            case 'high':
                return 'bg-warning/20 text-warning border-warning';
            case 'medium':
                return 'bg-info/20 text-info border-info';
            case 'low':
                return 'bg-success/20 text-success border-success';
            default:
                return 'bg-muted text-muted-foreground border-muted';
        }
    };

    return (
        <Dialog open={open} onOpenChange={handleOpenChange}>
            <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2 text-xl">
                        <Brain className="h-6 w-6 text-primary" />
                        AI In-Depth Analysis
                    </DialogTitle>
                    <DialogDescription>
                        Finding: {findingTitle}
                    </DialogDescription>
                </DialogHeader>

                {loading && (
                    <div className="flex items-center justify-center py-12">
                        <div className="text-center space-y-4">
                            <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto" />
                            <p className="text-muted-foreground">
                                Analyzing finding with AI...
                            </p>
                        </div>
                    </div>
                )}

                {error && (
                    <div className="rounded-lg border border-destructive bg-destructive/10 p-4">
                        <div className="flex items-start gap-3">
                            <XCircle className="h-5 w-5 text-destructive mt-0.5" />
                            <div>
                                <h4 className="font-semibold text-destructive">Analysis Failed</h4>
                                <p className="text-sm text-muted-foreground mt-1">{error}</p>
                            </div>
                        </div>
                    </div>
                )}

                {analysis && analysis.success && (
                    <div className="space-y-6">
                        {/* Detailed Analysis */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2 text-lg">
                                    <Brain className="h-5 w-5" />
                                    Detailed Analysis
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <p className="text-foreground leading-relaxed whitespace-pre-line">
                                    {analysis.analysis}
                                </p>
                            </CardContent>
                        </Card>

                        {/* Root Cause */}
                        {analysis.root_cause && (
                            <Card>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-lg">
                                        <Target className="h-5 w-5 text-warning" />
                                        Root Cause Analysis
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <p className="text-foreground leading-relaxed whitespace-pre-line">
                                        {analysis.root_cause}
                                    </p>
                                </CardContent>
                            </Card>
                        )}

                        {/* Impact Assessment */}
                        {analysis.impact_assessment && (
                            <Card>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-lg">
                                        <AlertCircle className="h-5 w-5 text-destructive" />
                                        Impact Assessment
                                    </CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                        <div>
                                            <p className="text-sm text-muted-foreground mb-2">Severity</p>
                                            <Badge className={getSeverityColor(analysis.impact_assessment.severity)}>
                                                {analysis.impact_assessment.severity}
                                            </Badge>
                                        </div>
                                        <div className="md:col-span-2">
                                            <p className="text-sm text-muted-foreground mb-2">Affected Scope</p>
                                            <p className="text-sm">{analysis.impact_assessment.affected_scope}</p>
                                        </div>
                                    </div>
                                    <Separator />
                                    <div>
                                        <p className="text-sm text-muted-foreground mb-2">Compliance Risk</p>
                                        <p className="text-sm">{analysis.impact_assessment.compliance_risk}</p>
                                    </div>
                                </CardContent>
                            </Card>
                        )}

                        {/* Recommended Actions */}
                        {analysis.recommended_actions && analysis.recommended_actions.length > 0 && (
                            <Card>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-lg">
                                        <CheckCircle2 className="h-5 w-5 text-success" />
                                        Recommended Actions
                                    </CardTitle>
                                    <CardDescription>
                                        Specific steps to address this finding
                                    </CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    {analysis.recommended_actions.map((action, idx) => (
                                        <div key={idx} className="rounded-lg border border-border p-4 space-y-3">
                                            <div className="flex items-start justify-between gap-4">
                                                <p className="font-medium flex-1">{action.action}</p>
                                                <Badge className={getPriorityColor(action.priority)}>
                                                    {action.priority}
                                                </Badge>
                                            </div>
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                                                <div className="flex items-center gap-2 text-muted-foreground">
                                                    <Users className="h-4 w-4" />
                                                    <span>{action.responsible_party}</span>
                                                </div>
                                                <div className="flex items-center gap-2 text-muted-foreground">
                                                    <Clock className="h-4 w-4" />
                                                    <span>{action.timeline}</span>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </CardContent>
                            </Card>
                        )}

                        {/* Monitoring Plan */}
                        {analysis.monitoring_plan && (
                            <Card>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-lg">
                                        <TrendingUp className="h-5 w-5 text-info" />
                                        Follow-up Monitoring Plan
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <p className="text-foreground leading-relaxed whitespace-pre-line">
                                        {analysis.monitoring_plan}
                                    </p>
                                </CardContent>
                            </Card>
                        )}

                        {/* Footer */}
                        <div className="flex justify-end pt-4">
                            <Button onClick={() => onOpenChange(false)}>
                                Close
                            </Button>
                        </div>
                    </div>
                )}
            </DialogContent>
        </Dialog>
    );
}
