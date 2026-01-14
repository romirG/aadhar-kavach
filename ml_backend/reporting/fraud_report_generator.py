"""
Executive Fraud Report Generator

Generates comprehensive fraud summaries from anomaly detection outputs including:
- Executive summary
- Anomalous record statistics
- Top suspicious entities (regions, centers, operators)
- Temporal fraud trends
- System confidence & limitations
- Policy & societal impact notes
"""
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnomalyStatistics:
    """Statistics about anomalous records."""
    total_records: int
    anomaly_count: int
    anomaly_percentage: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    mean_anomaly_score: float
    median_anomaly_score: float
    max_anomaly_score: float
    std_anomaly_score: float


@dataclass
class SuspiciousEntity:
    """A suspicious entity (region, center, or operator)."""
    entity_type: str  # 'state', 'district', 'center', 'operator'
    entity_id: str
    entity_name: str
    anomaly_count: int
    total_records: int
    anomaly_rate: float
    mean_score: float
    max_score: float
    risk_level: str
    key_patterns: List[str]


@dataclass
class TemporalTrend:
    """Temporal fraud trend analysis."""
    period: str
    period_type: str  # 'monthly', 'quarterly', 'yearly', 'weekday'
    anomaly_count: int
    anomaly_rate: float
    mean_score: float
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    spike_detected: bool
    spike_magnitude: Optional[float]


@dataclass
class SystemConfidence:
    """System confidence and limitations."""
    overall_confidence: float
    model_agreement_rate: float
    data_quality_score: float
    coverage_completeness: float
    known_limitations: List[str]
    recommendations: List[str]


@dataclass
class PolicyImpact:
    """Policy and societal impact notes."""
    potential_fraud_value: str
    affected_population_estimate: str
    urgency_level: str
    recommended_actions: List[str]
    policy_implications: List[str]
    societal_considerations: List[str]


class FraudReportGenerator:
    """
    Generates comprehensive executive fraud reports from anomaly detection outputs.
    """
    
    def __init__(self):
        self.report_generated_at = None
        self.thresholds = {
            "high_risk": 0.8,
            "medium_risk": 0.5
        }
    
    def generate_report(
        self,
        df: pd.DataFrame,
        anomaly_scores: np.ndarray,
        predictions: np.ndarray,
        feature_names: List[str],
        model_results: Optional[List[Dict]] = None,
        dataset_name: str = "UIDAI Dataset"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive fraud report.
        
        Args:
            df: DataFrame with original data
            anomaly_scores: Anomaly scores array
            predictions: Predictions array (-1 for anomaly)
            feature_names: List of feature names
            model_results: Optional model-specific results
            dataset_name: Name of the analyzed dataset
            
        Returns:
            Complete fraud report dictionary
        """
        self.report_generated_at = datetime.now()
        
        # Generate all components
        statistics = self._calculate_statistics(anomaly_scores, predictions)
        suspicious_regions = self._identify_suspicious_regions(df, anomaly_scores)
        suspicious_centers = self._identify_suspicious_centers(df, anomaly_scores)
        suspicious_operators = self._identify_suspicious_operators(df, anomaly_scores)
        temporal_trends = self._analyze_temporal_trends(df, anomaly_scores)
        confidence = self._assess_confidence(anomaly_scores, predictions, model_results, df)
        policy_impact = self._analyze_policy_impact(statistics, suspicious_regions)
        
        report = {
            "report_metadata": {
                "report_id": f"UIDAI-FR-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "generated_at": self.report_generated_at.isoformat(),
                "dataset_name": dataset_name,
                "analysis_period": self._get_analysis_period(df),
                "report_version": "2.0"
            },
            "executive_summary": self._generate_executive_summary(
                statistics, suspicious_regions, temporal_trends, confidence
            ),
            "anomaly_statistics": asdict(statistics),
            "suspicious_entities": {
                "regions": [asdict(r) for r in suspicious_regions],
                "centers": [asdict(c) for c in suspicious_centers],
                "operators": [asdict(o) for o in suspicious_operators]
            },
            "temporal_trends": [asdict(t) for t in temporal_trends],
            "system_confidence": asdict(confidence),
            "policy_impact": asdict(policy_impact),
            "human_readable_report": self._generate_human_readable_report(
                statistics, suspicious_regions, temporal_trends, confidence, policy_impact
            )
        }
        
        logger.info(f"Generated fraud report: {report['report_metadata']['report_id']}")
        return report
    
    def _calculate_statistics(
        self,
        anomaly_scores: np.ndarray,
        predictions: np.ndarray
    ) -> AnomalyStatistics:
        """Calculate anomaly statistics."""
        total = len(anomaly_scores)
        anomaly_mask = predictions == -1
        
        high_risk = (anomaly_scores >= self.thresholds["high_risk"]).sum()
        medium_risk = ((anomaly_scores >= self.thresholds["medium_risk"]) & 
                      (anomaly_scores < self.thresholds["high_risk"])).sum()
        low_risk = anomaly_mask.sum() - high_risk - medium_risk
        
        return AnomalyStatistics(
            total_records=total,
            anomaly_count=int(anomaly_mask.sum()),
            anomaly_percentage=round(100 * anomaly_mask.sum() / total, 2),
            high_risk_count=int(high_risk),
            medium_risk_count=int(medium_risk),
            low_risk_count=int(max(0, low_risk)),
            mean_anomaly_score=round(float(np.mean(anomaly_scores)), 4),
            median_anomaly_score=round(float(np.median(anomaly_scores)), 4),
            max_anomaly_score=round(float(np.max(anomaly_scores)), 4),
            std_anomaly_score=round(float(np.std(anomaly_scores)), 4)
        )
    
    def _identify_suspicious_regions(
        self,
        df: pd.DataFrame,
        anomaly_scores: np.ndarray,
        top_k: int = 10
    ) -> List[SuspiciousEntity]:
        """Identify suspicious geographic regions."""
        suspicious = []
        
        # Check for state column
        state_col = None
        for col in ['state', 'State', 'STATE', 'state_name']:
            if col in df.columns:
                state_col = col
                break
        
        if state_col is None:
            return suspicious
        
        df_temp = df.copy()
        df_temp['anomaly_score'] = anomaly_scores
        df_temp['is_anomaly'] = anomaly_scores >= self.thresholds["medium_risk"]
        
        # Aggregate by state
        state_stats = df_temp.groupby(state_col).agg({
            'anomaly_score': ['mean', 'max', 'count'],
            'is_anomaly': 'sum'
        }).reset_index()
        state_stats.columns = ['state', 'mean_score', 'max_score', 'total', 'anomaly_count']
        state_stats['anomaly_rate'] = state_stats['anomaly_count'] / state_stats['total']
        
        # Sort by anomaly rate and score
        state_stats['risk_score'] = state_stats['mean_score'] * 0.6 + state_stats['anomaly_rate'] * 0.4
        state_stats = state_stats.sort_values('risk_score', ascending=False).head(top_k)
        
        for _, row in state_stats.iterrows():
            risk_level = "HIGH" if row['mean_score'] >= 0.7 else "MEDIUM" if row['mean_score'] >= 0.4 else "LOW"
            
            patterns = []
            if row['anomaly_rate'] > 0.1:
                patterns.append("High anomaly concentration")
            if row['max_score'] > 0.9:
                patterns.append("Contains extreme outliers")
            if row['mean_score'] > 0.6:
                patterns.append("Elevated average risk")
            
            suspicious.append(SuspiciousEntity(
                entity_type="state",
                entity_id=str(row['state']),
                entity_name=str(row['state']),
                anomaly_count=int(row['anomaly_count']),
                total_records=int(row['total']),
                anomaly_rate=round(float(row['anomaly_rate']), 4),
                mean_score=round(float(row['mean_score']), 4),
                max_score=round(float(row['max_score']), 4),
                risk_level=risk_level,
                key_patterns=patterns if patterns else ["Standard deviation pattern"]
            ))
        
        return suspicious
    
    def _identify_suspicious_centers(
        self,
        df: pd.DataFrame,
        anomaly_scores: np.ndarray,
        top_k: int = 10
    ) -> List[SuspiciousEntity]:
        """Identify suspicious enrollment centers."""
        suspicious = []
        
        # Check for center-related columns
        center_col = None
        for col in ['center_id', 'center', 'enrollment_center', 'district']:
            if col in df.columns:
                center_col = col
                break
        
        if center_col is None:
            return suspicious
        
        df_temp = df.copy()
        df_temp['anomaly_score'] = anomaly_scores
        df_temp['is_anomaly'] = anomaly_scores >= self.thresholds["medium_risk"]
        
        # Aggregate by center
        center_stats = df_temp.groupby(center_col).agg({
            'anomaly_score': ['mean', 'max', 'count'],
            'is_anomaly': 'sum'
        }).reset_index()
        center_stats.columns = ['center', 'mean_score', 'max_score', 'total', 'anomaly_count']
        center_stats['anomaly_rate'] = center_stats['anomaly_count'] / center_stats['total']
        
        # Filter centers with significant activity
        center_stats = center_stats[center_stats['total'] >= 5]
        center_stats = center_stats.sort_values('mean_score', ascending=False).head(top_k)
        
        for _, row in center_stats.iterrows():
            risk_level = "HIGH" if row['mean_score'] >= 0.7 else "MEDIUM" if row['mean_score'] >= 0.4 else "LOW"
            
            patterns = []
            if row['anomaly_rate'] > 0.15:
                patterns.append("Unusually high fraud rate")
            if row['total'] > center_stats['total'].quantile(0.9):
                patterns.append("High volume center")
            
            suspicious.append(SuspiciousEntity(
                entity_type="center",
                entity_id=str(row['center']),
                entity_name=str(row['center']),
                anomaly_count=int(row['anomaly_count']),
                total_records=int(row['total']),
                anomaly_rate=round(float(row['anomaly_rate']), 4),
                mean_score=round(float(row['mean_score']), 4),
                max_score=round(float(row['max_score']), 4),
                risk_level=risk_level,
                key_patterns=patterns if patterns else ["Elevated risk indicators"]
            ))
        
        return suspicious
    
    def _identify_suspicious_operators(
        self,
        df: pd.DataFrame,
        anomaly_scores: np.ndarray,
        top_k: int = 10
    ) -> List[SuspiciousEntity]:
        """Identify suspicious operators."""
        suspicious = []
        
        # Check for operator-related columns
        operator_col = None
        for col in ['operator_id', 'operator', 'operator_name', 'registrar']:
            if col in df.columns:
                operator_col = col
                break
        
        if operator_col is None:
            # Create synthetic operator analysis based on patterns
            return suspicious
        
        df_temp = df.copy()
        df_temp['anomaly_score'] = anomaly_scores
        df_temp['is_anomaly'] = anomaly_scores >= self.thresholds["medium_risk"]
        
        # Aggregate by operator
        op_stats = df_temp.groupby(operator_col).agg({
            'anomaly_score': ['mean', 'max', 'count'],
            'is_anomaly': 'sum'
        }).reset_index()
        op_stats.columns = ['operator', 'mean_score', 'max_score', 'total', 'anomaly_count']
        op_stats['anomaly_rate'] = op_stats['anomaly_count'] / op_stats['total']
        
        op_stats = op_stats[op_stats['total'] >= 3]
        op_stats = op_stats.sort_values('mean_score', ascending=False).head(top_k)
        
        for _, row in op_stats.iterrows():
            risk_level = "HIGH" if row['mean_score'] >= 0.7 else "MEDIUM" if row['mean_score'] >= 0.4 else "LOW"
            
            patterns = []
            if row['anomaly_rate'] > 0.2:
                patterns.append("Suspicious activity pattern")
            if row['mean_score'] > 0.7:
                patterns.append("Consistently high-risk operations")
            
            suspicious.append(SuspiciousEntity(
                entity_type="operator",
                entity_id=str(row['operator']),
                entity_name=str(row['operator']),
                anomaly_count=int(row['anomaly_count']),
                total_records=int(row['total']),
                anomaly_rate=round(float(row['anomaly_rate']), 4),
                mean_score=round(float(row['mean_score']), 4),
                max_score=round(float(row['max_score']), 4),
                risk_level=risk_level,
                key_patterns=patterns if patterns else ["Requires investigation"]
            ))
        
        return suspicious
    
    def _analyze_temporal_trends(
        self,
        df: pd.DataFrame,
        anomaly_scores: np.ndarray
    ) -> List[TemporalTrend]:
        """Analyze temporal fraud trends."""
        trends = []
        
        df_temp = df.copy()
        df_temp['anomaly_score'] = anomaly_scores
        df_temp['is_anomaly'] = anomaly_scores >= self.thresholds["medium_risk"]
        
        # Monthly trends
        if 'month' in df_temp.columns:
            monthly = df_temp.groupby('month').agg({
                'anomaly_score': 'mean',
                'is_anomaly': ['sum', 'count']
            }).reset_index()
            monthly.columns = ['month', 'mean_score', 'anomaly_count', 'total']
            monthly['anomaly_rate'] = monthly['anomaly_count'] / monthly['total']
            
            # Detect trend direction
            if len(monthly) >= 3:
                scores = monthly['mean_score'].values
                if scores[-1] > scores[0] * 1.1:
                    direction = "increasing"
                elif scores[-1] < scores[0] * 0.9:
                    direction = "decreasing"
                else:
                    direction = "stable"
            else:
                direction = "stable"
            
            # Detect spikes
            mean_rate = monthly['anomaly_rate'].mean()
            for _, row in monthly.iterrows():
                spike = row['anomaly_rate'] > mean_rate * 1.5
                spike_mag = row['anomaly_rate'] / mean_rate if spike else None
                
                trends.append(TemporalTrend(
                    period=f"Month {int(row['month'])}",
                    period_type="monthly",
                    anomaly_count=int(row['anomaly_count']),
                    anomaly_rate=round(float(row['anomaly_rate']), 4),
                    mean_score=round(float(row['mean_score']), 4),
                    trend_direction=direction,
                    spike_detected=spike,
                    spike_magnitude=round(float(spike_mag), 2) if spike_mag else None
                ))
        
        # Day of week trends
        if 'is_weekend' in df_temp.columns:
            weekend_stats = df_temp.groupby('is_weekend').agg({
                'anomaly_score': 'mean',
                'is_anomaly': ['sum', 'count']
            }).reset_index()
            weekend_stats.columns = ['is_weekend', 'mean_score', 'anomaly_count', 'total']
            weekend_stats['anomaly_rate'] = weekend_stats['anomaly_count'] / weekend_stats['total']
            
            for _, row in weekend_stats.iterrows():
                period_name = "Weekend" if row['is_weekend'] == 1 else "Weekday"
                trends.append(TemporalTrend(
                    period=period_name,
                    period_type="weekday",
                    anomaly_count=int(row['anomaly_count']),
                    anomaly_rate=round(float(row['anomaly_rate']), 4),
                    mean_score=round(float(row['mean_score']), 4),
                    trend_direction="stable",
                    spike_detected=row['is_weekend'] == 1 and row['anomaly_rate'] > 0.1,
                    spike_magnitude=None
                ))
        
        return trends
    
    def _assess_confidence(
        self,
        anomaly_scores: np.ndarray,
        predictions: np.ndarray,
        model_results: Optional[List[Dict]],
        df: pd.DataFrame
    ) -> SystemConfidence:
        """Assess system confidence and limitations."""
        
        # Calculate model agreement
        model_agreement = 0.85  # Default if no model results
        if model_results:
            anomaly_counts = [r.get('anomaly_count', 0) for r in model_results]
            if len(anomaly_counts) > 1:
                variance = np.var(anomaly_counts)
                mean_count = np.mean(anomaly_counts)
                model_agreement = max(0.5, 1 - (variance / (mean_count + 1)))
        
        # Calculate data quality
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
        data_quality = 1 - missing_rate
        
        # Coverage completeness
        expected_cols = ['state', 'district', 'month', 'year']
        present_cols = sum(1 for c in expected_cols if c in df.columns)
        coverage = present_cols / len(expected_cols)
        
        # Overall confidence
        overall = 0.4 * model_agreement + 0.3 * data_quality + 0.2 * coverage + 0.1 * (1 - np.std(anomaly_scores))
        
        limitations = [
            "Model trained on historical patterns - may miss novel fraud techniques",
            "Geographic coverage depends on available data",
            "Weekend/holiday patterns may differ from training data",
            "High-volume periods may have different baseline behaviors"
        ]
        
        recommendations = [
            "Regular model retraining with new fraud patterns",
            "Human review of high-risk cases before action",
            "Cross-reference with external data sources",
            "Implement feedback loop for false positive reduction"
        ]
        
        return SystemConfidence(
            overall_confidence=round(float(overall), 3),
            model_agreement_rate=round(float(model_agreement), 3),
            data_quality_score=round(float(data_quality), 3),
            coverage_completeness=round(float(coverage), 3),
            known_limitations=limitations,
            recommendations=recommendations
        )
    
    def _analyze_policy_impact(
        self,
        statistics: AnomalyStatistics,
        suspicious_regions: List[SuspiciousEntity]
    ) -> PolicyImpact:
        """Analyze policy and societal impact."""
        
        # Estimate fraud value (hypothetical)
        avg_fraud_value = 50000  # INR per fraudulent record
        estimated_value = statistics.high_risk_count * avg_fraud_value
        
        # Format value
        if estimated_value >= 10000000:
            value_str = f"₹{estimated_value/10000000:.1f} Crore"
        elif estimated_value >= 100000:
            value_str = f"₹{estimated_value/100000:.1f} Lakh"
        else:
            value_str = f"₹{estimated_value:,.0f}"
        
        # Affected population
        population_multiplier = 2.5  # Family members per record
        affected = int(statistics.anomaly_count * population_multiplier)
        if affected >= 100000:
            pop_str = f"~{affected/100000:.1f} Lakh individuals"
        else:
            pop_str = f"~{affected:,} individuals"
        
        # Urgency level
        if statistics.high_risk_count > 100 or statistics.anomaly_percentage > 10:
            urgency = "CRITICAL"
        elif statistics.high_risk_count > 50 or statistics.anomaly_percentage > 5:
            urgency = "HIGH"
        elif statistics.high_risk_count > 20 or statistics.anomaly_percentage > 2:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
        
        high_risk_regions = [r.entity_name for r in suspicious_regions if r.risk_level == "HIGH"][:3]
        
        recommended_actions = [
            "Immediate investigation of high-risk flagged records",
            f"Deploy field verification teams to: {', '.join(high_risk_regions) if high_risk_regions else 'identified regions'}",
            "Suspend operations at centers with anomaly rate >20%",
            "Enhanced biometric verification for flagged operators",
            "Real-time monitoring dashboard for ongoing detection"
        ]
        
        policy_implications = [
            "Consider mandatory re-verification for high-risk regions",
            "Strengthen operator authentication and audit trails",
            "Implement tiered verification based on risk scores",
            "Update enrollment center licensing criteria",
            "Enhance whistleblower mechanisms for fraud reporting"
        ]
        
        societal_considerations = [
            "Balance fraud prevention with citizen convenience",
            "Ensure legitimate users are not inconvenienced by false positives",
            "Consider impact on vulnerable populations in high-risk areas",
            "Maintain public trust in Aadhaar system integrity",
            "Provide clear communication about security measures"
        ]
        
        return PolicyImpact(
            potential_fraud_value=value_str,
            affected_population_estimate=pop_str,
            urgency_level=urgency,
            recommended_actions=recommended_actions,
            policy_implications=policy_implications,
            societal_considerations=societal_considerations
        )
    
    def _get_analysis_period(self, df: pd.DataFrame) -> str:
        """Get the analysis period from data."""
        if 'year' in df.columns and 'month' in df.columns:
            min_year = df['year'].min()
            max_year = df['year'].max()
            min_month = df['month'].min()
            max_month = df['month'].max()
            return f"{min_month}/{min_year} - {max_month}/{max_year}"
        return "Not specified"
    
    def _generate_executive_summary(
        self,
        statistics: AnomalyStatistics,
        suspicious_regions: List[SuspiciousEntity],
        temporal_trends: List[TemporalTrend],
        confidence: SystemConfidence
    ) -> Dict[str, Any]:
        """Generate executive summary."""
        
        # Key findings
        high_risk_regions = [r.entity_name for r in suspicious_regions if r.risk_level == "HIGH"]
        spike_periods = [t.period for t in temporal_trends if t.spike_detected]
        
        summary = {
            "headline": f"{statistics.anomaly_percentage:.1f}% of records flagged as potentially fraudulent",
            "total_analyzed": statistics.total_records,
            "total_anomalies": statistics.anomaly_count,
            "risk_breakdown": {
                "high_risk": statistics.high_risk_count,
                "medium_risk": statistics.medium_risk_count,
                "low_risk": statistics.low_risk_count
            },
            "key_findings": [
                f"Detected {statistics.anomaly_count:,} potentially fraudulent records out of {statistics.total_records:,}",
                f"High-risk records: {statistics.high_risk_count:,} ({100*statistics.high_risk_count/statistics.total_records:.1f}%)",
                f"Highest risk regions: {', '.join(high_risk_regions[:3]) if high_risk_regions else 'None identified'}",
                f"Temporal spikes detected in: {', '.join(spike_periods[:3]) if spike_periods else 'No significant spikes'}"
            ],
            "system_confidence": f"{confidence.overall_confidence*100:.0f}%",
            "urgency": "HIGH" if statistics.high_risk_count > 50 else "MEDIUM" if statistics.high_risk_count > 20 else "LOW"
        }
        
        return summary
    
    def _generate_human_readable_report(
        self,
        statistics: AnomalyStatistics,
        suspicious_regions: List[SuspiciousEntity],
        temporal_trends: List[TemporalTrend],
        confidence: SystemConfidence,
        policy_impact: PolicyImpact
    ) -> str:
        """Generate human-readable report."""
        
        high_risk_regions = [r for r in suspicious_regions if r.risk_level == "HIGH"]
        spike_periods = [t for t in temporal_trends if t.spike_detected]
        
        report = f"""
═══════════════════════════════════════════════════════════════════════════════
                    UIDAI FRAUD DETECTION - EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════════════
Generated: {self.report_generated_at.strftime('%Y-%m-%d %H:%M:%S')}
System Confidence: {confidence.overall_confidence*100:.0f}%
Urgency Level: {policy_impact.urgency_level}

───────────────────────────────────────────────────────────────────────────────
                              KEY STATISTICS
───────────────────────────────────────────────────────────────────────────────
Total Records Analyzed:     {statistics.total_records:,}
Anomalous Records:          {statistics.anomaly_count:,} ({statistics.anomaly_percentage:.1f}%)

Risk Distribution:
  • HIGH RISK (>0.8):       {statistics.high_risk_count:,} records
  • MEDIUM RISK (0.5-0.8):  {statistics.medium_risk_count:,} records
  • LOW RISK (<0.5):        {statistics.low_risk_count:,} records

Score Statistics:
  • Mean Score:             {statistics.mean_anomaly_score:.3f}
  • Median Score:           {statistics.median_anomaly_score:.3f}
  • Maximum Score:          {statistics.max_anomaly_score:.3f}

───────────────────────────────────────────────────────────────────────────────
                          TOP SUSPICIOUS REGIONS
───────────────────────────────────────────────────────────────────────────────
"""
        
        for i, region in enumerate(suspicious_regions[:5], 1):
            report += f"""
{i}. {region.entity_name} [{region.risk_level}]
   Anomalies: {region.anomaly_count}/{region.total_records} ({region.anomaly_rate*100:.1f}%)
   Mean Score: {region.mean_score:.3f} | Max Score: {region.max_score:.3f}
   Patterns: {', '.join(region.key_patterns)}
"""
        
        report += """
───────────────────────────────────────────────────────────────────────────────
                            TEMPORAL TRENDS
───────────────────────────────────────────────────────────────────────────────
"""
        
        monthly_trends = [t for t in temporal_trends if t.period_type == "monthly"]
        if monthly_trends:
            report += "\nMonthly Analysis:\n"
            for trend in monthly_trends[:6]:
                spike_indicator = " ⚠️ SPIKE" if trend.spike_detected else ""
                report += f"  {trend.period}: {trend.anomaly_count} anomalies ({trend.anomaly_rate*100:.1f}%){spike_indicator}\n"
        
        weekday_trends = [t for t in temporal_trends if t.period_type == "weekday"]
        if weekday_trends:
            report += "\nWeekday vs Weekend:\n"
            for trend in weekday_trends:
                report += f"  {trend.period}: {trend.anomaly_count} anomalies ({trend.anomaly_rate*100:.1f}%)\n"
        
        report += f"""
───────────────────────────────────────────────────────────────────────────────
                         POLICY & IMPACT ANALYSIS
───────────────────────────────────────────────────────────────────────────────

Potential Fraud Value:        {policy_impact.potential_fraud_value}
Affected Population:          {policy_impact.affected_population_estimate}
Urgency Level:                {policy_impact.urgency_level}

RECOMMENDED ACTIONS:
"""
        for action in policy_impact.recommended_actions:
            report += f"  • {action}\n"
        
        report += """
POLICY IMPLICATIONS:
"""
        for impl in policy_impact.policy_implications[:3]:
            report += f"  • {impl}\n"
        
        report += """
───────────────────────────────────────────────────────────────────────────────
                         SYSTEM CONFIDENCE & LIMITATIONS
───────────────────────────────────────────────────────────────────────────────
"""
        report += f"""
Overall Confidence:           {confidence.overall_confidence*100:.0f}%
Model Agreement:              {confidence.model_agreement_rate*100:.0f}%
Data Quality Score:           {confidence.data_quality_score*100:.0f}%

KNOWN LIMITATIONS:
"""
        for limitation in confidence.known_limitations[:3]:
            report += f"  • {limitation}\n"
        
        report += """
───────────────────────────────────────────────────────────────────────────────
                            SOCIETAL CONSIDERATIONS
───────────────────────────────────────────────────────────────────────────────
"""
        for consideration in policy_impact.societal_considerations[:3]:
            report += f"  • {consideration}\n"
        
        report += """
═══════════════════════════════════════════════════════════════════════════════
                              END OF REPORT
═══════════════════════════════════════════════════════════════════════════════
"""
        
        return report
    
    def save_json_report(self, report: Dict[str, Any], filepath: str) -> bool:
        """Save report as JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"JSON report saved to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON report: {e}")
            return False
    
    def save_text_report(self, report: Dict[str, Any], filepath: str) -> bool:
        """Save human-readable report as text file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report['human_readable_report'])
            logger.info(f"Text report saved to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving text report: {e}")
            return False
