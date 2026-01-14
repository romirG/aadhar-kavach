"""
Executive Fraud Report Generation Demo

Demonstrates the fraud report generator with sample data.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import json

from reporting.fraud_report_generator import FraudReportGenerator


def create_sample_data():
    """Create sample UIDAI data for demonstration."""
    np.random.seed(42)
    n_samples = 500
    
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    data = {
        'date': dates,
        'state': np.random.choice(['Maharashtra', 'Karnataka', 'Delhi', 'Gujarat', 
                                  'Tamil Nadu', 'Uttar Pradesh', 'West Bengal',
                                  'Rajasthan', 'Bihar', 'Andhra Pradesh'], n_samples),
        'district': np.random.choice(['District_A', 'District_B', 'District_C', 
                                     'District_D', 'District_E'], n_samples),
        'month': dates.month,
        'year': dates.year,
        'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
        'total_enrolments': np.random.poisson(50, n_samples),
        'total_demo_updates': np.random.poisson(20, n_samples),
        'total_bio_updates': np.random.poisson(15, n_samples),
        'operator_id': np.random.choice([f'OP_{i:03d}' for i in range(20)], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add some anomalies
    # High anomalies in Maharashtra
    maharashtra_mask = df['state'] == 'Maharashtra'
    anomaly_indices = np.where(maharashtra_mask)[0][:15]
    df.loc[anomaly_indices, 'total_enrolments'] = np.random.randint(300, 600, len(anomaly_indices))
    
    # Weekend fraud
    weekend_mask = df['is_weekend'] == 1
    weekend_indices = np.where(weekend_mask)[0][:10]
    df.loc[weekend_indices, 'total_bio_updates'] = np.random.randint(100, 200, len(weekend_indices))
    
    return df


def generate_scores(df):
    """Generate synthetic anomaly scores."""
    n = len(df)
    
    # Base scores
    anomaly_scores = np.random.beta(2, 5, n)
    
    # Boost for high values
    for col in ['total_enrolments', 'total_demo_updates', 'total_bio_updates']:
        threshold = df[col].quantile(0.9)
        high_mask = df[col] > threshold
        anomaly_scores[high_mask] = np.clip(anomaly_scores[high_mask] + 0.3, 0, 1)
    
    # Weekend boost
    weekend_mask = df['is_weekend'] == 1
    anomaly_scores[weekend_mask] = np.clip(anomaly_scores[weekend_mask] + 0.15, 0, 1)
    
    # Maharashtra boost
    mh_mask = df['state'] == 'Maharashtra'
    anomaly_scores[mh_mask] = np.clip(anomaly_scores[mh_mask] + 0.1, 0, 1)
    
    # Predictions
    predictions = np.where(anomaly_scores > 0.5, -1, 1)
    
    return anomaly_scores, predictions


def main():
    """Run the fraud report generation demo."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  UIDAI FRAUD DETECTION - EXECUTIVE REPORT GENERATION DEMO".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'report_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    print("Generating sample data...")
    df = create_sample_data()
    anomaly_scores, predictions = generate_scores(df)
    print(f"✓ Generated {len(df)} records")
    print(f"✓ Anomalies: {(predictions == -1).sum()} ({100*(predictions == -1).sum()/len(df):.1f}%)")
    
    # Model results simulation
    model_results = [
        {'model_name': 'isolation_forest', 'anomaly_count': int((predictions == -1).sum() * 1.1)},
        {'model_name': 'hdbscan', 'anomaly_count': int((predictions == -1).sum() * 0.9)},
        {'model_name': 'autoencoder', 'anomaly_count': int((predictions == -1).sum() * 1.0)}
    ]
    
    # Generate report
    print("\nGenerating executive fraud report...")
    generator = FraudReportGenerator()
    
    feature_names = ['total_enrolments', 'total_demo_updates', 'total_bio_updates']
    
    report = generator.generate_report(
        df=df,
        anomaly_scores=anomaly_scores,
        predictions=predictions,
        feature_names=feature_names,
        model_results=model_results,
        dataset_name="Aadhaar Monthly Enrolment Data (Demo)"
    )
    
    # Print executive summary
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    summary = report['executive_summary']
    print(f"\n📊 {summary['headline']}")
    print(f"\nTotal Analyzed: {summary['total_analyzed']:,}")
    print(f"Total Anomalies: {summary['total_anomalies']:,}")
    print(f"System Confidence: {summary['system_confidence']}")
    print(f"Urgency Level: {summary['urgency']}")
    
    print("\n🔍 Key Findings:")
    for finding in summary['key_findings']:
        print(f"  • {finding}")
    
    # Print suspicious regions
    print("\n" + "=" * 80)
    print("TOP SUSPICIOUS REGIONS")
    print("=" * 80)
    for i, region in enumerate(report['suspicious_entities']['regions'][:5], 1):
        print(f"\n{i}. {region['entity_name']} [{region['risk_level']}]")
        print(f"   Anomaly Rate: {region['anomaly_rate']*100:.1f}%")
        print(f"   Mean Score: {region['mean_score']:.3f}")
    
    # Print temporal trends
    print("\n" + "=" * 80)
    print("TEMPORAL TRENDS")
    print("=" * 80)
    monthly_trends = [t for t in report['temporal_trends'] if t['period_type'] == 'monthly']
    for trend in monthly_trends[:6]:
        spike = " ⚠️ SPIKE" if trend['spike_detected'] else ""
        print(f"  {trend['period']}: {trend['anomaly_count']} anomalies ({trend['anomaly_rate']*100:.1f}%){spike}")
    
    # Save reports
    print("\n" + "=" * 80)
    print("SAVING REPORTS")
    print("=" * 80)
    
    # JSON report
    json_path = os.path.join(output_dir, 'fraud_report.json')
    generator.save_json_report(report, json_path)
    print(f"✓ JSON report saved: {json_path}")
    
    # Text report
    text_path = os.path.join(output_dir, 'fraud_report.txt')
    generator.save_text_report(report, text_path)
    print(f"✓ Text report saved: {text_path}")
    
    # Print human-readable report preview
    print("\n" + "=" * 80)
    print("HUMAN-READABLE REPORT PREVIEW")
    print("=" * 80)
    print(report['human_readable_report'][:2000] + "...\n[Truncated - See full report in file]")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"\nReports saved to: {output_dir}")
    print("\n")


if __name__ == "__main__":
    main()
