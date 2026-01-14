"""
XAI Examples - Demonstrating the Explainable AI Layer for UIDAI Fraud Detection

This script demonstrates:
1. Feature deviation analysis
2. Fraud explanation generation
3. Sample outputs for different fraud patterns
4. Integration with ensemble anomaly detection
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import json

from explainability.feature_attribution import FeatureAttribution, ReasonGenerator
from explainability.deviation_analyzer import DeviationAnalyzer
from explainability.fraud_explainer import FraudExplainer


def create_sample_data():
    """Create sample UIDAI data for demonstration."""
    np.random.seed(42)
    n_samples = 100
    
    # Create normal data
    data = {
        'state': np.random.choice(['Maharashtra', 'Karnataka', 'Delhi', 'Gujarat'], n_samples),
        'district': np.random.choice(['District_A', 'District_B', 'District_C'], n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'year': np.full(n_samples, 2024),
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'total_enrolments': np.random.poisson(50, n_samples),
        'total_demo_updates': np.random.poisson(20, n_samples),
        'total_bio_updates': np.random.poisson(15, n_samples),
        'state_event_count': np.random.randint(100, 500, n_samples),
        'district_event_count': np.random.randint(50, 200, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add some anomalies
    # Anomaly 1: Geo-inconsistent activity (high activity in unusual location)
    df.loc[5, 'state_event_count'] = 2000  # Very high
    df.loc[5, 'district_event_count'] = 1500
    
    # Anomaly 2: Volume spike (excessive enrolments)
    df.loc[10, 'total_enrolments'] = 500  # 10x normal
    df.loc[10, 'total_demo_updates'] = 200
    
    # Anomaly 3: Weekend fraud
    df.loc[15, 'is_weekend'] = 1
    df.loc[15, 'total_bio_updates'] = 150  # High activity on weekend
    
    # Anomaly 4: Operator fraud pattern
    df.loc[20, 'total_demo_updates'] = 300  # Excessive updates
    df.loc[20, 'total_bio_updates'] = 250
    
    return df


def demonstrate_feature_attribution():
    """Demonstrate feature attribution analysis."""
    print("=" * 80)
    print("DEMONSTRATION 1: Feature Attribution Analysis")
    print("=" * 80)
    
    df = create_sample_data()
    
    # Convert to numpy array
    feature_names = ['total_enrolments', 'total_demo_updates', 'total_bio_updates',
                     'state_event_count', 'district_event_count', 'is_weekend']
    X = df[feature_names].values
    
    # Fit attribution
    attributor = FeatureAttribution()
    attributor.fit(X, feature_names)
    
    # Analyze anomalous record
    anomaly_idx = 10  # Volume spike anomaly
    sample = X[anomaly_idx:anomaly_idx+1]
    
    print(f"\nAnalyzing Record #{anomaly_idx}:")
    print(f"Feature Values: {dict(zip(feature_names, sample[0]))}")
    
    # Get top contributors
    contributors = attributor.get_top_contributors(sample, top_k=5)
    
    print("\nTop Contributing Features:")
    for i, contrib in enumerate(contributors, 1):
        print(f"{i}. {contrib['feature']}")
        print(f"   Value: {contrib['value']:.2f} (Mean: {contrib['mean']:.2f})")
        print(f"   Direction: {contrib['direction']}")
        print(f"   Z-Score: {contrib['z_score']:.2f}")
        print(f"   Attribution: {contrib['attribution']:.3f}")
    
    print("\n")


def demonstrate_reason_generation():
    """Demonstrate human-readable reason generation."""
    print("=" * 80)
    print("DEMONSTRATION 2: Human-Readable Fraud Reasons")
    print("=" * 80)
    
    df = create_sample_data()
    feature_names = ['total_enrolments', 'total_demo_updates', 'total_bio_updates',
                     'state_event_count', 'district_event_count', 'is_weekend']
    X = df[feature_names].values
    
    attributor = FeatureAttribution()
    attributor.fit(X, feature_names)
    
    reason_generator = ReasonGenerator()
    
    # Test different anomaly patterns
    test_cases = [
        (10, 0.85, "Volume Spike"),
        (5, 0.75, "Geo-Inconsistent"),
        (15, 0.70, "Weekend Fraud"),
        (20, 0.90, "Operator Fraud")
    ]
    
    for idx, score, pattern_name in test_cases:
        sample = X[idx:idx+1]
        contributors = attributor.get_top_contributors(sample, top_k=5)
        
        # Generate reasons
        reasons = reason_generator.generate_reasons(contributors, score)
        fraud_pattern = reason_generator.generate_fraud_pattern(contributors)
        
        print(f"\n{'─' * 80}")
        print(f"Record #{idx} - {pattern_name}")
        print(f"Anomaly Score: {score:.2f}")
        print(f"{'─' * 80}")
        print(f"\nFraud Pattern: {fraud_pattern}")
        print(f"\nReasons:")
        for i, reason in enumerate(reasons, 1):
            print(f"  {i}. {reason}")
    
    print("\n")


def demonstrate_deviation_analysis():
    """Demonstrate deviation analysis."""
    print("=" * 80)
    print("DEMONSTRATION 3: Deviation Analysis")
    print("=" * 80)
    
    df = create_sample_data()
    feature_names = ['total_enrolments', 'total_demo_updates', 'total_bio_updates',
                     'state_event_count', 'district_event_count', 'is_weekend']
    
    # Fit analyzer
    analyzer = DeviationAnalyzer()
    analyzer.fit(df, feature_names)
    
    # Analyze anomalous record
    anomaly_idx = 10
    sample_dict = df.iloc[anomaly_idx].to_dict()
    
    print(f"\nAnalyzing Record #{anomaly_idx} for Deviations:")
    
    # Get deviations
    deviations = analyzer.analyze_deviations(sample_dict, top_k=5)
    
    print(f"\nTop Deviations:")
    for i, dev in enumerate(deviations, 1):
        print(f"\n{i}. {dev.feature_name}")
        print(f"   Value: {dev.value:.2f} (Expected: {dev.expected_value:.2f})")
        print(f"   Deviation: {dev.deviation:+.2f}")
        print(f"   Type: {dev.deviation_type}")
        print(f"   Severity: {dev.severity.upper()}")
        print(f"   Z-Score: {dev.z_score:.2f}")
        print(f"   Explanation: {dev.explanation}")
    
    # Temporal deviations
    temporal_flags = analyzer.detect_temporal_deviations(sample_dict)
    if temporal_flags:
        print(f"\nTemporal Flags:")
        for flag in temporal_flags:
            print(f"  - {flag['explanation']} (Severity: {flag['severity']})")
    
    print("\n")


def demonstrate_fraud_explainer():
    """Demonstrate comprehensive fraud explanation."""
    print("=" * 80)
    print("DEMONSTRATION 4: Comprehensive Fraud Explanation")
    print("=" * 80)
    
    df = create_sample_data()
    feature_names = ['total_enrolments', 'total_demo_updates', 'total_bio_updates',
                     'state_event_count', 'district_event_count', 'is_weekend']
    X = df[feature_names].values
    
    # Fit explainer
    explainer = FraudExplainer()
    explainer.fit(X, df, feature_names)
    
    # Explain anomalous record
    anomaly_idx = 20
    sample = X[anomaly_idx]
    sample_dict = df.iloc[anomaly_idx].to_dict()
    anomaly_score = 0.92
    
    # Simulate model scores
    model_scores = {
        'isolation_forest': 0.88,
        'hdbscan': 0.95,
        'autoencoder': 0.93
    }
    
    explanation = explainer.explain(
        sample,
        sample_dict,
        anomaly_score,
        "Highly Suspicious",
        model_scores,
        record_id=f"UIDAI_{anomaly_idx}"
    )
    
    print(f"\n{'═' * 80}")
    print(f"FRAUD EXPLANATION REPORT")
    print(f"{'═' * 80}")
    print(f"\nRecord ID: {explanation.record_id}")
    print(f"Anomaly Score: {explanation.anomaly_score:.3f}")
    print(f"Label: {explanation.anomaly_label}")
    print(f"Confidence: {explanation.confidence:.3f}")
    print(f"Severity: {explanation.severity}")
    
    print(f"\n{'─' * 80}")
    print(f"FRAUD PATTERN")
    print(f"{'─' * 80}")
    print(f"{explanation.fraud_pattern}")
    
    print(f"\n{'─' * 80}")
    print(f"PRIMARY REASONS")
    print(f"{'─' * 80}")
    for i, reason in enumerate(explanation.primary_reasons, 1):
        print(f"{i}. {reason}")
    
    print(f"\n{'─' * 80}")
    print(f"TOP ANOMALOUS FEATURES")
    print(f"{'─' * 80}")
    for i, feature in enumerate(explanation.top_features, 1):
        print(f"{i}. {feature['feature']}: {feature['value']:.2f} "
              f"(Mean: {feature['mean']:.2f}, Z-Score: {feature['z_score']:.2f})")
    
    print(f"\n{'─' * 80}")
    print(f"MODEL CONTRIBUTIONS")
    print(f"{'─' * 80}")
    for model_name, contrib in explanation.model_contributions.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Score: {contrib['score']:.3f}")
        print(f"  Weight: {contrib['weight']:.2f}")
        print(f"  Type: {contrib['type']}")
        print(f"  Finding: {contrib['finding']}")
    
    print(f"\n{'─' * 80}")
    print(f"RECOMMENDATION")
    print(f"{'─' * 80}")
    print(f"{explanation.recommendation}")
    
    print(f"\n{'═' * 80}\n")


def demonstrate_batch_explanation():
    """Demonstrate batch explanation and summary."""
    print("=" * 80)
    print("DEMONSTRATION 5: Batch Explanation & Summary")
    print("=" * 80)
    
    df = create_sample_data()
    feature_names = ['total_enrolments', 'total_demo_updates', 'total_bio_updates',
                     'state_event_count', 'district_event_count', 'is_weekend']
    X = df[feature_names].values
    
    # Fit explainer
    explainer = FraudExplainer()
    explainer.fit(X, df, feature_names)
    
    # Explain multiple anomalies
    anomaly_indices = [5, 10, 15, 20]
    anomaly_scores = np.array([0.75, 0.85, 0.70, 0.92])
    anomaly_labels = ["Suspicious", "Highly Suspicious", "Suspicious", "Highly Suspicious"]
    
    explanations = explainer.explain_batch(
        X[anomaly_indices],
        df.iloc[anomaly_indices],
        anomaly_scores,
        anomaly_labels,
        record_ids=[f"UIDAI_{i}" for i in anomaly_indices]
    )
    
    # Generate summary
    summary = explainer.generate_summary_report(explanations)
    
    print(f"\n{'═' * 80}")
    print(f"BATCH ANALYSIS SUMMARY")
    print(f"{'═' * 80}")
    print(f"\nTotal Records Analyzed: {summary['total_records']}")
    print(f"Average Anomaly Score: {summary['avg_anomaly_score']:.3f}")
    print(f"Average Confidence: {summary['avg_confidence']:.3f}")
    
    print(f"\n{'─' * 80}")
    print(f"SEVERITY DISTRIBUTION")
    print(f"{'─' * 80}")
    for severity, count in summary['severity_distribution'].items():
        print(f"{severity}: {count} records")
    
    print(f"\n{'─' * 80}")
    print(f"TOP FRAUD PATTERNS")
    print(f"{'─' * 80}")
    for i, pattern in enumerate(summary['top_fraud_patterns'], 1):
        print(f"{i}. {pattern['pattern']} ({pattern['count']} occurrences)")
    
    print(f"\n{'─' * 80}")
    print(f"MOST ANOMALOUS FEATURES")
    print(f"{'─' * 80}")
    for i, feature in enumerate(summary['most_anomalous_features'][:5], 1):
        print(f"{i}. {feature['feature']} (mentioned in {feature['mentions']} explanations)")
    
    print(f"\n{'═' * 80}\n")


def save_sample_outputs():
    """Save sample outputs to files."""
    print("=" * 80)
    print("SAVING SAMPLE OUTPUTS")
    print("=" * 80)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'sample_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    df = create_sample_data()
    feature_names = ['total_enrolments', 'total_demo_updates', 'total_bio_updates',
                     'state_event_count', 'district_event_count', 'is_weekend']
    X = df[feature_names].values
    
    explainer = FraudExplainer()
    explainer.fit(X, df, feature_names)
    
    # Generate sample explanations
    samples = [
        (5, 0.75, "Geo-Inconsistent Activity"),
        (10, 0.85, "Volume Spike Fraud"),
        (15, 0.70, "Weekend Activity Fraud"),
        (20, 0.92, "Operator Fraud")
    ]
    
    for idx, score, pattern_name in samples:
        sample = X[idx]
        sample_dict = df.iloc[idx].to_dict()
        
        model_scores = {
            'isolation_forest': score - 0.05,
            'hdbscan': score + 0.05,
            'autoencoder': score
        }
        
        label = "Highly Suspicious" if score >= 0.7 else "Suspicious"
        
        explanation = explainer.explain(
            sample,
            sample_dict,
            score,
            label,
            model_scores,
            record_id=f"UIDAI_{idx}"
        )
        
        # Convert to dict
        output = {
            "record_id": explanation.record_id,
            "pattern_name": pattern_name,
            "anomaly_score": explanation.anomaly_score,
            "anomaly_label": explanation.anomaly_label,
            "confidence": explanation.confidence,
            "severity": explanation.severity,
            "fraud_pattern": explanation.fraud_pattern,
            "primary_reasons": explanation.primary_reasons,
            "top_features": explanation.top_features,
            "deviations": explanation.deviations,
            "model_contributions": explanation.model_contributions,
            "temporal_flags": explanation.temporal_flags,
            "geographic_flags": explanation.geographic_flags,
            "recommendation": explanation.recommendation,
            "timestamp": explanation.timestamp
        }
        
        # Save to file
        filename = f"{pattern_name.lower().replace(' ', '_')}_explanation.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Saved: {filename}")
    
    print(f"\nSample outputs saved to: {output_dir}")
    print("=" * 80 + "\n")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  UIDAI FRAUD DETECTION - EXPLAINABLE AI (XAI) DEMONSTRATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    demonstrate_feature_attribution()
    demonstrate_reason_generation()
    demonstrate_deviation_analysis()
    demonstrate_fraud_explainer()
    demonstrate_batch_explanation()
    save_sample_outputs()
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  DEMONSTRATION COMPLETE".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")


if __name__ == "__main__":
    main()
