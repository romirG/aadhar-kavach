"""
Groq AI Service for Deep Analysis and Policy Recommendations

Uses Groq's Llama 3.3 70B model to:
- Generate detailed analysis insights from risk data
- Provide policy change recommendations
- Create actionable operational guidance
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger(__name__)

# Groq API Configuration
GROQ_API_KEYS = [
    "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",
    "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"
]
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

current_key_index = 0


def get_api_key() -> str:
    """Get current API key with rotation for rate limits"""
    global current_key_index
    key = GROQ_API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
    return key


def call_groq_api(prompt: str, max_tokens: int = 2000) -> Optional[str]:
    """
    Call Groq API with automatic key rotation.
    """
    for attempt in range(len(GROQ_API_KEYS)):
        api_key = get_api_key()
        
        try:
            response = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are a senior policy analyst at UIDAI (Unique Identification Authority of India). 
You analyze biometric risk data and provide actionable policy recommendations to improve Aadhaar authentication success rates.
Your recommendations should be specific, data-driven, and implementable.
Format your responses in clear sections with bullet points."""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            elif response.status_code == 429:
                logger.warning(f"Rate limited on key, trying next...")
                continue
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"Groq API request failed: {e}")
            continue
    
    return None


def generate_deep_analysis(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate AI-powered deep analysis from risk assessment data.
    
    Returns insights about patterns, root causes, and trends.
    """
    # Extract key metrics for the prompt
    summary = analysis_data.get('summary', {})
    high_risk_regions = analysis_data.get('high_risk_regions', [])[:10]
    age_bucket_analysis = analysis_data.get('age_bucket_analysis', {})
    centre_performance = analysis_data.get('centre_performance', [])[:5]
    survival_data = analysis_data.get('survival_data', {})
    
    prompt = f"""Analyze the following UIDAI biometric re-enrollment risk assessment data and provide deep insights:

## Summary Statistics
- Total Records Analyzed: {summary.get('total_records_analyzed', 'N/A')}
- Regions Analyzed: {summary.get('regions_analyzed', 'N/A')}
- High Risk Regions: {summary.get('high_risk_count', 'N/A')}
- Average Risk Score: {summary.get('average_risk_score', 'N/A'):.1%}
- Model Accuracy: {summary.get('model_accuracy', 'N/A'):.1%}

## Top High-Risk Regions
{json.dumps(high_risk_regions, indent=2)}

## Age Bucket Risk Analysis
{json.dumps(age_bucket_analysis, indent=2)}

## Lowest Performing Centres
{json.dumps(centre_performance, indent=2)}

## Survival Analysis Statistics
{json.dumps(survival_data.get('statistics', {}), indent=2)}

Please provide:

1. **KEY FINDINGS** - Top 3-5 critical insights from this data
2. **ROOT CAUSE ANALYSIS** - Why are certain regions/demographics at higher risk?
3. **TREND PATTERNS** - What patterns do you observe across age groups and regions?
4. **RISK FACTORS RANKING** - Rank the most significant risk factors
5. **OPERATIONAL GAPS** - Identify gaps in current biometric capture operations

Be specific and reference the actual data provided."""

    result = call_groq_api(prompt)
    
    if result:
        return {
            "success": True,
            "analysis": result,
            "model": GROQ_MODEL,
            "type": "deep_analysis"
        }
    else:
        return {
            "success": False,
            "error": "Failed to generate AI analysis",
            "analysis": None
        }


def generate_policy_recommendations(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate AI-powered policy change recommendations.
    
    Returns actionable policy suggestions for UIDAI officials.
    """
    summary = analysis_data.get('summary', {})
    high_risk_regions = analysis_data.get('high_risk_regions', [])[:10]
    age_bucket_analysis = analysis_data.get('age_bucket_analysis', {})
    recommendations = analysis_data.get('recommendations', [])
    
    prompt = f"""Based on the UIDAI biometric risk assessment data, generate specific policy change recommendations:

## Current Risk Summary
- High Risk Regions: {summary.get('high_risk_count', 'N/A')} out of {summary.get('regions_analyzed', 'N/A')}
- Average Risk Score: {summary.get('average_risk_score', 'N/A'):.1%}
- Critical regions (>90% risk): {len([r for r in high_risk_regions if r.get('risk_score', 0) > 0.9])}

## High-Risk States Requiring Attention
{json.dumps([r.get('state', 'Unknown') for r in high_risk_regions[:5]], indent=2)}

## Age Demographics at Risk
{json.dumps(age_bucket_analysis, indent=2)}

## Current System Recommendations
{json.dumps(recommendations[:3], indent=2)}

Please provide ACTIONABLE POLICY RECOMMENDATIONS in these categories:

1. **IMMEDIATE ACTIONS (0-30 days)**
   - Emergency interventions for critical regions
   - Quick wins to reduce authentication failures

2. **SHORT-TERM POLICIES (1-6 months)**
   - Infrastructure improvements
   - Training and capacity building
   - Process changes

3. **LONG-TERM STRATEGIC CHANGES (6-24 months)**
   - Technology upgrades
   - Policy framework changes
   - Preventive measures

4. **BUDGET ALLOCATION PRIORITIES**
   - Where to invest for maximum impact
   - Cost-benefit considerations

5. **MONITORING & EVALUATION METRICS**
   - KPIs to track policy effectiveness
   - Success criteria

Make recommendations specific to Indian government context and UIDAI operations.
Reference specific states/regions from the data where applicable."""

    result = call_groq_api(prompt, max_tokens=2500)
    
    if result:
        return {
            "success": True,
            "recommendations": result,
            "model": GROQ_MODEL,
            "type": "policy_recommendations"
        }
    else:
        return {
            "success": False,
            "error": "Failed to generate policy recommendations",
            "recommendations": None
        }


def generate_region_specific_analysis(region_name: str, region_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate AI analysis for a specific region/state.
    """
    prompt = f"""Analyze the biometric risk profile for {region_name} and provide targeted recommendations:

## Region Data
{json.dumps(region_data, indent=2)}

Please provide:
1. **RISK ASSESSMENT** - Overall risk level and contributing factors
2. **DEMOGRAPHIC CONCERNS** - Which population segments need priority attention
3. **OPERATIONAL ISSUES** - Specific problems in this region's biometric capture
4. **RECOMMENDED INTERVENTIONS** - 3-5 actionable steps for this region
5. **RESOURCE REQUIREMENTS** - What resources are needed for improvement

Be specific to the local context of {region_name}."""

    result = call_groq_api(prompt, max_tokens=1500)
    
    if result:
        return {
            "success": True,
            "region": region_name,
            "analysis": result,
            "model": GROQ_MODEL
        }
    else:
        return {
            "success": False,
            "error": f"Failed to generate analysis for {region_name}",
            "analysis": None
        }


# Singleton instance
groq_service = {
    "deep_analysis": generate_deep_analysis,
    "policy_recommendations": generate_policy_recommendations,
    "region_analysis": generate_region_specific_analysis
}
