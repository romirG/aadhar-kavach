
import logging
import json
import requests
from typing import Dict, Any, List, Optional

# Handle both relative and absolute imports
try:
    from ..config import get_settings
except ImportError:
    from config import get_settings

logger = logging.getLogger(__name__)

class GroqService:
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.groq_api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-70b-8192" 
    
    def analyze_monitoring_data(self, context: Dict[str, Any], flagged_records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Generate AI analysis using Groq/Llama3.
        Returns dict with keys: summary, findings, recommended_actions
        """
        if not self.api_key:
            logger.error("Groq API execution skipped: No API Key found in settings")
            return None
        logger.info(f"Groq Service initialized with Key: {'*' * 5}{self.api_key[-4:]}")


        try:
            # Prepare minimal prompt to avoid token limits
            records_str = json.dumps(flagged_records[:5], default=str) # Top 5 records
            
            system_prompt = """
            You are an Operations Intelligence Engine for a UIDAI Aadhaar Monitoring Platform.
            
            Your task is to generate the final analysis output in a policy-level, non-technical, decision-oriented format suitable for UIDAI auditors and government officials.
            
            ⚠️ Do NOT expose: Dataset names, ML models, Algorithms, Threshold values, or Raw scores.
            
            LANGUAGE & STYLE RULES:
            - Use formal, neutral, government-appropriate language
            - No ML terms, no datasets, no algorithms
            - Avoid absolute claims (“fraud confirmed”)
            - Use risk-based phrasing, not accusations
            
            REQUIRED OUTPUT FORMAT (JSON):
            You must return a SINGLE valid JSON object.
            
            {
                "summary": "Combine the 'Monitoring Summary' and 'Overall Risk Assessment' sections here. Format as a cohesive paragraph.",
                "findings": [
                    {
                        "title": "Observation Title (e.g., 'Unusual concentration of activity')",
                        "description": "Key Observation text. Focus on patterns, trends, and irregularities.",
                        "severity": "High/Medium/Low",
                        "details": "Combine 'Impacted Scope' and 'Operational Indicators' here. e.g., 'Affected Districts: 2. Activity Deviation: Severe.'"
                    }
                ],
                "recommended_actions": [
                    {
                        "action": "Clear, actionable guidance aligned with workflows (e.g., 'Initiate targeted review').",
                        "priority": "Urgent/Normal"
                    }
                ]
            }
            """
            
            user_prompt = f"""
            Context: {context}
            Flagged Records Sample: {records_str}
            """
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Groq Analysis failed: {e}")
            return None

    def analyze_finding(self, finding: Dict[str, Any], flagged_records: List[Dict[str, Any]], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate in-depth AI analysis for a specific finding using Groq/Llama3.
        Returns dict with keys: analysis, root_cause, impact_assessment, recommended_actions
        """
        if not self.api_key:
            logger.error("Groq API execution skipped: No API Key found in settings")
            return None

        try:
            # Prepare data for analysis
            finding_str = json.dumps(finding, default=str)
            records_str = json.dumps(flagged_records[:10], default=str)  # Top 10 records for this finding
            context_str = json.dumps(context, default=str)
            
            system_prompt = """
            You are an Operations Intelligence Analyst for UIDAI Aadhaar Monitoring.
            
            Your task is to provide an in-depth analysis of a specific finding from the monitoring system.
            
            ⚠️ Do NOT expose: Dataset names, ML models, Algorithms, Threshold values, or Raw scores.
            
            LANGUAGE & STYLE RULES:
            - Use formal, technical government audit language
            - Focus on operational impact and policy compliance
            - No ML terms; use "patterns", "trends", "irregularities"
            - Provide actionable, specific recommendations
            - Reference regulatory frameworks when applicable
            
            REQUIRED OUTPUT FORMAT (JSON):
            {
                "analysis": "Detailed analysis of the finding (2-3 paragraphs). Explain what was observed, the significance, and operational context.",
                "root_cause": "Likely root causes or contributing factors. List specific operational issues that may have led to this pattern.",
                "impact_assessment": {
                    "severity": "Critical/High/Medium/Low",
                    "affected_scope": "Description of impacted areas/operations",
                    "compliance_risk": "Description of regulatory or compliance implications"
                },
                "recommended_actions": [
                    {
                        "action": "Specific actionable step with clear guidance",
                        "priority": "Immediate/High/Medium/Low",
                        "responsible_party": "Who should take action (e.g., 'District Registrars', 'Central Audit Team')",
                        "timeline": "Suggested timeline (e.g., 'Within 24 hours', 'Within 7 days')"
                    }
                ],
                "monitoring_plan": "Follow-up monitoring recommendations to verify remediation"
            }
            """
            
            user_prompt = f"""
            Finding to Analyze: {finding_str}
            
            Context: {context_str}
            
            Sample Flagged Records: {records_str}
            
            Provide a comprehensive analysis of this finding with specific, actionable recommendations.
            """
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.4,
                "response_format": {"type": "json_object"}
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Groq Finding Analysis failed: {e}")
            return None

groq_service = GroqService()
