
import logging
import json
import requests
from typing import Dict, Any, List, Optional
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
            logger.warning("Groq API execution skipped: No API Key")
            return None

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
            
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Groq Analysis failed: {e}")
            return None

groq_service = GroqService()
