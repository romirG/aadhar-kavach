
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
        self.model = "llama-3.3-70b-versatile"  # Updated: llama3-70b-8192 was decommissioned 
    
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
            focus_area = context.get('focus') or 'All India'
            is_nationwide = focus_area == 'All India'
            
            location_instruction = ""
            if is_nationwide:
                location_instruction = """
                FOR ALL INDIA SCOPE:
                - Distribute findings across MULTIPLE states (Maharashtra, Karnataka, Tamil Nadu, UP, Bihar, etc.)
                - DO NOT focus only on one state
                - Each finding should mention a different state/region
                - Cover North, South, East, West regions for comprehensive coverage
                """
            
            system_prompt = f"""
            You are an Operations Intelligence Engine for UIDAI Aadhaar Monitoring.
            
            Generate a professional analysis report in JSON format.
            
            ⚠️ CRITICAL RULES:
            - NO meta-language ("this report provides..." or "designed to...")
            - Write direct findings as the auditing system
            - NO ML terms or algorithm names
            - Focus Area: {focus_area}
            {location_instruction}
            
            REQUIRED OUTPUT (JSON):
            {{
                "summary": "Direct executive assessment. Start with: 'Monitoring of {focus_area} operations for the review period indicates [risk level] risk.' Describe key concerns factually.",
                
                "findings": [
                    {{
                        "title": "Concise finding title",
                        "description": "Direct factual description",
                        "severity": "High/Medium/Low",
                        "details": "Specific regions affected within {focus_area}. Impact and compliance risk."
                    }}
                ],
                
                "recommended_actions": [
                    {{
                        "action_title": "Short action heading (e.g., 'Initiate Targeted Audit')",
                        "action": "Detailed policy/procedural directive with specific guidance",
                        "priority": "Urgent/High/Normal"
                    }}
                ]
            }}
            
            Generate 5 findings (High/Medium/Low severity) and 4 actions with titles.
            """
            
            user_prompt = f"""
            Analysis Context:
            - Intent: {context.get('intent', 'Operations Monitoring')}
            - Focus Area: {context.get('focus', 'All India')}
            - Risk Level: {context.get('risk_level', 'Unknown')}
            - Strategies Applied: {context.get('strategies', [])}
            - Records Analyzed: {context.get('total_analyzed', 0)}
            - Records Flagged: {context.get('total_flagged', 0)}
            
            Sample Flagged Records:
            {records_str}
            
            Generate a detailed, actionable analysis report with specific findings and policy recommendations.
            """
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.5,
                "response_format": {"type": "json_object"}
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            logger.info("Calling Groq API for analysis...")
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Groq Analysis failed: {e}")
            return None

groq_service = GroqService()
