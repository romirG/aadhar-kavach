
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
            
            # CRITICAL: Robust None/empty handling for focus_area
            raw_focus = context.get('focus')
            if raw_focus is None or raw_focus == '' or str(raw_focus).lower() == 'none':
                focus_area = 'All India'
            else:
                focus_area = raw_focus
            is_nationwide = focus_area == 'All India'
            
            # Current timestamp for uniqueness in recommendations
            import random
            from datetime import datetime
            current_date = datetime.now().strftime("%B %d, %Y")
            random_seed = random.randint(1000, 9999)
            
            # States to distribute findings across for All India scope
            all_india_states = [
                "Maharashtra", "Karnataka", "Tamil Nadu", "Uttar Pradesh", "Bihar",
                "Rajasthan", "Gujarat", "West Bengal", "Madhya Pradesh", "Telangana",
                "Andhra Pradesh", "Kerala", "Punjab", "Jharkhand", "Odisha"
            ]
            random.shuffle(all_india_states)
            selected_states = all_india_states[:5]  # Pick exactly 5 states for 5 findings
            
            location_instruction = ""
            if is_nationwide:
                location_instruction = f"""
                FOR ALL INDIA SCOPE - MANDATORY STATE DISTRIBUTION:
                You MUST generate exactly 5 findings, one for EACH of these states IN ORDER:
                1. Finding 1: {selected_states[0]} - must mention {selected_states[0]} in location field
                2. Finding 2: {selected_states[1]} - must mention {selected_states[1]} in location field
                3. Finding 3: {selected_states[2]} - must mention {selected_states[2]} in location field
                4. Finding 4: {selected_states[3]} - must mention {selected_states[3]} in location field
                5. Finding 5: {selected_states[4]} - must mention {selected_states[4]} in location field
                
                CRITICAL: DO NOT use "Andhra Pradesh" for all findings. NEVER use "Unknown".
                Each finding MUST have a DIFFERENT state from the list above.
                """
            else:
                location_instruction = f"""
                SINGLE STATE FOCUS: {focus_area}
                - All 5 findings should mention specific districts/regions within {focus_area}
                - Be specific about locations within this state
                """
            
            system_prompt = f"""
            You are the Operations Intelligence Engine for UIDAI Aadhaar Monitoring System.
            Report Date: {current_date} | Session: {random_seed}
            
            Generate a professional analysis report in JSON format.
            
            ⚠️ STRICT RULES - WILL BE REJECTED IF VIOLATED:
            1. NO self-referential language: NEVER write "this report provides", "designed to give", "aimed at helping"
            2. NO phrases like "actionable insights", "comprehensive analysis", "key takeaways"
            3. Write DIRECT findings as if you ARE the monitoring system, not describing it
            4. Summary MUST start with: "Operations review for {focus_area} (review period: {current_date}) indicates..."
            5. Every finding and action MUST be UNIQUE - no repetition
            
            {location_instruction}
            
            REQUIRED JSON OUTPUT:
            {{
                "summary": "Operations review for {focus_area} (review period: {current_date}) indicates [risk level] operational risk. [2-3 sentences about specific concerns - no meta-language]",
                
                "findings": [
                    {{
                        "title": "Specific finding title",
                        "description": "Direct factual statement",
                        "severity": "High/Medium/Low",
                        "location": "State name from the mandatory list",
                        "details": "Comprehensive root cause analysis. Must include: 1) Specific district names/clusters 2) Volume metrics (approximate) 3) Temporal patterns 4) Policy clauses violated. Minimum 2-3 detailed sentences."
                    }}
                ],
                
                "recommended_actions": [
                    {{
                        "action_title": "Clear action heading (e.g., 'Deploy Field Audit Team')",
                        "action_category": "Type of action (Audit/Investigation/Training/Infrastructure/Policy)",
                        "action": "Detailed explanation with specific steps, timeline, and responsible authorities",
                        "priority": "Urgent/High/Normal",
                        "target_region": "Specific state/region this action targets"
                    }}
                ]
            }}
            
            Generate exactly 5 findings (distributed across states) and 4 detailed actions.
            """
            
            user_prompt = f"""
            Analysis Context for Session {random_seed}:
            - Intent: {context.get('intent', 'Operations Monitoring')}
            - Focus Area: {context.get('focus', 'All India')}
            - Current Risk Level: {context.get('risk_level', 'Medium')}
            - Strategies Applied: {context.get('strategies', [])}
            - Total Records Analyzed: {context.get('total_analyzed', 0)}
            - Total Records Flagged: {context.get('total_flagged', 0)}
            
            Sample Flagged Records for Analysis:
            {records_str}
            
            Generate a UNIQUE report with SPECIFIC findings based on this data.
            Remember: Each finding for a DIFFERENT location, each action SPECIFIC to issues found.
            """
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.75,  # Higher for more variety
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
