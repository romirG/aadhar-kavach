# Operations Monitoring - API Documentation

## Base URLs

- **Frontend**: `http://localhost:8080`
- **Express Server**: `http://localhost:3001`
- **ML Backend**: `http://localhost:8000`

## Request Flow

```
Frontend (React) → Express Server (Port 3001) → ML Backend (Port 8000)
```

## Endpoints

### 1. Get Available Monitoring Intents

**Endpoint**: `GET /api/monitor/intents`

**Description**: Returns list of monitoring options and vigilance levels for UI display.

**Response**:
```json
{
  "intents": [
    {
      "id": "check_enrollments",
      "display_name": "Check today's enrollment operations for issues",
      "description": "Monitor new Aadhaar enrollment activities for irregularities"
    },
    {
      "id": "review_updates",
      "display_name": "Review update requests for irregularities",
      "description": "Examine demographic and address update patterns"
    },
    {
      "id": "verify_biometrics",
      "display_name": "Verify biometric submissions for anomalies",
      "description": "Monitor biometric update and verification patterns"
    },
    {
      "id": "comprehensive_check",
      "display_name": "Run comprehensive integrity check",
      "description": "Complete system-wide integrity assessment"
    }
  ],
  "vigilance_levels": [
    {
      "id": "routine",
      "name": "Routine",
      "description": "Quick check for obvious concerns"
    },
    {
      "id": "standard",
      "name": "Standard",
      "description": "Balanced monitoring"
    },
    {
      "id": "enhanced",
      "name": "Enhanced",
      "description": "Thorough review"
    },
    {
      "id": "maximum",
      "name": "Maximum",
      "description": "Complete scrutiny"
    }
  ]
}
```

---

### 2. Submit Monitoring Request

**Endpoint**: `POST /api/monitor`

**Description**: Creates a new monitoring job and starts background processing.

**Request Body**:
```json
{
  "intent": "check_enrollments",
  "focus_area": "Maharashtra",
  "time_period": "today",
  "vigilance": "standard",
  "record_limit": 1000
}
```

**Parameters**:
| Field | Type | Required | Values | Description |
|-------|------|----------|--------|-------------|
| intent | string | Yes | check_enrollments, review_updates, verify_biometrics, comprehensive_check | What to monitor |
| focus_area | string | No | State name or "All India" | Geographic focus |
| time_period | string | Yes | today, last_7_days, this_month | Analysis period |
| vigilance | string | Yes | routine, standard, enhanced, maximum | Monitoring intensity |
| record_limit | number | No | 100-10000 | Max records to analyze |

**Response**:
```json
{
  "job_id": "abc123xyz456",
  "status": "pending",
  "message": "Monitoring request received. Analyzing check enrollments.",
  "estimated_time": "1-2 minutes"
}
```

---

### 3. Check Job Status

**Endpoint**: `GET /api/monitor/status/{job_id}`

**Description**: Returns current status and progress of a monitoring job.

**URL Parameters**:
- `job_id` (string, required): Job identifier returned from submit request

**Response**:
```json
{
  "job_id": "abc123xyz456",
  "status": "processing",
  "progress": 65,
  "message": "Looking for patterns and concerns...",
  "started_at": "2026-01-16T10:30:00",
  "completed_at": null
}
```

**Status Values**:
- `pending`: Job queued, not started
- `processing`: Analysis in progress
- `completed`: Job finished successfully
- `failed`: Job encountered an error

---

### 4. Get Monitoring Results

**Endpoint**: `GET /api/monitor/results/{job_id}`

**Description**: Returns complete analysis results for a completed job.

**URL Parameters**:
- `job_id` (string, required): Job identifier

**Response**:
```json
{
  "job_id": "abc123xyz456",
  "status": "completed",
  "summary": "Comprehensive analysis of enrollment operations in Maharashtra reveals elevated activity patterns requiring attention. Review flagged records for potential operational anomalies.",
  "risk": {
    "risk_index": 67,
    "risk_level": "Medium",
    "confidence": "High"
  },
  "findings": [
    {
      "title": "Elevated enrollment velocity detected",
      "description": "Unusual spike in new enrollments exceeding historical baseline by 34%",
      "severity": "High",
      "location": "Maharashtra",
      "details": "Concentration in 3 districts: Mumbai, Pune, Nagpur"
    }
  ],
  "recommended_actions": [
    {
      "action": "Initiate field verification of enrollment centers in Maharashtra",
      "priority": "High"
    },
    {
      "action": "Review operator credentials and authentication logs for Maharashtra",
      "priority": "High"
    }
  ],
  "records_analyzed": 1000,
  "flagged_for_review": 87,
  "cleared": 913,
  "analysis_scope": "Maharashtra",
  "time_period": "Today",
  "report_id": "RPT-ABC123XYZ456",
  "completed_at": "2026-01-16T10:32:15",
  "flagged_records": [
    {
      "state": "Maharashtra",
      "district": "Mumbai",
      "enrollment_count": 1234,
      "flagged_reason": "High velocity of updates detected",
      "risk_score": "87/100"
    }
  ]
}
```

---

### 5. Analyze Individual Finding (AI Deep-Dive)

**Endpoint**: `POST /api/monitor/analyze-finding`

**Description**: Performs in-depth AI analysis on a specific finding using Groq API.

**Request Body**:
```json
{
  "job_id": "abc123xyz456",
  "finding_index": 0
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| job_id | string | Yes | ID of completed monitoring job |
| finding_index | number | Yes | Index of finding to analyze (0-based) |

**Response**:
```json
{
  "success": true,
  "analysis": "The observed elevation in enrollment velocity represents a statistically significant deviation from historical patterns for Maharashtra. This pattern is concentrated in urban districts with high population density, suggesting either:\n\n1. Legitimate surge due to awareness campaigns or policy changes\n2. Operational irregularities requiring verification\n\nThe 34% increase exceeds normal seasonal variation and warrants immediate attention to rule out systematic anomalies.",
  "root_cause": "Analysis indicates three primary contributing factors:\n\n1. Recent government initiative promoting Aadhaar enrollment\n2. Deployment of additional enrollment centers in urban areas\n3. Possible operational shortcuts or reduced verification protocols\n\nThe concentration in specific districts suggests localized factors rather than state-wide policy changes.",
  "impact_assessment": {
    "severity": "High",
    "affected_scope": "3 districts in Maharashtra (Mumbai, Pune, Nagpur) with approximately 87 flagged records out of 1000 analyzed",
    "compliance_risk": "Potential violation of enrollment quality protocols if verification steps are bypassed. May impact data integrity and authentication reliability."
  },
  "recommended_actions": [
    {
      "action": "Deploy audit teams to flagged enrollment centers for on-site verification of processes",
      "priority": "Immediate",
      "responsible_party": "District Registrars - Mumbai, Pune, Nagpur",
      "timeline": "Within 24 hours"
    },
    {
      "action": "Review enrollment operator performance metrics and authentication logs for anomalies",
      "priority": "High",
      "responsible_party": "Central Monitoring Team",
      "timeline": "Within 48 hours"
    },
    {
      "action": "Implement enhanced verification protocols for high-velocity centers",
      "priority": "High",
      "responsible_party": "State UIDAI Coordinator",
      "timeline": "Within 7 days"
    }
  ],
  "monitoring_plan": "Implement daily monitoring of enrollment velocity for Maharashtra for next 30 days. Set alerts for any center exceeding 20% historical average. Conduct follow-up audit in 2 weeks to verify corrective actions."
}
```

**Error Response**:
```json
{
  "success": false,
  "error": "AI analysis service unavailable. Please ensure Groq API key is configured."
}
```

---

## Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 202 | Accepted | Job pending or processing |
| 400 | Bad Request | Invalid input parameters |
| 404 | Not Found | Job ID not found |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | ML backend not reachable |

---

## Rate Limiting

- No rate limiting currently implemented
- Groq API has its own rate limits (check console.groq.com)

---

## Authentication

- No authentication currently required
- Add API keys for production deployment

---

## Error Handling

### Common Errors

**ML Backend Unavailable**:
```json
{
  "error": "ML Backend Unavailable",
  "message": "The ML backend server is not running on port 8000."
}
```

**Job Not Found**:
```json
{
  "detail": "Job not found"
}
```

**Invalid Finding Index**:
```json
{
  "detail": "Invalid finding index. Must be between 0 and 2"
}
```

---

## WebSocket Support

Currently not supported. Use polling for status updates.

**Recommended Polling Interval**: 1 second during processing

---

## Example Usage

### JavaScript/Fetch
```javascript
// Submit request
const response = await fetch('http://localhost:3001/api/monitor', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    intent: 'check_enrollments',
    focus_area: 'Maharashtra',
    time_period: 'today',
    vigilance: 'standard'
  })
});
const { job_id } = await response.json();

// Poll status
const checkStatus = async () => {
  const statusRes = await fetch(
    `http://localhost:3001/api/monitor/status/${job_id}`
  );
  const status = await statusRes.json();
  
  if (status.status === 'completed') {
    const resultsRes = await fetch(
      `http://localhost:3001/api/monitor/results/${job_id}`
    );
    const results = await resultsRes.json();
    console.log(results);
  }
};
```

### cURL
```bash
# Submit request
curl -X POST http://localhost:3001/api/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "check_enrollments",
    "focus_area": "Maharashtra",
    "time_period": "today",
    "vigilance": "standard"
  }'

# Check status
curl http://localhost:3001/api/monitor/status/abc123

# Get results
curl http://localhost:3001/api/monitor/results/abc123

# Analyze finding
curl -X POST http://localhost:3001/api/monitor/analyze-finding \
  -H "Content-Type: application/json" \
  -d '{"job_id": "abc123", "finding_index": 0}'
```

---

**Last Updated**: January 16, 2026
