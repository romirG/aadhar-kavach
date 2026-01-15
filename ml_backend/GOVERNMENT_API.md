# UIDAI Fraud Detection System - Government API Reference

## Overview

This API provides **policy-level controls** for Aadhaar fraud detection.
All technical implementation details are abstracted for government use.

> **Security Note**: This system is designed for UIDAI-compliant operations.
> All interactions are logged for audit purposes.

---

## Primary Interface: Policy Engine

### Base URL
```
http://localhost:8000/api/policy
```

---

## 1. Get Available Policies

**GET** `/api/policy/policies`

View available fraud detection policies.

### Response
```json
[
  {
    "policy_id": "STANDARD",
    "name": "Standard Operations",
    "description": "Balanced approach for routine operations",
    "security_level": "Standard",
    "throughput": "High",
    "recommended_for": "Established centers, routine operations"
  },
  {
    "policy_id": "HIGH_SECURITY",
    "name": "High Security Operations",
    "description": "Maximum scrutiny for sensitive operations",
    "security_level": "Enhanced",
    "recommended_for": "New centers, flagged regions"
  }
]
```

---

## 2. Analyze Records

**POST** `/api/policy/analyze`

Analyze records using selected policy.

### Request
```json
{
  "policy_id": "STANDARD",
  "region": "Maharashtra",
  "record_limit": 500
}
```

### Response
```json
{
  "analysis_id": "abc123...",
  "timestamp": "2026-01-15T02:30:00",
  "policy_applied": {
    "policy_id": "STANDARD",
    "policy_name": "Standard Operations",
    "security_level": "Standard"
  },
  "summary": {
    "total_records": 500,
    "disposition_breakdown": {
      "requires_immediate_action": 5,
      "flagged_for_investigation": 23,
      "pending_review": 45,
      "cleared": 427
    },
    "risk_assessment": "MODERATE - Standard monitoring sufficient",
    "compliance_status": "COMPLIANT",
    "data_integrity": "VERIFIED"
  },
  "flagged_records": [
    {
      "record_id": "REC-000042",
      "disposition": "FLAG",
      "risk_indicator": "High",
      "requires_action": true,
      "action_priority": 2,
      "reason_summary": "Significant deviation in enrollment volume",
      "recommended_action": "Flag for investigation team"
    }
  ],
  "next_steps": [
    "Assign 23 flagged records to investigation team",
    "Schedule review batch for 45 records"
  ],
  "audit_reference": "AUD-2026-01-15-001"
}
```

---

## 3. Get Audit Trail

**GET** `/api/policy/audit-trail`

Retrieve audit trail for compliance.

### Response
```json
{
  "audit_trail": [...],
  "session_id": "sess-123",
  "compliance_level": "standard",
  "classification": "OFFICIAL"
}
```

---

## 4. Compliance Report

**GET** `/api/policy/compliance-report`

Generate report for government submission.

---

## 5. System Status

**GET** `/api/policy/system-status`

Check system operational status.

---

## Record Dispositions

| Disposition | Meaning | Required Action |
|-------------|---------|-----------------|
| **ESCALATE** | Critical - Immediate attention | Escalate to Deputy Director |
| **FLAG** | Suspicious - Investigation needed | Assign to investigation team |
| **REVIEW** | Minor deviation - Manual review | Schedule for review batch |
| **CLEAR** | Normal - Proceed | Standard processing |

---

## Policy Presets

| Policy | Security Level | Use Case |
|--------|---------------|----------|
| `STANDARD` | Standard | Routine operations |
| `HIGH_SECURITY` | Enhanced | Sensitive regions |
| `HIGH_THROUGHPUT` | Standard | Trusted regions |
| `SPECIAL_DRIVE` | Enhanced | Government drives |

---

## Compliance

- **Data Classification**: OFFICIAL
- **Retention Period**: 7 years
- **Audit**: Full trail maintained
- **UIDAI Guidelines**: Compliant

---

## Security Features

✅ No ML model names exposed  
✅ No algorithm parameters visible  
✅ No technical thresholds shown  
✅ Full audit trail for compliance  
✅ Government-friendly terminology  
✅ Role-based access ready  

---

*UIDAI Fraud Detection System v2.0*  
*Classification: OFFICIAL*
