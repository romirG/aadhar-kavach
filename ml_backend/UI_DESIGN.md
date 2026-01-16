# UIDAI Fraud Detection - User Interaction Design

## Design Philosophy

> "An auditor should feel empowered, not overwhelmed."

This system lets users express **monitoring intent** in natural language.
No datasets. No parameters. No technical jargon.

---

## Primary Interface: Monitoring Intent

### Main Prompt
```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│    🛡️  Aadhaar Integrity Monitoring                              │
│                                                                  │
│    What would you like to monitor today?                         │
│                                                                  │
│    ┌────────────────────────────────────────────────────────┐   │
│    │ ○  Check today's enrollment operations for issues      │   │
│    │ ○  Review update requests for irregularities           │   │
│    │ ○  Verify biometric submissions for anomalies          │   │
│    │ ○  Run comprehensive integrity check (all operations)  │   │
│    └────────────────────────────────────────────────────────┘   │
│                                                                  │
│                              [Begin Monitoring →]                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Monitoring Intent Options

### Option 1: Enrollment Monitoring
**UI Text:** "Check today's enrollment operations for issues"

**Rationale:** 
- Uses "enrollment" (familiar to auditors) not "enrolment dataset"
- "Check for issues" is action-oriented and non-technical
- "Today's" implies recency without asking for date ranges

---

### Option 2: Update Request Review
**UI Text:** "Review update requests for irregularities"

**Rationale:**
- "Review" implies careful examination (auditor's role)
- "Irregularities" is government-preferred over "anomalies"
- Covers demographic updates without mentioning data types

---

### Option 3: Biometric Verification
**UI Text:** "Verify biometric submissions for anomalies"

**Rationale:**
- "Verify" aligns with audit language
- "Submissions" focuses on the action being audited
- "Anomalies" is acceptable here as biometric fraud is understood

---

### Option 4: Comprehensive Check
**UI Text:** "Run comprehensive integrity check (all operations)"

**Rationale:**
- "Comprehensive" conveys thoroughness
- "Integrity check" is the proper government term
- "(all operations)" clarifies scope without listing datasets

---

## Contextual Inputs (Optional)

### Location Filter
```
┌──────────────────────────────────────────────────────────────┐
│  📍 Focus Area (optional)                                     │
│                                                               │
│  ┌─────────────────────────────────────────┐                 │
│  │ All India                            ▼  │                 │
│  └─────────────────────────────────────────┘                 │
│                                                               │
│  Or specify: [Maharashtra ▼] [Pune ▼]                        │
└──────────────────────────────────────────────────────────────┘
```

**UI Text:** "Focus Area (optional)"

**Rationale:**
- "Focus Area" is intuitive, not "Region Filter"
- "Optional" removes pressure
- "All India" default shows comprehensive coverage
- Dropdown prevents typos

---

### Time Period
```
┌──────────────────────────────────────────────────────────────┐
│  📅 Analysis Period (optional)                                │
│                                                               │
│  ○ Today's operations                                         │
│  ○ Last 7 days                                                │
│  ○ This month                                                 │
│  ○ Custom range                                               │
└──────────────────────────────────────────────────────────────┘
```

**UI Text:** "Analysis Period (optional)"

**Rationale:**
- "Analysis Period" is formal and clear
- Pre-defined options prevent confusion
- "Today's operations" is the default (most common use)
- "Custom range" available but not prominent

---

### Vigilance Level
```
┌──────────────────────────────────────────────────────────────┐
│  🔒 Vigilance Level                                           │
│                                                               │
│  ─────────●───────────────────────────────                   │
│  Routine    Standard    Enhanced    Maximum                   │
│                  ↑                                            │
│              (Default)                                        │
│                                                               │
│  Standard: Balanced monitoring for normal operations          │
└──────────────────────────────────────────────────────────────┘
```

**UI Text Options:**
| Level | UI Label | Description |
|-------|----------|-------------|
| Low | Routine | "Quick check for obvious concerns" |
| Medium | Standard | "Balanced monitoring for normal operations" |
| High | Enhanced | "Thorough review for sensitive periods" |
| Critical | Maximum | "Complete scrutiny for special circumstances" |

**Rationale:**
- "Vigilance Level" is government-appropriate
- Slider is intuitive with clear labels
- Description updates based on selection
- No numeric thresholds or percentages

---

## Advanced Options (Hidden by Default)

### Collapsed State
```
┌──────────────────────────────────────────────────────────────┐
│  ⚙️ Advanced Options                                    [+]   │
└──────────────────────────────────────────────────────────────┘
```

### Expanded State (On Click)
```
┌──────────────────────────────────────────────────────────────┐
│  ⚙️ Advanced Options                                    [-]   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Processing Priority                                          │
│  ○ Normal queue                                               │
│  ○ Priority processing                                        │
│                                                               │
│  Report Format                                                │
│  ○ Summary view                                               │
│  ○ Detailed report                                            │
│  ○ Compliance format                                          │
│                                                               │
│  Notification                                                 │
│  ☐ Email me when complete                                    │
│  ☐ Alert if critical issues found                            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**Rationale for Hiding:**
- 95% of users don't need these
- Reduces cognitive load
- Clean initial interface
- Power users can access when needed

---

## Results Display

### Summary Card
```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  ✅ Monitoring Complete                                       │
│                                                               │
│  ──────────────────────────────────────────────────────────  │
│                                                               │
│  📊 1,247 records reviewed                                    │
│                                                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│  │     12      │ │     34      │ │   1,201     │             │
│  │  Flagged    │ │ For Review  │ │   Cleared   │             │
│  │  ⚠️ Action  │ │ 📋 Pending  │ │ ✅ Normal   │             │
│  └─────────────┘ └─────────────┘ └─────────────┘             │
│                                                               │
│  Overall Status: MODERATE - Standard follow-up recommended   │
│                                                               │
│  [View Flagged Items]  [Download Report]  [Schedule Review]  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**Rationale:**
- "Monitoring Complete" not "Analysis Finished"
- Visual cards for quick scanning
- Action-oriented buttons
- Status in plain language

---

## Flagged Item Display

```
┌──────────────────────────────────────────────────────────────┐
│  ⚠️ Flagged Item #1                           Priority: High  │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  📍 Location: Pune, Maharashtra                               │
│  📅 Date: 14 January 2026                                     │
│                                                               │
│  Issue Identified:                                            │
│  Unusual enrollment volume detected at this center.           │
│  Activity levels significantly exceed normal patterns.        │
│                                                               │
│  Recommended Action:                                          │
│  Verify operator credentials and cross-reference with         │
│  other flagged records from this region.                      │
│                                                               │
│  [Assign to Team]  [Mark as Reviewed]  [Escalate]            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**Rationale:**
- "Issue Identified" not "Anomaly Detected"
- "Unusual volume" not "statistical outlier"
- "Recommended Action" guides next steps
- Clear action buttons

---

## Words to Avoid / Use

| ❌ Don't Use | ✅ Use Instead |
|-------------|----------------|
| Anomaly score | Risk indicator |
| Dataset | Operations / Records |
| Threshold | Vigilance level |
| Model | System |
| Algorithm | Process |
| Prediction | Assessment |
| Training | Calibration |
| False positive | Cleared on review |
| Contamination | Flagging sensitivity |
| Hyperparameter | Setting |

---

## Complete Flow Summary

```
┌─────────────────┐
│ Select Intent   │  "What would you like to monitor?"
└────────┬────────┘
         ▼
┌─────────────────┐
│ Optional Inputs │  Focus Area, Time Period, Vigilance
└────────┬────────┘
         ▼
┌─────────────────┐
│ Begin Monitoring│  One-click start
└────────┬────────┘
         ▼
┌─────────────────┐
│ View Results    │  Flagged, For Review, Cleared
└────────┬────────┘
         ▼
┌─────────────────┐
│ Take Action     │  Assign, Review, Escalate, Report
└─────────────────┘
```

---

## Implementation Notes

### API Mapping (Internal Only)

| User Intent | Internal Dataset | Policy |
|-------------|-----------------|--------|
| "Check enrollments" | enrolment | STANDARD |
| "Review updates" | demographic | STANDARD |
| "Verify biometrics" | biometric | STANDARD |
| "Comprehensive check" | all | HIGH_SECURITY |

| Vigilance Level | Internal Policy |
|-----------------|-----------------|
| Routine | HIGH_THROUGHPUT |
| Standard | STANDARD |
| Enhanced | HIGH_SECURITY |
| Maximum | HIGH_SECURITY + secondary |

*Users never see this mapping.*

---

*Design Document v1.0*  
*UIDAI Fraud Detection - Government UI*
