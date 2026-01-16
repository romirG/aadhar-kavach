# Frontend Files Location Map

## Actual File Paths

All frontend files are located in the `src/` directory at the project root.

### Pages
- **File**: `../../src/pages/Monitoring.tsx`
- **Purpose**: Main operations monitoring interface
- **Lines**: ~440 lines
- **Key Features**:
  - Monitoring request form
  - Real-time job status tracking
  - Results display with findings
  - AI analysis dialog integration
- **State Management**:
  - Selected intent, state, vigilance, time period
  - Job ID and status tracking
  - Results and error handling
  - AI dialog state

### Components
- **File**: `../../src/components/AIAnalysisDialog.tsx`
- **Purpose**: AI analysis modal dialog
- **Lines**: ~250 lines
- **Sections Displayed**:
  - Detailed Analysis
  - Root Cause Analysis
  - Impact Assessment (severity, scope, compliance)
  - Recommended Actions (with priority, responsible party, timeline)
  - Follow-up Monitoring Plan
- **Features**:
  - Auto-loads analysis on open
  - Loading states
  - Error handling
  - Color-coded badges
  - Responsive design

### Services
- **File**: `../../src/services/api.ts`
- **Purpose**: API client and type definitions
- **Lines**: ~368 lines
- **Functions**:
  ```typescript
  getMonitoringIntents(): Promise<{intents, vigilance_levels}>
  submitMonitoringRequest(request): Promise<MonitoringJobResponse>
  getMonitoringStatus(jobId): Promise<StatusResponse>
  getMonitoringResults(jobId): Promise<MonitoringResults>
  analyzeFinding(jobId, findingIndex): Promise<FindingAnalysisResponse>
  ```
- **Type Definitions**:
  - `MonitoringIntent`
  - `VigilanceLevel`
  - `MonitoringRequest`
  - `MonitoringResults`
  - `Finding`
  - `ActionItem`
  - `ImpactAssessment`
  - `DetailedActionItem`
  - `FindingAnalysisResponse`

### UI Components (from Shadcn/ui)
Located in `../../src/components/ui/`:
- `dialog.tsx` - Modal dialog
- `button.tsx` - Button variants
- `card.tsx` - Card containers
- `badge.tsx` - Status badges
- `select.tsx` - Dropdown selects
- `alert.tsx` - Alert messages

### Utilities
- **File**: `../../src/lib/utils.ts`
- **Purpose**: Helper functions and utilities

## Quick Navigation Commands

```bash
# View monitoring page
cat ../../src/pages/Monitoring.tsx

# View AI dialog
cat ../../src/components/AIAnalysisDialog.tsx

# View API service
cat ../../src/services/api.ts

# Start frontend dev server
cd ../.. && npm run dev
```

## Component Hierarchy

```
Monitoring.tsx (Page)
├── DashboardLayout
├── Card (Control Panel)
│   ├── Select (Intent)
│   ├── Select (Focus Area)
│   ├── Select (Time Period)
│   ├── Select (Vigilance Level)
│   └── Button (Start Monitoring)
├── Card (Progress Display)
├── Card (Stats - 4 cards)
├── Card (Executive Summary)
├── Card (Key Findings)
│   └── For each finding:
│       ├── Finding details
│       ├── Button (AI Analysis) ← Opens dialog
│       └── Button (Details)
├── Card (Recommended Actions)
└── AIAnalysisDialog ← Modal
    ├── Detailed Analysis Card
    ├── Root Cause Card
    ├── Impact Assessment Card
    ├── Recommended Actions Card
    └── Monitoring Plan Card
```

## Props and State Flow

### Monitoring.tsx State
```typescript
// Form state
selectedIntent: string
selectedState: string
selectedVigilance: string
selectedPeriod: string

// Job state
loading: boolean
jobId: string | null
status: StatusResponse | null
results: MonitoringResults | null
error: string | null

// Dialog state
aiDialogOpen: boolean
selectedFindingIndex: number
selectedFindingTitle: string
```

### AIAnalysisDialog Props
```typescript
interface AIAnalysisDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  jobId: string;
  findingIndex: number;
  findingTitle: string;
}
```

## API Integration Points

### 1. Load Intents (on mount)
```typescript
const data = await getMonitoringIntents();
setIntents(data.intents);
setVigilanceLevels(data.vigilance_levels);
```

### 2. Submit Request
```typescript
const response = await submitMonitoringRequest({
  intent, focus_area, time_period, vigilance, record_limit
});
setJobId(response.job_id);
```

### 3. Poll Status (interval)
```typescript
const newStatus = await getMonitoringStatus(jobId);
if (newStatus.status === 'completed') {
  const res = await getMonitoringResults(jobId);
  setResults(res);
}
```

### 4. Analyze Finding (on button click)
```typescript
const result = await analyzeFinding(jobId, findingIndex);
// Display in dialog
```

## Styling

### Tailwind Classes Used
- Layout: `space-y-6`, `flex`, `grid`, `gap-4`
- Cards: `shadow-card`, `border`, `rounded-lg`
- Text: `text-2xl`, `font-bold`, `text-muted-foreground`
- Colors: `bg-success`, `bg-warning`, `bg-destructive`
- Interactive: `hover:`, `transition-all`

### Theme Variables (from globals.css)
- `--primary`, `--secondary`
- `--success`, `--warning`, `--destructive`
- `--muted`, `--muted-foreground`
- `--border`, `--background`

## File Change History

### Last Modified
- `Monitoring.tsx`: 
  - Added AI analysis button to findings
  - Integrated AIAnalysisDialog component
  - Added state for dialog management
  
- `AIAnalysisDialog.tsx`: 
  - Created new component
  - Implemented all analysis sections
  - Added loading and error states
  
- `api.ts`: 
  - Added `analyzeFinding()` function
  - Added new type definitions for analysis response
  - Added `flagged_records` to MonitoringResults

### Recent Changes
1. Imported Brain icon from lucide-react
2. Added handler `handleAnalyzeFinding()`
3. Enhanced findings display with two buttons
4. Added AI dialog state management
