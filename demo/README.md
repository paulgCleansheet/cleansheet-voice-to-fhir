# Voice-to-FHIR Demo

End-to-end clinical documentation demo that converts voice recordings to structured FHIR R4 bundles.

## Quick Start

1. **Start the backend servers:**
   ```bash
   cd cleansheet-voice-to-fhir
   python launch_servers.py
   ```
   Wait for both servers to initialize:
   - MedASR Local Server (port 3002)
   - Voice-to-FHIR API (port 8001)

2. **Open the demo:**
   ```bash
   # Option A: Direct file open
   start demo/index.html          # Windows
   open demo/index.html           # macOS

   # Option B: Local server (recommended)
   python -m http.server 8000
   # Then visit http://localhost:8000/demo/
   ```

3. **Verify connection:** The header shows "Connected (v0.1.0)" when the API is reachable.

## Features

### Input Panel
- **Note Type Selector**: 13 clinical workflow types (General Encounter, Emergency, H&P, etc.)
- **Voice Recording**: Click the microphone to record, click again to stop
- **File Upload**: Drag-and-drop or click to upload `.wav`, `.webm`, `.mp3`, `.ogg`, `.m4a` files
- **Audio Preview**: Playback recorded/uploaded audio before processing

### Processing Queue
- **Queue Management**: Add multiple items and process them together
- **Click to Navigate**: Click any completed (green checkmark) item to view its review
- **Selection Indicator**: Blue border highlights the currently selected item
- **Parallel Mode**: Toggle between sequential and parallel processing (up to 3 concurrent)
- **Metrics Display**: Shows timing breakdown (transcription, extraction, FHIR transform)
- **Error Handling**: Retry failed items or remove from queue

### Clinician Review
Three-tab interface for reviewing extracted data:

**Tab 1: Clinical Notes**
- Auto-generated note sections based on workflow type
- Editable content (click to edit)
- Original transcript reference panel

**Tab 2: Structured EHR Data**
- Categorized display: Vitals, Conditions, Medications, Allergies, Labs, Procedures, Family History
- Approve/reject individual items
- **ICD-10 codes** automatically added to conditions via verified lookup database (500+ codes with fuzzy matching)

**Tab 3: Clinician Orders**
- Extracted orders: Medications, Labs, Consults, Procedures
- Approve/reject workflow before submission

### Export Features
- **Bulk Export**: Export all processed items as a single JSON file for analysis
- **Individual Export**: Export each item as a separate `.actual.json` file for comparison with expected outputs
- **Export Format**: Includes metadata, patient info, EHR data, orders, and original transcript

### Mock EHR
- Stores approved submissions locally (localStorage)
- View FHIR bundles as JSON
- Clear all records

## Backend Features

### Post-Processing Pipeline
After MedGemma extraction, the backend applies automatic enhancement:

**Transcript Marker Extraction:**
- Extracts chief complaint from `[CHIEF COMPLAINT]`, `CC:`, or "presents with" patterns
- Parses `[FAMILY HISTORY]` for relationship + condition pairs
- Extracts tobacco, alcohol, occupation from `[SOCIAL HISTORY]` section

**Validation & Filtering:**
- Removes placeholder values ("null", "not mentioned", "unknown", etc.)
- Filters invalid vitals and allergies
- Removes non-medication items from medication orders

### API Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/workflows` | GET | List available note types |
| `/api/v1/process-audio` | POST | Process audio through full pipeline |

## Workflow Types

| ID | Name | Use Case |
|----|------|----------|
| `general` | General Encounter | Standard clinical encounters |
| `emergency` | Emergency / Trauma | ED visits, trauma, acute care |
| `intake` | Patient Intake | New patient registration |
| `followup` | Follow-up Visit | Return visits, progress notes |
| `procedure` | Procedure Note | Surgical documentation |
| `discharge` | Discharge Summary | Hospital discharge |
| `radiology` | Radiology Dictation | Imaging interpretation |
| `lab_review` | Lab Review | Laboratory result review |
| `respiratory` | Respiratory Assessment | RT assessments |
| `icu` | ICU / Critical Care | Critical care documentation |
| `cardiology` | Cardiology | Cardiac encounters |
| `pediatrics` | Pediatrics | Pediatric encounters |
| `neurology` | Neurology | Neurological assessments |

## Demo Mode

When no `HF_TOKEN` is configured on the server, it returns synthetic demo data. This allows testing the UI without model access.

## Keyboard Shortcuts

- **Space** (when focused on record button): Start/stop recording

## Browser Support

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 90+ | Full support |
| Firefox | 88+ | Full support |
| Safari | 14+ | Full support |
| Edge | 90+ | Full support |

Note: Microphone access requires HTTPS or localhost.

## Troubleshooting

**"Disconnected" status:**
- Ensure `launch_servers.py` is running
- Check that port 8001 is not blocked
- Verify the API at http://localhost:8001/docs

**Recording doesn't work:**
- Check browser microphone permissions
- Use HTTPS or localhost (not file://)
- Try a different browser

**Processing fails:**
- Check server console for errors
- Verify HuggingFace token if using cloud models
- Try demo mode (no token required)

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         demo/index.html                               │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Input    │  │   Queue      │  │   Review     │  │  Mock EHR  │ │
│  │   Panel    │  │   Panel      │  │   Panel      │  │   Panel    │ │
│  └────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
                  localhost:8001 (FastAPI)
                  ├── /api/v1/health
                  ├── /api/v1/workflows
                  └── /api/v1/process-audio
```

## Design System

Follows CleansheetMedical design patterns:
- **Fonts**: Questrial (headings), Barlow Light (body)
- **Primary Color**: #0066CC
- **Icons**: Phosphor Icons
