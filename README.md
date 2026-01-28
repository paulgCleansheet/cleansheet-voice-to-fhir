# Cleansheet Voice-to-FHIR Pipeline

Edge-deployable clinical voice documentation using MedASR and MedGemma.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FHIR R4](https://img.shields.io/badge/FHIR-R4-orange.svg)](https://www.hl7.org/fhir/)

## Overview

This pipeline transforms clinical voice recordings into structured FHIR R4 resources using Google's Health AI Developer Foundations (HAI-DEF) models:

- **MedASR** - Medical speech recognition with 58-82% better accuracy than general ASR
- **MedGemma** - Medical vision-language model for structured entity extraction

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Voice     │───▶│   MedASR    │───▶│  MedGemma   │───▶│    Post-    │───▶│  FHIR R4    │
│   Input     │    │ Transcribe  │    │  Structure  │    │  Processor  │    │   Bundle    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Features

- **Edge Deployment** - Run locally on NVIDIA Jetson, Intel NUC, or Raspberry Pi
- **Cloud Hybrid** - Automatic fallback to HuggingFace Inference API
- **Clinical Workflows** - Pre-built prompts for intake, charting, emergency, ICU, and specialty workflows
- **Post-Processing** - Automatic extraction from transcript markers and validation filtering
- **FHIR R4 Output** - Standard format compatible with any FHIR-compliant EHR
- **Real-time Streaming** - Sub-second latency on supported hardware
- **Web Demo** - Full-featured browser UI for recording, processing, and clinician review

## Quick Start

### Prerequisites

- Python 3.10+
- HuggingFace account with access to HAI-DEF models
- (Optional) NVIDIA GPU for local inference

### Installation

```bash
# Clone repository
git clone https://github.com/CleansheetLLC/cleansheet-voice-to-fhir.git
cd cleansheet-voice-to-fhir

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download models (requires HuggingFace token)
export HF_TOKEN=your_token_here
python scripts/download_models.py
```

### Basic Usage

```python
from voice_to_fhir import Pipeline

# Initialize with cloud backend (simplest)
pipeline = Pipeline.cloud()

# Process a transcript directly
transcript = """
Patient is a 45-year-old male with chest pain for 2 hours.
Blood pressure 150/90, heart rate 88. Taking lisinopril 10mg daily.
Allergic to penicillin.
"""
bundle = pipeline.process_transcript(transcript)
print(pipeline.to_json(bundle, indent=2))

# Or process an audio file
bundle = pipeline.process_file("recording.wav", workflow="general")
pipeline.save(bundle, "output.json")
```

### Command Line

```bash
# Process single file
voice-to-fhir process recording.wav --output result.json

# Process with specific workflow
voice-to-fhir process recording.wav --workflow emergency --output ed_note.json

# Real-time capture from microphone
voice-to-fhir capture --duration 30 --workflow intake

# Batch processing
voice-to-fhir batch ./recordings/ ./fhir_output/ --pattern "*.wav"

# List audio devices
voice-to-fhir devices
```

### Web Demo

A full-featured browser interface for end-to-end clinical documentation:

```bash
# Start backend servers
python launch_servers.py

# Open demo in browser
start demo/index.html  # Windows
open demo/index.html   # macOS
```

Features:
- Voice recording and file upload
- Processing queue with parallel mode
- Clinician review with editable notes
- Structured EHR data approval workflow
- Export to JSON for analysis

See [demo/README.md](demo/README.md) for full documentation.

### Run Examples

```bash
# See examples/ directory for more detailed usage
python examples/basic_usage.py
python examples/workflow_comparison.py
python examples/batch_processing.py input/ output/ --parallel 4
```

## Reference Architectures

Deployment guides for different hardware platforms:

| Platform | Use Case | Latency | Documentation |
|----------|----------|---------|---------------|
| **NVIDIA Jetson Orin** | Mobile workstation | <500ms | [Guide](reference-architectures/jetson-orin/) |
| **Intel NUC** | Push cart / desktop | <2s | [Guide](reference-architectures/intel-nuc/) |
| **Raspberry Pi 5** | Wall-mounted tablet | <5s | [Guide](reference-architectures/raspberry-pi/) |
| **Cloud (HuggingFace)** | Development / fallback | <3s | [Guide](reference-architectures/cloud-hybrid/) |

## Clinical Workflows

Pre-configured extraction prompt templates optimized for different clinical documentation contexts. Each workflow emphasizes relevant clinical data and uses appropriate output schemas.

### General Workflows

| Workflow | Use Case | Key Extractions |
|----------|----------|-----------------|
| `general` | Standard clinical encounters | Conditions, medications, allergies, vitals |
| `emergency` | ED visits, urgent care | Triage, acuity, critical findings, trauma |
| `intake` | Patient registration | Full history (medical, surgical, family, social) |
| `followup` | Return visits | Progress, medication efficacy, trends |
| `procedure` | Surgical documentation | Technique, specimens, complications |
| `discharge` | Hospital discharge | Medication reconciliation, follow-up |
| `radiology` | Imaging dictation | Modality, findings, impressions |
| `lab_review` | Lab results | Values, interpretations, critical flags |

### Specialist Workflows

| Workflow | Use Case | Key Extractions |
|----------|----------|-----------------|
| `respiratory` | RT assessments, ventilator management | Vent settings, ABGs, lung sounds, weaning parameters, O2 therapy |
| `icu` | Critical care documentation | Vasopressors, sedation (RASS), lines/drains, organ function, fluid balance |
| `cardiology` | Cardiac encounters | ECG findings, echo results, troponin, cath findings, risk stratification |
| `pediatrics` | Pediatric encounters | Growth percentiles, developmental milestones, immunizations, weight-based dosing |
| `neurology` | Neurological encounters | Mental status, cranial nerves, motor/sensory exam, NIHSS, seizure documentation |

### Usage

```bash
# Specify workflow via CLI
voice-to-fhir process recording.wav --workflow emergency

# Or via Python
bundle = pipeline.process_transcript(transcript, workflow="emergency")
```

### Workflow Comparison

Different workflows extract different information from the same input:

```bash
# See how workflows differ
python examples/workflow_comparison.py
```

## Post-Processing

After MedGemma extraction, an automatic post-processing step enhances the results:

### Transcript Marker Extraction

Many clinical transcripts use bracketed section markers. The post-processor extracts data from these when MedGemma misses them:

| Marker | Extracted Data |
|--------|----------------|
| `[CHIEF COMPLAINT]` | Chief complaint text |
| `[FAMILY HISTORY]` | Family member + condition pairs |
| `[SOCIAL HISTORY]` | Tobacco, alcohol, occupation, drugs |
| `CC:` | Chief complaint (alternative format) |

Also recognizes natural language patterns like "presents with", "mother has diabetes", "former smoker quit 10 years ago".

### Validation & Filtering

Automatically removes invalid or placeholder data:

- **Placeholder values**: "null", "not mentioned", "unknown", "n/a", etc.
- **Invalid vitals**: Vitals with placeholder values instead of measurements
- **Invalid allergies**: Allergies with "null" or "unknown" as the substance
- **Non-medications**: Items like "await pathology results" in medication orders

This ensures cleaner data for clinician review and downstream systems.

### ICD-10 Code Enrichment

Conditions are automatically enriched with verified ICD-10-CM codes using a local lookup database:

```
Extracted Condition     →  ICD-10-CM Code
─────────────────────────────────────────
hypertension            →  I10 (Essential hypertension)
type 2 diabetes         →  E11.9 (Type 2 diabetes mellitus)
acute coronary syndrome →  I24.9 (Acute ischemic heart disease)
chest pain              →  R07.9 (Chest pain, unspecified)
```

**Why a lookup database instead of LLM-generated codes?**

LLM-generated ICD-10 codes are unreliable for billing and compliance:
- Models hallucinate plausible-looking but invalid codes
- No verification against official ICD-10-CM code sets
- Inconsistent coding for the same condition

The lookup approach provides:
- **500+ verified codes** covering common conditions
- **Synonym matching** ("heart attack" → "myocardial infarction" → I21.9)
- **Fuzzy matching** (85% threshold) for minor spelling variations
- **Confidence scores** indicating match quality

Codes are sourced from the official CMS ICD-10-CM code set.

### RxNorm Medication Verification

Medications are verified against a local RxNorm-based database with fuzzy matching to handle transcription errors:

```
Transcribed Medication  →  Verified Status
─────────────────────────────────────────
lisinopril              →  ✓ Verified (RxCUI: 29046)
atorvastatin            →  ✓ Verified (RxCUI: 83367)
pirro lactone           →  ✗ Unverified (spironolactone?)
Toroptooum              →  ✗ Unverified (tiotropium?)
```

The verification module provides:
- **200+ common medications** organized by drug class
- **Brand-to-generic mapping** (Lipitor → atorvastatin)
- **Fuzzy matching** (85% threshold) using SequenceMatcher
- **Drug class identification** for clinical decision support

Unverified medications are flagged in the clinician review UI with yellow alerts, prompting manual verification before approval.

### Order-Diagnosis Linking

Orders (medications, labs, consults, procedures) are automatically linked to diagnoses using clinical rules:

```
Order                   →  Linked Diagnosis
─────────────────────────────────────────
atorvastatin 40mg       →  E78.5 (Hyperlipidemia)
HbA1c lab order         →  E11.9 (Type 2 diabetes)
Cardiology consult      →  I25.10 (CAD) or I50.9 (CHF)
Echocardiogram          →  I50.9 (Heart failure)
```

The linking algorithm:
1. **Matches against patient conditions first** (highest confidence)
2. **Falls back to clinical rules** based on drug class, lab type, or specialty
3. **Supports manual override** via clinician review UI

Clinical rule coverage:
- **Medications**: 40+ drug classes with typical indications
- **Labs**: 50+ common tests with monitoring indications
- **Consults**: 30+ specialties with typical referral conditions
- **Procedures**: 50+ procedures with typical indications

### Autocomplete Databases

The demo UI provides autocomplete suggestions for manual data entry:

| Field Type | Data Source | Coverage |
|------------|-------------|----------|
| Medications | RxNorm lookup database | ~200 common medications |
| Conditions | ICD-10-CM lookup database | ~500 conditions |
| Lab Tests | LOINC-based list | ~50 common panels/tests |
| Procedures | Common procedure list | ~40 procedures |
| Consult Specialties | Medical specialty list | ~25 specialties |
| Allergies | Common allergen list | ~30 allergens |

Autocomplete uses prefix matching with debounced search for responsive performance.

## FHIR Output

The pipeline generates standard FHIR R4 resources:

| Resource Type | Generated From |
|---------------|----------------|
| `Patient` | Demographics mentioned in speech |
| `Encounter` | Visit context and workflow type |
| `Condition` | Chief complaint, diagnoses |
| `Observation` | Vital signs, clinical findings |
| `MedicationRequest` | Medication orders |
| `MedicationStatement` | Current medications |
| `Procedure` | Procedures performed |
| `AllergyIntolerance` | Allergies mentioned |
| `ClinicalImpression` | Assessment and plan |

Example output:

```json
{
  "resourceType": "Bundle",
  "type": "transaction",
  "entry": [
    {
      "resource": {
        "resourceType": "Condition",
        "code": {
          "coding": [{
            "system": "http://snomed.info/sct",
            "code": "29857009",
            "display": "Chest pain"
          }]
        },
        "subject": {"reference": "Patient/example"}
      }
    }
  ]
}
```

## Project Structure

```
cleansheet-voice-to-fhir/
├── src/voice_to_fhir/
│   ├── capture/              # Audio capture and VAD
│   ├── transcription/        # MedASR integration
│   ├── extraction/           # MedGemma structured extraction
│   │   ├── prompts/          # Workflow-specific prompts
│   │   ├── medgemma_client.py  # MedGemma API client
│   │   └── post_processor.py   # Transcript marker extraction & validation
│   ├── fhir/                 # FHIR R4 transformation
│   ├── pipeline/             # End-to-end orchestration
│   └── cli.py                # Command-line interface
├── demo/                     # Web demo UI
│   ├── index.html            # Full-featured browser interface
│   └── README.md             # Demo documentation
├── configs/                  # Configuration templates
│   ├── cloud.yaml            # HuggingFace API backend
│   ├── local.yaml            # Local GPU inference
│   ├── edge-jetson.yaml      # NVIDIA Jetson deployment
│   └── emergency.yaml        # ED-optimized settings
├── examples/                 # Usage examples
│   ├── basic_usage.py        # Simplest example
│   ├── live_capture.py       # Microphone capture
│   ├── batch_processing.py   # Multi-file processing
│   └── ...                   # More examples
├── tests/                    # Test suite
└── scripts/                  # Utility scripts
    ├── download_models.py    # Model downloader
    └── boundary_check.py     # IP protection scanner
```

## Configuration

Pre-built configurations for different deployment scenarios:

| Config | Backend | Use Case |
|--------|---------|----------|
| `configs/cloud.yaml` | HuggingFace API | Development, testing |
| `configs/local.yaml` | Local GPU | Privacy-sensitive, offline |
| `configs/edge-jetson.yaml` | Jetson + TensorRT | Mobile carts, ambulances |
| `configs/edge-cpu.yaml` | CPU only | Low-cost hardware |
| `configs/emergency.yaml` | Cloud | ED workflow optimized |
| `configs/radiology.yaml` | Cloud | Radiology dictation |

### Usage

```bash
# Use predefined config
voice-to-fhir process recording.wav --config configs/local.yaml

# Or in Python
pipeline = Pipeline.from_config("configs/emergency.yaml")
```

### Configuration Structure

```yaml
name: voice-to-fhir
version: "1.0.0"

capture:
  sample_rate: 16000        # Audio sample rate (Hz)
  channels: 1               # Mono audio
  vad_enabled: true         # Voice activity detection
  vad_mode: 3               # 0-3, higher = more aggressive

transcription:
  backend: cloud            # cloud or local
  model_id: google/medasr   # HuggingFace model ID

extraction:
  backend: cloud            # cloud or local
  model_id: google/medgemma-4b
  max_tokens: 2048
  temperature: 0.1
  workflow: general         # Default workflow

fhir:
  version: R4
  validate: true
  output_format: json
```

See [configs/README.md](configs/README.md) for full configuration reference.

## Examples

The `examples/` directory contains scripts demonstrating various use cases:

| Example | Description |
|---------|-------------|
| `basic_usage.py` | Simplest transcript-to-FHIR |
| `process_audio_file.py` | Process audio with CLI options |
| `live_capture.py` | Real-time microphone capture |
| `batch_processing.py` | Parallel multi-file processing |
| `workflow_comparison.py` | Compare workflow outputs |
| `custom_config.py` | Programmatic configuration |
| `fhir_server_integration.py` | Post bundles to FHIR server |
| `extract_entities_only.py` | Extraction without FHIR transform |

```bash
# Run any example
python examples/basic_usage.py
python examples/live_capture.py --duration 30
python examples/batch_processing.py input/ output/ --parallel 4
```

See [examples/README.md](examples/README.md) for detailed documentation.

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
ruff check src/

# Run type checking
mypy src/

# Run boundary check (IP protection)
python scripts/boundary_check.py
```

## Benchmarks

Performance on reference hardware (end-to-end latency):

| Platform | Transcription | Extraction | Total | Notes |
|----------|---------------|------------|-------|-------|
| Jetson Orin 64GB | 180ms | 250ms | 480ms | TensorRT FP16 |
| Intel NUC i7 | 450ms | 1.2s | 1.8s | ONNX Runtime |
| Raspberry Pi 5 | 1.5s (cloud) | 2.8s (cloud) | 4.5s | Hybrid mode |
| Cloud only | 800ms | 1.5s | 2.5s | HuggingFace API |

See [benchmarks/](benchmarks/) for detailed methodology and results.

## Acknowledgments

### Models & Standards
- [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations) for MedASR and MedGemma
- [HL7 FHIR](https://www.hl7.org/fhir/) for the healthcare interoperability standard
- [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) on Kaggle

### Clinical Terminology & Data Sources
- **ICD-10-CM**: Centers for Medicare & Medicaid Services (CMS) — Public domain
- **RxNorm**: National Library of Medicine (NLM) — [UMLS License](https://www.nlm.nih.gov/databases/umls.html) (free for US entities)
- **LOINC**: Regenstrief Institute — [LOINC License](https://loinc.org/license/) (free, attribution required)

The medication and condition lookup databases in this project are curated subsets derived from these public terminology standards. They are provided for demonstration and research purposes.

### Libraries
- [difflib](https://docs.python.org/3/library/difflib.html) — Python standard library for fuzzy string matching

## License

This project is licensed under [CC BY 4.0](LICENSE).

When using this software, please provide attribution:

> Voice-to-FHIR Pipeline by Cleansheet LLC
> https://github.com/CleansheetLLC/cleansheet-voice-to-fhir

## Disclaimer

This software is intended for research and development purposes. It is **not** a medical device and has **not** been cleared or approved by the FDA or any other regulatory body for clinical use.

The output of this pipeline should be reviewed by qualified healthcare professionals before being used for clinical decision-making.

---

**Note:** This pipeline outputs standard FHIR R4 resources that can be consumed by any FHIR-compliant EHR system. The [Cleansheet Medical](https://cleansheet.info) platform is one such consumer, available separately under commercial license. This open source pipeline contains no proprietary UI, clinical decision support algorithms, or platform-specific integrations.
