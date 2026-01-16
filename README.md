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
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Voice     │───▶│   MedASR    │───▶│  MedGemma   │───▶│  FHIR R4    │
│   Input     │    │ Transcribe  │    │  Structure  │    │   Bundle    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Features

- **Edge Deployment** - Run locally on NVIDIA Jetson, Intel NUC, or Raspberry Pi
- **Cloud Hybrid** - Automatic fallback to HuggingFace Inference API
- **Clinical Workflows** - Pre-built prompts for intake, charting, emergency, ICU
- **FHIR R4 Output** - Standard format compatible with any FHIR-compliant EHR
- **Real-time Streaming** - Sub-second latency on supported hardware

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
│   │   └── prompts/          # Workflow-specific prompts
│   ├── fhir/                 # FHIR R4 transformation
│   ├── pipeline/             # End-to-end orchestration
│   └── cli.py                # Command-line interface
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

- [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations) for MedASR and MedGemma
- [HL7 FHIR](https://www.hl7.org/fhir/) for the healthcare interoperability standard
- [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) on Kaggle

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
