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

# Initialize pipeline
pipeline = Pipeline.from_config("configs/cloud.yaml")

# Process audio file
bundle = pipeline.process_file("recording.wav")

# Output FHIR JSON
print(bundle.to_json())
```

### Command Line

```bash
# Process single file
python -m voice_to_fhir process recording.wav --output result.json

# Real-time capture
python -m voice_to_fhir capture --workflow intake --output encounter.json

# Batch processing
python -m voice_to_fhir batch ./recordings/ --output ./fhir_output/
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

Pre-configured prompts and FHIR mappings for common documentation scenarios:

### Ambulatory
- [Patient Intake](clinical-workflows/ambulatory/intake.md)
- [Provider Examination](clinical-workflows/ambulatory/provider-exam.md)
- [Visit Checkout](clinical-workflows/ambulatory/checkout.md)

### Inpatient
- [Hospital Admission](clinical-workflows/inpatient/admission.md)
- [Daily Rounds](clinical-workflows/inpatient/rounding.md)
- [Nursing Handoff](clinical-workflows/inpatient/nursing-handoff.md)
- [Discharge Summary](clinical-workflows/inpatient/discharge.md)

### Emergency / Critical Care
- [ED Triage](clinical-workflows/emergency/triage.md)
- [Trauma Documentation](clinical-workflows/emergency/trauma.md)
- [ICU Procedures](clinical-workflows/icu/procedures.md)

### Pre-Hospital
- [Ambulance On-Scene](clinical-workflows/ambulance/on-scene.md)
- [Transport Documentation](clinical-workflows/ambulance/transport.md)

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
├── src/
│   └── voice_to_fhir/
│       ├── capture/          # Audio capture and VAD
│       ├── transcription/    # MedASR integration
│       ├── extraction/       # MedGemma structured extraction
│       ├── fhir/             # FHIR R4 transformation
│       └── pipeline/         # End-to-end orchestration
├── reference-architectures/  # Hardware deployment guides
├── clinical-workflows/       # Use case documentation
├── configs/                  # Configuration templates
├── examples/                 # Usage examples
├── tests/                    # Test suite
└── scripts/                  # Utility scripts
```

## Configuration

Configuration is managed via YAML files:

```yaml
# configs/example.yaml
pipeline:
  name: "voice-to-fhir"

capture:
  sample_rate: 16000
  vad_enabled: true

transcription:
  backend: "cloud"  # or "local"
  model: "google/medasr"

extraction:
  backend: "cloud"
  model: "google/medgemma-4b"
  workflow: "general"

fhir:
  version: "R4"
  validate: true
```

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
