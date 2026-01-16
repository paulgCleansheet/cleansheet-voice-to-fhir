# Examples

This directory contains example scripts demonstrating various use cases for the voice-to-FHIR pipeline.

## Prerequisites

Before running examples, ensure you have:

1. Installed the package: `pip install -e .`
2. Set the HuggingFace token: `export HF_TOKEN=your_token_here`

## Quick Start

### Basic Usage
```bash
python examples/basic_usage.py
```
Simplest example - processes a transcript and outputs FHIR JSON.

### Process Audio File
```bash
python examples/process_audio_file.py recording.wav --output result.json
```
Process an audio file through the full pipeline.

### Live Capture
```bash
python examples/live_capture.py --duration 30
```
Capture audio from microphone with voice activity detection.

## Examples Overview

| Script | Description |
|--------|-------------|
| `basic_usage.py` | Simplest transcript-to-FHIR example |
| `process_audio_file.py` | Process audio files with options |
| `live_capture.py` | Real-time microphone capture |
| `batch_processing.py` | Process multiple files in batch |
| `workflow_comparison.py` | Compare different clinical workflows |
| `custom_config.py` | Create custom configurations programmatically |
| `fhir_server_integration.py` | Post bundles to FHIR server |
| `extract_entities_only.py` | Entity extraction without FHIR transform |
| `streaming_realtime.py` | Real-time streaming with partial results |

## Workflow Examples

The pipeline supports different clinical workflows, each optimized for specific documentation contexts:

```bash
# General clinical encounter
python examples/process_audio_file.py recording.wav --workflow general

# Emergency department
python examples/process_audio_file.py recording.wav --workflow emergency

# Patient intake
python examples/process_audio_file.py recording.wav --workflow intake

# Follow-up visit
python examples/process_audio_file.py recording.wav --workflow followup

# Procedure documentation
python examples/process_audio_file.py recording.wav --workflow procedure

# Discharge summary
python examples/process_audio_file.py recording.wav --workflow discharge

# Radiology dictation
python examples/process_audio_file.py recording.wav --workflow radiology

# Lab results review
python examples/process_audio_file.py recording.wav --workflow lab_review
```

## Configuration Examples

Use predefined configurations from the `configs/` directory:

```bash
# Cloud backend (default)
python examples/process_audio_file.py recording.wav --config configs/cloud.yaml

# Local GPU inference
python examples/process_audio_file.py recording.wav --config configs/local.yaml

# Edge deployment (Jetson)
python examples/process_audio_file.py recording.wav --config configs/edge-jetson.yaml

# Emergency department optimized
python examples/process_audio_file.py recording.wav --config configs/emergency.yaml
```

## Output

All examples output FHIR R4 Bundles. Example output:

```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Condition",
        "id": "condition-1",
        "code": {
          "coding": [
            {
              "system": "http://hl7.org/fhir/sid/icd-10",
              "code": "I10",
              "display": "Essential hypertension"
            }
          ]
        }
      }
    }
  ]
}
```
