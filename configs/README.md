# Configuration Files

This directory contains example configuration files for different deployment scenarios.

## Available Configurations

| Config | Backend | Use Case |
|--------|---------|----------|
| `cloud.yaml` | HuggingFace API | Development, testing, low-volume |
| `local.yaml` | Local GPU | Privacy-sensitive, offline, high-volume |
| `edge-jetson.yaml` | Jetson GPU | Mobile carts, ambulances, remote clinics |
| `edge-cpu.yaml` | CPU only | Low-cost hardware without GPU |
| `emergency.yaml` | Cloud | Emergency department workflow |
| `radiology.yaml` | Cloud | Radiology dictation workflow |

## Usage

### With CLI

```bash
# Process with specific config
voice-to-fhir process recording.wav --config configs/cloud.yaml

# Batch process with config
voice-to-fhir batch input/ output/ --config configs/local.yaml
```

### With Python API

```python
from voice_to_fhir import Pipeline

# Load from config file
pipeline = Pipeline.from_config("configs/cloud.yaml")

# Process transcript
bundle = pipeline.process_transcript("Patient has chest pain...")
```

## Configuration Reference

### Full Configuration Structure

```yaml
name: pipeline-name
version: "1.0.0"

capture:
  sample_rate: 16000          # Audio sample rate (Hz)
  channels: 1                 # Mono (1) or stereo (2)
  chunk_duration_ms: 100      # Audio chunk size for VAD
  vad_enabled: true           # Voice activity detection
  vad_mode: 3                 # 0-3, higher = more aggressive

transcription:
  backend: cloud              # cloud or local
  model_id: google/medasr     # HuggingFace model ID (cloud)
  model_path: models/medasr   # Local model path
  device: cuda                # cuda or cpu
  precision: fp16             # fp32, fp16, int8
  use_tensorrt: false         # TensorRT acceleration (Jetson)

extraction:
  backend: cloud              # cloud or local
  model_id: google/medgemma-4b
  model_path: models/medgemma-4b
  device: cuda
  precision: int8
  max_tokens: 2048            # Max output tokens
  temperature: 0.1            # Lower = more deterministic
  workflow: general           # Default workflow

fhir:
  version: R4                 # FHIR version
  base_url: http://example.org/fhir
  validate: true              # Validate output bundles
  output_format: json         # json or ndjson
```

## Deployment Scenarios

### Development / Testing
Use `cloud.yaml` - simplest setup, just needs HF_TOKEN.

### Production - Privacy Sensitive
Use `local.yaml` - all processing on-premise, no data leaves network.

### Edge - Mobile Medical Cart
Use `edge-jetson.yaml` - optimized for Jetson with TensorRT acceleration.

### Edge - Budget Hardware
Use `edge-cpu.yaml` - runs on any x86/ARM device with sufficient RAM.

### Specialty Workflows
Use `emergency.yaml` or `radiology.yaml` for workflow-specific optimizations.

## Environment Variables

Configurations can reference environment variables:

- `HF_TOKEN` - HuggingFace API token (required for cloud backend)
- `CUDA_VISIBLE_DEVICES` - GPU selection for local inference

## Creating Custom Configurations

1. Copy a base config: `cp configs/cloud.yaml configs/my-config.yaml`
2. Edit settings as needed
3. Use with: `--config configs/my-config.yaml`

Or create programmatically:

```python
from voice_to_fhir.pipeline.config import PipelineConfig

config = PipelineConfig()
config.extraction.workflow = "emergency"
config.extraction.temperature = 0.05

# Save to file
import yaml
with open("my-config.yaml", "w") as f:
    yaml.dump(config.to_dict(), f)
```
