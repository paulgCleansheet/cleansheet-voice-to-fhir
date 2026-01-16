"""
Pipeline Configuration

Configuration management for voice-to-FHIR pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CaptureConfig:
    """Audio capture configuration."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 100
    vad_enabled: bool = True
    vad_mode: int = 3


@dataclass
class TranscriptionConfig:
    """Transcription configuration."""

    backend: str = "cloud"  # cloud, local
    model_id: str = "google/medasr"
    model_path: str = "models/medasr"
    device: str = "cuda"
    precision: str = "fp16"
    use_tensorrt: bool = False


@dataclass
class ExtractionConfig:
    """Extraction configuration."""

    backend: str = "cloud"  # cloud, local
    model_id: str = "google/medgemma-4b"
    model_path: str = "models/medgemma-4b"
    device: str = "cuda"
    precision: str = "int8"
    max_tokens: int = 2048
    temperature: float = 0.1
    workflow: str = "general"
    prompts_dir: str = "src/voice_to_fhir/extraction/prompts"


@dataclass
class FHIROutputConfig:
    """FHIR output configuration."""

    version: str = "R4"
    base_url: str = "http://example.org/fhir"
    validate: bool = True
    output_format: str = "json"  # json, ndjson


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    name: str = "voice-to-fhir"
    version: str = "0.1.0"

    capture: CaptureConfig = field(default_factory=CaptureConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    fhir: FHIROutputConfig = field(default_factory=FHIROutputConfig)

    # API credentials (from environment if not set)
    hf_token: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary."""
        config = cls()

        if "name" in data:
            config.name = data["name"]
        if "version" in data:
            config.version = data["version"]
        if "hf_token" in data:
            config.hf_token = data["hf_token"]

        # Capture config
        if "capture" in data:
            cap = data["capture"]
            config.capture = CaptureConfig(
                sample_rate=cap.get("sample_rate", 16000),
                channels=cap.get("channels", 1),
                chunk_duration_ms=cap.get("chunk_duration_ms", 100),
                vad_enabled=cap.get("vad_enabled", True),
                vad_mode=cap.get("vad_mode", 3),
            )

        # Transcription config
        if "transcription" in data:
            trans = data["transcription"]
            config.transcription = TranscriptionConfig(
                backend=trans.get("backend", "cloud"),
                model_id=trans.get("model_id", "google/medasr"),
                model_path=trans.get("model_path", "models/medasr"),
                device=trans.get("device", "cuda"),
                precision=trans.get("precision", "fp16"),
                use_tensorrt=trans.get("use_tensorrt", False),
            )

        # Extraction config
        if "extraction" in data:
            ext = data["extraction"]
            config.extraction = ExtractionConfig(
                backend=ext.get("backend", "cloud"),
                model_id=ext.get("model_id", "google/medgemma-4b"),
                model_path=ext.get("model_path", "models/medgemma-4b"),
                device=ext.get("device", "cuda"),
                precision=ext.get("precision", "int8"),
                max_tokens=ext.get("max_tokens", 2048),
                temperature=ext.get("temperature", 0.1),
                workflow=ext.get("workflow", "general"),
                prompts_dir=ext.get("prompts_dir", "src/voice_to_fhir/extraction/prompts"),
            )

        # FHIR config
        if "fhir" in data:
            fhir = data["fhir"]
            config.fhir = FHIROutputConfig(
                version=fhir.get("version", "R4"),
                base_url=fhir.get("base_url", "http://example.org/fhir"),
                validate=fhir.get("validate", True),
                output_format=fhir.get("output_format", "json"),
            )

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "capture": {
                "sample_rate": self.capture.sample_rate,
                "channels": self.capture.channels,
                "chunk_duration_ms": self.capture.chunk_duration_ms,
                "vad_enabled": self.capture.vad_enabled,
                "vad_mode": self.capture.vad_mode,
            },
            "transcription": {
                "backend": self.transcription.backend,
                "model_id": self.transcription.model_id,
                "model_path": self.transcription.model_path,
                "device": self.transcription.device,
                "precision": self.transcription.precision,
                "use_tensorrt": self.transcription.use_tensorrt,
            },
            "extraction": {
                "backend": self.extraction.backend,
                "model_id": self.extraction.model_id,
                "model_path": self.extraction.model_path,
                "device": self.extraction.device,
                "precision": self.extraction.precision,
                "max_tokens": self.extraction.max_tokens,
                "temperature": self.extraction.temperature,
                "workflow": self.extraction.workflow,
            },
            "fhir": {
                "version": self.fhir.version,
                "base_url": self.fhir.base_url,
                "validate": self.fhir.validate,
                "output_format": self.fhir.output_format,
            },
        }


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load configuration from YAML file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return PipelineConfig.from_dict(data)
