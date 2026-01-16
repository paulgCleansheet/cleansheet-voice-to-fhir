"""
Tests for pipeline configuration.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from voice_to_fhir.pipeline.config import (
    PipelineConfig,
    CaptureConfig,
    TranscriptionConfig,
    ExtractionConfig,
    FHIROutputConfig,
    load_config,
)


class TestCaptureConfig:
    """Tests for CaptureConfig."""

    def test_default_values(self):
        """Test default capture configuration."""
        config = CaptureConfig()

        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_duration_ms == 100
        assert config.vad_enabled is True
        assert config.vad_mode == 3

    def test_custom_values(self):
        """Test custom capture configuration."""
        config = CaptureConfig(
            sample_rate=8000,
            channels=2,
            chunk_duration_ms=50,
            vad_enabled=False,
        )

        assert config.sample_rate == 8000
        assert config.channels == 2
        assert config.vad_enabled is False


class TestTranscriptionConfig:
    """Tests for TranscriptionConfig."""

    def test_default_values(self):
        """Test default transcription configuration."""
        config = TranscriptionConfig()

        assert config.backend == "cloud"
        assert config.model_id == "google/medasr"
        assert config.device == "cuda"

    def test_local_backend(self):
        """Test local backend configuration."""
        config = TranscriptionConfig(
            backend="local",
            model_path="/models/medasr",
            device="cpu",
        )

        assert config.backend == "local"
        assert config.model_path == "/models/medasr"


class TestExtractionConfig:
    """Tests for ExtractionConfig."""

    def test_default_values(self):
        """Test default extraction configuration."""
        config = ExtractionConfig()

        assert config.backend == "cloud"
        assert config.model_id == "google/medgemma-4b"
        assert config.max_tokens == 2048
        assert config.temperature == 0.1

    def test_custom_workflow(self):
        """Test custom workflow configuration."""
        config = ExtractionConfig(
            workflow="emergency",
            max_tokens=4096,
        )

        assert config.workflow == "emergency"
        assert config.max_tokens == 4096


class TestFHIROutputConfig:
    """Tests for FHIROutputConfig."""

    def test_default_values(self):
        """Test default FHIR output configuration."""
        config = FHIROutputConfig()

        assert config.version == "R4"
        assert config.validate is True
        assert config.output_format == "json"

    def test_custom_base_url(self):
        """Test custom base URL."""
        config = FHIROutputConfig(
            base_url="http://my-fhir-server.example.org/fhir",
        )

        assert config.base_url == "http://my-fhir-server.example.org/fhir"


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_values(self):
        """Test default pipeline configuration."""
        config = PipelineConfig()

        assert config.name == "voice-to-fhir"
        assert isinstance(config.capture, CaptureConfig)
        assert isinstance(config.transcription, TranscriptionConfig)
        assert isinstance(config.extraction, ExtractionConfig)
        assert isinstance(config.fhir, FHIROutputConfig)

    def test_from_dict(self, sample_config_dict: dict[str, Any]):
        """Test creating config from dictionary."""
        config = PipelineConfig.from_dict(sample_config_dict)

        assert config.name == "test-pipeline"
        assert config.capture.sample_rate == 16000
        assert config.transcription.backend == "cloud"

    def test_from_dict_partial(self):
        """Test creating config from partial dictionary."""
        data = {
            "name": "partial-config",
            "transcription": {
                "backend": "local",
            },
        }

        config = PipelineConfig.from_dict(data)

        assert config.name == "partial-config"
        assert config.transcription.backend == "local"
        # Other values should be defaults
        assert config.capture.sample_rate == 16000

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = PipelineConfig()
        data = config.to_dict()

        assert data["name"] == "voice-to-fhir"
        assert "capture" in data
        assert "transcription" in data
        assert "extraction" in data
        assert "fhir" in data
        assert data["capture"]["sample_rate"] == 16000

    def test_roundtrip(self):
        """Test config survives dict roundtrip."""
        original = PipelineConfig()
        original.capture.sample_rate = 8000
        original.transcription.backend = "local"

        data = original.to_dict()
        restored = PipelineConfig.from_dict(data)

        assert restored.capture.sample_rate == 8000
        assert restored.transcription.backend == "local"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_file(self, temp_config_file: Path):
        """Test loading config from file."""
        config = load_config(temp_config_file)

        assert config.name == "test-pipeline"
        assert config.transcription.model_id == "google/medasr"

    def test_load_config_not_found(self, tmp_path: Path):
        """Test loading non-existent config file."""
        fake_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_config(fake_path)

    def test_load_config_invalid_yaml(self, tmp_path: Path):
        """Test loading invalid YAML file."""
        invalid_path = tmp_path / "invalid.yaml"
        invalid_path.write_text("{ invalid yaml: [")

        with pytest.raises(Exception):  # yaml.YAMLError
            load_config(invalid_path)

    def test_load_minimal_config(self, tmp_path: Path):
        """Test loading minimal config file."""
        config_path = tmp_path / "minimal.yaml"
        config_path.write_text("name: minimal\n")

        config = load_config(config_path)

        assert config.name == "minimal"
        # All other values should be defaults
        assert config.capture.sample_rate == 16000

    def test_load_full_config(self, tmp_path: Path):
        """Test loading full config file."""
        full_config = {
            "name": "full-test",
            "version": "2.0.0",
            "capture": {
                "sample_rate": 8000,
                "channels": 2,
                "chunk_duration_ms": 50,
                "vad_enabled": False,
                "vad_mode": 2,
            },
            "transcription": {
                "backend": "local",
                "model_id": "custom/asr",
                "model_path": "/models/custom-asr",
                "device": "cpu",
                "precision": "fp32",
                "use_tensorrt": False,
            },
            "extraction": {
                "backend": "local",
                "model_id": "custom/llm",
                "model_path": "/models/custom-llm",
                "device": "cpu",
                "precision": "fp16",
                "max_tokens": 4096,
                "temperature": 0.2,
                "workflow": "emergency",
            },
            "fhir": {
                "version": "R4",
                "base_url": "http://example.org/fhir",
                "validate": False,
                "output_format": "ndjson",
            },
        }

        config_path = tmp_path / "full.yaml"
        with open(config_path, "w") as f:
            yaml.dump(full_config, f)

        config = load_config(config_path)

        assert config.name == "full-test"
        assert config.capture.sample_rate == 8000
        assert config.transcription.device == "cpu"
        assert config.extraction.workflow == "emergency"
        assert config.fhir.validate is False
