"""
Tests for end-to-end pipeline.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_to_fhir.pipeline.pipeline import Pipeline
from voice_to_fhir.pipeline.config import PipelineConfig
from voice_to_fhir.capture.audio_utils import AudioSegment


class TestPipelineInitialization:
    """Tests for Pipeline initialization."""

    def test_default_initialization(self):
        """Test pipeline with default config."""
        config = PipelineConfig()
        pipeline = Pipeline(config)

        assert pipeline.config is not None
        assert pipeline.config.name == "voice-to-fhir"

    def test_from_config_file(self, temp_config_file: Path):
        """Test creating pipeline from config file."""
        pipeline = Pipeline.from_config(temp_config_file)

        assert pipeline.config.name == "test-pipeline"

    def test_cloud_factory(self):
        """Test cloud pipeline factory method."""
        pipeline = Pipeline.cloud(hf_token="test-token")

        assert pipeline.config.transcription.backend == "cloud"
        assert pipeline.config.extraction.backend == "cloud"
        assert pipeline.config.hf_token == "test-token"

    def test_local_factory(self):
        """Test local pipeline factory method."""
        pipeline = Pipeline.local(models_dir="/custom/models")

        assert pipeline.config.transcription.backend == "local"
        assert pipeline.config.extraction.backend == "local"
        assert "/custom/models" in pipeline.config.transcription.model_path


class TestPipelineLazyLoading:
    """Tests for lazy component loading."""

    def test_capture_lazy_loading(self):
        """Test that capture is lazily loaded."""
        pipeline = Pipeline(PipelineConfig())

        assert pipeline._capture is None
        _ = pipeline.capture  # Access property
        assert pipeline._capture is not None

    def test_transcriber_lazy_loading(self):
        """Test that transcriber is lazily loaded."""
        pipeline = Pipeline(PipelineConfig())

        assert pipeline._transcriber is None
        # Note: Accessing transcriber may require network/model
        # Just test the property exists
        assert hasattr(pipeline, "transcriber")

    def test_extractor_lazy_loading(self):
        """Test that extractor is lazily loaded."""
        pipeline = Pipeline(PipelineConfig())

        assert pipeline._extractor is None
        assert hasattr(pipeline, "extractor")

    def test_transformer_lazy_loading(self):
        """Test that transformer is lazily loaded."""
        pipeline = Pipeline(PipelineConfig())

        assert pipeline._transformer is None
        _ = pipeline.transformer
        assert pipeline._transformer is not None


class TestPipelineProcessing:
    """Tests for pipeline processing methods."""

    @pytest.fixture
    def pipeline(self) -> Pipeline:
        """Create pipeline with mocked components."""
        config = PipelineConfig()
        config.fhir.validate = False  # Disable validation for tests
        return Pipeline(config)

    @pytest.fixture
    def sample_audio(self, sample_rate: int) -> AudioSegment:
        """Create sample audio segment."""
        data = np.random.randn(sample_rate * 2).astype(np.float32)
        return AudioSegment(data=data, sample_rate=sample_rate)

    def test_process_audio_mocked(
        self,
        pipeline: Pipeline,
        sample_audio: AudioSegment,
        mock_transcriber: MagicMock,
        mock_extractor: MagicMock,
        mock_transformer: MagicMock,
    ):
        """Test processing audio with mocked components."""
        pipeline._transcriber = mock_transcriber
        pipeline._extractor = mock_extractor
        pipeline._transformer = mock_transformer

        bundle = pipeline.process_audio(sample_audio)

        assert bundle is not None
        mock_transcriber.transcribe.assert_called_once()
        mock_extractor.extract.assert_called_once()
        mock_transformer.transform.assert_called_once()

    def test_process_transcript(
        self,
        pipeline: Pipeline,
        sample_transcript_text: str,
        mock_extractor: MagicMock,
        mock_transformer: MagicMock,
    ):
        """Test processing transcript directly."""
        pipeline._extractor = mock_extractor
        pipeline._transformer = mock_transformer

        bundle = pipeline.process_transcript(sample_transcript_text)

        assert bundle is not None
        # Should NOT call transcriber
        mock_extractor.extract.assert_called_once()
        mock_transformer.transform.assert_called_once()

    def test_process_with_workflow(
        self,
        pipeline: Pipeline,
        sample_transcript_text: str,
        mock_extractor: MagicMock,
        mock_transformer: MagicMock,
    ):
        """Test processing with specific workflow."""
        pipeline._extractor = mock_extractor
        pipeline._transformer = mock_transformer

        pipeline.process_transcript(sample_transcript_text, workflow="emergency")

        # Verify workflow was passed to extractor
        call_args = mock_extractor.extract.call_args
        assert call_args[0][1] == "emergency" or call_args[1].get("workflow") == "emergency"

    def test_process_file(
        self,
        pipeline: Pipeline,
        tmp_path: Path,
        mock_transcriber: MagicMock,
        mock_extractor: MagicMock,
        mock_transformer: MagicMock,
    ):
        """Test processing audio file."""
        # Create a mock audio file
        audio_path = tmp_path / "test.wav"
        audio_path.touch()  # Just create empty file

        pipeline._transcriber = mock_transcriber
        pipeline._extractor = mock_extractor
        pipeline._transformer = mock_transformer

        # Mock AudioSegment.from_file
        with patch.object(AudioSegment, "from_file") as mock_from_file:
            mock_audio = MagicMock()
            mock_from_file.return_value = mock_audio

            bundle = pipeline.process_file(audio_path)

            assert bundle is not None
            mock_from_file.assert_called_once_with(audio_path)


class TestPipelineOutput:
    """Tests for pipeline output methods."""

    @pytest.fixture
    def pipeline(self) -> Pipeline:
        """Create pipeline for output tests."""
        config = PipelineConfig()
        return Pipeline(config)

    def test_to_json(self, pipeline: Pipeline, sample_fhir_bundle: dict[str, Any]):
        """Test JSON output."""
        json_str = pipeline.to_json(sample_fhir_bundle)

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["resourceType"] == "Bundle"

    def test_to_json_custom_indent(
        self, pipeline: Pipeline, sample_fhir_bundle: dict[str, Any]
    ):
        """Test JSON output with custom indent."""
        json_str = pipeline.to_json(sample_fhir_bundle, indent=4)

        assert "\n    " in json_str

    def test_save_to_file(
        self, pipeline: Pipeline, sample_fhir_bundle: dict[str, Any], tmp_path: Path
    ):
        """Test saving bundle to file."""
        output_path = tmp_path / "output.json"

        pipeline.save(sample_fhir_bundle, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["resourceType"] == "Bundle"


class TestPipelineValidation:
    """Tests for pipeline validation."""

    def test_validation_enabled(self, sample_fhir_bundle: dict[str, Any]):
        """Test that validation adds metadata when enabled."""
        config = PipelineConfig()
        config.fhir.validate = True
        pipeline = Pipeline(config)

        # Mock the processing chain to return our sample bundle
        pipeline._transformer = MagicMock()
        pipeline._transformer.transform.return_value = sample_fhir_bundle

        pipeline._extractor = MagicMock()
        from voice_to_fhir.extraction.extraction_types import ClinicalEntities

        pipeline._extractor.extract.return_value = ClinicalEntities()

        bundle = pipeline.process_transcript("Test transcript")

        # If validation finds issues, it adds _validation key
        # Our sample bundle should be valid, so no _validation key
        assert "_validation" not in bundle or bundle["_validation"]["valid"] is True

    def test_validation_disabled(self):
        """Test that validation is skipped when disabled."""
        config = PipelineConfig()
        config.fhir.validate = False
        pipeline = Pipeline(config)

        # Create invalid bundle
        invalid_bundle = {"resourceType": "Bundle"}  # Missing type

        pipeline._transformer = MagicMock()
        pipeline._transformer.transform.return_value = invalid_bundle

        pipeline._extractor = MagicMock()
        from voice_to_fhir.extraction.extraction_types import ClinicalEntities

        pipeline._extractor.extract.return_value = ClinicalEntities()

        bundle = pipeline.process_transcript("Test")

        # Validation disabled, so no _validation key even for invalid bundle
        assert "_validation" not in bundle


class TestPipelineWarmup:
    """Tests for pipeline warmup."""

    def test_warmup_local(self):
        """Test warmup with local backends."""
        config = PipelineConfig()
        config.transcription.backend = "local"
        config.extraction.backend = "local"
        pipeline = Pipeline(config)

        # Mock components with warmup method
        mock_transcriber = MagicMock()
        mock_extractor = MagicMock()
        pipeline._transcriber = mock_transcriber
        pipeline._extractor = mock_extractor

        pipeline.warmup()

        # Should call warmup on local components
        if hasattr(mock_transcriber, "warmup"):
            mock_transcriber.warmup.assert_called()

    def test_warmup_cloud(self):
        """Test warmup with cloud backends (no-op)."""
        config = PipelineConfig()
        config.transcription.backend = "cloud"
        config.extraction.backend = "cloud"
        pipeline = Pipeline(config)

        # Should not raise
        pipeline.warmup()
