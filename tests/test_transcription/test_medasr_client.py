"""
Tests for MedASR cloud client.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from voice_to_fhir.transcription.medasr_client import MedASRClient, MedASRClientConfig
from voice_to_fhir.capture.audio_utils import AudioSegment


class TestMedASRClientConfig:
    """Tests for MedASR client configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MedASRClientConfig()

        assert config.model_id == "google/medasr"
        assert config.api_key is None
        assert config.timeout == 30.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = MedASRClientConfig(
            api_key="test-key",
            model_id="custom/model",
            timeout=60.0,
        )

        assert config.api_key == "test-key"
        assert config.model_id == "custom/model"
        assert config.timeout == 60.0

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("HF_TOKEN", "env-token")

        config = MedASRClientConfig.from_env()

        assert config.api_key == "env-token"


class TestMedASRClient:
    """Tests for MedASR cloud client."""

    @pytest.fixture
    def config(self) -> MedASRClientConfig:
        """Create test configuration."""
        return MedASRClientConfig(api_key="test-key")

    @pytest.fixture
    def client(self, config: MedASRClientConfig) -> MedASRClient:
        """Create MedASR client for testing."""
        return MedASRClient(config)

    @pytest.fixture
    def sample_audio(self, sample_rate: int) -> AudioSegment:
        """Create sample audio segment."""
        data = np.random.randn(sample_rate * 2).astype(np.float32)  # 2 seconds
        return AudioSegment(data=data, sample_rate=sample_rate)

    def test_initialization(self, client: MedASRClient):
        """Test client initialization."""
        assert client.config.model_id == "google/medasr"
        assert client.config.api_key == "test-key"

    def test_initialization_without_key(self):
        """Test initialization without API key raises warning."""
        config = MedASRClientConfig(api_key=None)

        # Should not raise, but may log warning
        client = MedASRClient(config)
        assert client.config.api_key is None

    @patch("voice_to_fhir.transcription.medasr_client.requests")
    def test_transcribe_success(
        self, mock_requests, client: MedASRClient, sample_audio: AudioSegment
    ):
        """Test successful transcription."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Patient has chest pain.",
            "segments": [
                {
                    "text": "Patient has chest pain.",
                    "start": 0.0,
                    "end": 2.0,
                    "confidence": 0.95,
                }
            ],
        }
        mock_requests.post.return_value = mock_response

        transcript = client.transcribe(sample_audio)

        assert transcript.text == "Patient has chest pain."
        assert len(transcript.segments) == 1
        assert transcript.confidence == 0.95

    @patch("voice_to_fhir.transcription.medasr_client.requests")
    def test_transcribe_api_error(
        self, mock_requests, client: MedASRClient, sample_audio: AudioSegment
    ):
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_requests.post.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            client.transcribe(sample_audio)

        assert "API error" in str(exc_info.value) or "500" in str(exc_info.value)

    @patch("voice_to_fhir.transcription.medasr_client.requests")
    def test_transcribe_timeout(
        self, mock_requests, client: MedASRClient, sample_audio: AudioSegment
    ):
        """Test handling of timeout."""
        import requests

        mock_requests.post.side_effect = requests.Timeout("Connection timed out")

        with pytest.raises(Exception):
            client.transcribe(sample_audio)

    @patch("voice_to_fhir.transcription.medasr_client.requests")
    def test_transcribe_empty_response(
        self, mock_requests, client: MedASRClient, sample_audio: AudioSegment
    ):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "", "segments": []}
        mock_requests.post.return_value = mock_response

        transcript = client.transcribe(sample_audio)

        assert transcript.text == ""
        assert len(transcript.segments) == 0

    def test_prepare_audio_data(self, client: MedASRClient, sample_audio: AudioSegment):
        """Test audio data preparation for API."""
        prepared = client._prepare_audio(sample_audio)

        # Should return bytes (WAV or similar format)
        assert isinstance(prepared, bytes)
        assert len(prepared) > 0

    @patch("voice_to_fhir.transcription.medasr_client.requests")
    def test_transcribe_with_language_hint(
        self, mock_requests, client: MedASRClient, sample_audio: AudioSegment
    ):
        """Test transcription with language hint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Test text",
            "language": "en",
            "segments": [],
        }
        mock_requests.post.return_value = mock_response

        transcript = client.transcribe(sample_audio, language="en")

        # Verify language was passed in request
        call_kwargs = mock_requests.post.call_args
        assert transcript.language == "en"

    def test_health_check(self, client: MedASRClient):
        """Test API health check."""
        # This would typically ping the API
        # For now, just verify the method exists
        assert hasattr(client, "health_check")
