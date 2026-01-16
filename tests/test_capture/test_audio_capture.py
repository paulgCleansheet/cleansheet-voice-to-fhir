"""
Tests for audio capture functionality.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from voice_to_fhir.capture.audio_capture import AudioCapture, CaptureConfig
from voice_to_fhir.capture.vad import VADConfig


class TestCaptureConfig:
    """Tests for CaptureConfig class."""

    def test_default_config(self):
        """Test default capture configuration."""
        config = CaptureConfig()

        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_duration_ms == 100
        assert config.vad_enabled is True

    def test_custom_config(self):
        """Test custom capture configuration."""
        vad_config = VADConfig(webrtc_mode=2)
        config = CaptureConfig(
            sample_rate=8000,
            channels=2,
            chunk_duration_ms=50,
            vad_enabled=False,
            vad_config=vad_config,
        )

        assert config.sample_rate == 8000
        assert config.channels == 2
        assert config.chunk_duration_ms == 50
        assert config.vad_enabled is False
        assert config.vad_config.webrtc_mode == 2

    def test_chunk_samples_calculation(self):
        """Test calculation of samples per chunk."""
        config = CaptureConfig(sample_rate=16000, chunk_duration_ms=100)

        # 100ms at 16kHz = 1600 samples
        assert config.chunk_samples == 1600


class TestAudioCapture:
    """Tests for AudioCapture class."""

    @pytest.fixture
    def capture(self) -> AudioCapture:
        """Create AudioCapture instance for testing."""
        config = CaptureConfig(vad_enabled=False)
        return AudioCapture(config)

    @pytest.fixture
    def capture_with_vad(self) -> AudioCapture:
        """Create AudioCapture instance with VAD enabled."""
        config = CaptureConfig(vad_enabled=True)
        return AudioCapture(config)

    def test_initialization(self, capture: AudioCapture):
        """Test AudioCapture initialization."""
        assert capture.config.sample_rate == 16000
        assert capture.is_capturing is False

    def test_list_devices(self):
        """Test listing audio devices."""
        devices = AudioCapture.list_devices()

        # Should return a list (may be empty if no devices)
        assert isinstance(devices, list)
        for device in devices:
            assert "index" in device
            assert "name" in device

    @patch("voice_to_fhir.capture.audio_capture.sd")
    def test_start_capture(self, mock_sd, capture: AudioCapture):
        """Test starting audio capture."""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        capture.start()

        assert capture.is_capturing is True
        mock_stream.start.assert_called_once()

    @patch("voice_to_fhir.capture.audio_capture.sd")
    def test_stop_capture(self, mock_sd, capture: AudioCapture):
        """Test stopping audio capture."""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        capture.start()
        capture.stop()

        assert capture.is_capturing is False
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    @patch("voice_to_fhir.capture.audio_capture.sd")
    def test_double_start(self, mock_sd, capture: AudioCapture):
        """Test that double start doesn't create multiple streams."""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        capture.start()
        capture.start()  # Second start should be no-op

        assert mock_sd.InputStream.call_count == 1

    @patch("voice_to_fhir.capture.audio_capture.sd")
    def test_stop_without_start(self, mock_sd, capture: AudioCapture):
        """Test that stop without start is safe."""
        capture.stop()  # Should not raise

        assert capture.is_capturing is False

    def test_callback_processing(self, capture: AudioCapture):
        """Test audio callback processing."""
        # Simulate audio data from callback
        indata = np.random.randn(1600, 1).astype(np.float32)

        # Process through callback
        capture._audio_callback(indata, frames=1600, time_info=None, status=None)

        # Should have data in buffer
        assert not capture._buffer.empty()

    def test_callback_with_vad(self, capture_with_vad: AudioCapture):
        """Test audio callback with VAD enabled."""
        # Create speech-like data
        t = np.linspace(0, 0.1, 1600)
        indata = (np.sin(2 * np.pi * 300 * t) * 0.5).reshape(-1, 1).astype(np.float32)

        capture_with_vad._audio_callback(indata, frames=1600, time_info=None, status=None)

        # Should process without error
        # VAD will determine if this is speech or not

    def test_get_chunk(self, capture: AudioCapture):
        """Test getting audio chunk from buffer."""
        # Add data to buffer
        indata = np.random.randn(1600, 1).astype(np.float32)
        capture._audio_callback(indata, frames=1600, time_info=None, status=None)

        # Get chunk
        chunk = capture.get_chunk(timeout=1.0)

        assert chunk is not None
        assert len(chunk.data) == 1600

    def test_get_chunk_timeout(self, capture: AudioCapture):
        """Test timeout when no data available."""
        # Don't add any data
        chunk = capture.get_chunk(timeout=0.1)

        assert chunk is None

    @patch("voice_to_fhir.capture.audio_capture.sd")
    def test_capture_context_manager(self, mock_sd):
        """Test using AudioCapture as context manager."""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        config = CaptureConfig()
        with AudioCapture(config) as capture:
            assert capture.is_capturing is True

        assert capture.is_capturing is False
        mock_stream.stop.assert_called()

    def test_capture_until_silence(self, capture: AudioCapture):
        """Test capture_until_silence with mock data."""
        # This is more of an integration test
        # For unit testing, we'd mock the internal components
        pass  # Skip for now, requires more complex mocking

    def test_stream_generator(self, capture: AudioCapture):
        """Test stream generator."""
        # Add some chunks to buffer
        for i in range(3):
            indata = np.random.randn(1600, 1).astype(np.float32)
            capture._audio_callback(indata, frames=1600, time_info=None, status=None)

        # Get chunks from stream
        chunks = []
        for chunk in capture.stream():
            chunks.append(chunk)
            if len(chunks) >= 3:
                break

        assert len(chunks) == 3
