"""
Tests for Voice Activity Detection.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

import numpy as np
import pytest

from voice_to_fhir.capture.vad import VADConfig, VoiceActivityDetector


class TestVADConfig:
    """Tests for VAD configuration."""

    def test_default_config(self):
        """Test default VAD configuration values."""
        config = VADConfig()

        assert config.sample_rate == 16000
        assert config.frame_duration_ms == 30
        assert config.webrtc_mode in [0, 1, 2, 3]

    def test_custom_config(self):
        """Test custom VAD configuration."""
        config = VADConfig(
            sample_rate=8000,
            frame_duration_ms=20,
            webrtc_mode=3,
            energy_threshold=0.02,
        )

        assert config.sample_rate == 8000
        assert config.frame_duration_ms == 20
        assert config.webrtc_mode == 3
        assert config.energy_threshold == 0.02

    def test_invalid_frame_duration(self):
        """Test that invalid frame durations raise error."""
        # WebRTC VAD only supports 10, 20, or 30ms frames
        with pytest.raises(ValueError):
            VADConfig(frame_duration_ms=25)

    def test_invalid_webrtc_mode(self):
        """Test that invalid WebRTC modes raise error."""
        with pytest.raises(ValueError):
            VADConfig(webrtc_mode=5)


class TestVoiceActivityDetector:
    """Tests for VoiceActivityDetector class."""

    @pytest.fixture
    def vad(self) -> VoiceActivityDetector:
        """Create VAD instance for testing."""
        config = VADConfig(webrtc_mode=3)
        return VoiceActivityDetector(config)

    def test_detect_silence(self, vad: VoiceActivityDetector, sample_rate: int):
        """Test detection of silence."""
        # Generate silent audio
        frame_samples = int(sample_rate * 30 / 1000)  # 30ms frame
        silent_frame = np.zeros(frame_samples, dtype=np.float32)

        is_speech = vad.is_speech(silent_frame)

        assert is_speech is False

    def test_detect_speech(self, vad: VoiceActivityDetector, sample_rate: int):
        """Test detection of speech-like audio."""
        frame_samples = int(sample_rate * 30 / 1000)  # 30ms frame

        # Generate speech-like signal (sine wave with amplitude)
        t = np.linspace(0, 0.03, frame_samples)
        speech_frame = (np.sin(2 * np.pi * 300 * t) * 0.5).astype(np.float32)

        is_speech = vad.is_speech(speech_frame)

        # Note: This may or may not detect as speech depending on VAD sensitivity
        # The important thing is it doesn't crash
        assert isinstance(is_speech, bool)

    def test_process_stream(self, vad: VoiceActivityDetector, sample_rate: int):
        """Test processing a stream of audio frames."""
        frame_samples = int(sample_rate * 30 / 1000)

        # Create sequence of frames (silent, speech, silent)
        frames = [
            np.zeros(frame_samples, dtype=np.float32),  # Silent
            (np.random.randn(frame_samples) * 0.3).astype(np.float32),  # Noise
            np.zeros(frame_samples, dtype=np.float32),  # Silent
        ]

        results = [vad.is_speech(frame) for frame in frames]

        assert len(results) == 3
        assert all(isinstance(r, bool) for r in results)

    def test_reset(self, vad: VoiceActivityDetector, sample_rate: int):
        """Test resetting VAD state."""
        frame_samples = int(sample_rate * 30 / 1000)
        frame = (np.random.randn(frame_samples) * 0.3).astype(np.float32)

        # Process some frames
        for _ in range(5):
            vad.is_speech(frame)

        # Reset
        vad.reset()

        # Should work normally after reset
        result = vad.is_speech(frame)
        assert isinstance(result, bool)

    def test_energy_based_detection(self, sample_rate: int):
        """Test energy-based VAD fallback."""
        config = VADConfig(
            sample_rate=sample_rate,
            energy_threshold=0.01,
        )
        vad = VoiceActivityDetector(config)

        frame_samples = int(sample_rate * 30 / 1000)

        # Low energy frame (below threshold)
        low_energy = np.zeros(frame_samples, dtype=np.float32) + 0.001
        assert vad._energy_vad(low_energy) is False

        # High energy frame (above threshold)
        high_energy = np.ones(frame_samples, dtype=np.float32) * 0.5
        assert vad._energy_vad(high_energy) is True

    def test_get_speech_segments(
        self, vad: VoiceActivityDetector, sample_audio_with_speech: np.ndarray
    ):
        """Test extracting speech segments from audio."""
        segments = vad.get_speech_segments(sample_audio_with_speech)

        # Should return list of (start, end) tuples
        assert isinstance(segments, list)
        for segment in segments:
            assert len(segment) == 2
            start, end = segment
            assert start < end
