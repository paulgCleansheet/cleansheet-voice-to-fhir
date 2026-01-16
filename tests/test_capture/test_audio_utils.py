"""
Tests for audio utilities.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

import numpy as np
import pytest

from voice_to_fhir.capture.audio_utils import AudioSegment, AudioChunk


class TestAudioSegment:
    """Tests for AudioSegment class."""

    def test_create_from_array(self, sample_audio_data: np.ndarray, sample_rate: int):
        """Test creating AudioSegment from numpy array."""
        segment = AudioSegment(
            data=sample_audio_data,
            sample_rate=sample_rate,
        )

        assert segment.sample_rate == sample_rate
        assert len(segment.data) == len(sample_audio_data)
        assert segment.channels == 1

    def test_duration_calculation(self, sample_rate: int):
        """Test duration property calculation."""
        duration_seconds = 2.5
        samples = int(sample_rate * duration_seconds)
        data = np.zeros(samples, dtype=np.float32)

        segment = AudioSegment(data=data, sample_rate=sample_rate)

        assert abs(segment.duration - duration_seconds) < 0.001

    def test_empty_segment(self, sample_rate: int):
        """Test creating empty audio segment."""
        segment = AudioSegment.empty(sample_rate)

        assert segment.sample_rate == sample_rate
        assert len(segment.data) == 0
        assert segment.duration == 0.0

    def test_concatenate_segments(self, sample_rate: int):
        """Test concatenating multiple audio segments."""
        data1 = np.ones(sample_rate, dtype=np.float32)  # 1 second
        data2 = np.ones(sample_rate, dtype=np.float32) * 2  # 1 second

        segment1 = AudioSegment(data=data1, sample_rate=sample_rate)
        segment2 = AudioSegment(data=data2, sample_rate=sample_rate)

        combined = AudioSegment.concatenate([segment1, segment2])

        assert combined.duration == 2.0
        assert len(combined.data) == 2 * sample_rate

    def test_concatenate_empty_list(self, sample_rate: int):
        """Test concatenating empty list returns empty segment."""
        combined = AudioSegment.concatenate([])

        assert combined.duration == 0.0

    def test_slice_segment(self, sample_rate: int):
        """Test slicing audio segment by time."""
        duration = 3.0
        data = np.arange(int(sample_rate * duration), dtype=np.float32)
        segment = AudioSegment(data=data, sample_rate=sample_rate)

        # Slice from 1.0 to 2.0 seconds
        sliced = segment.slice(1.0, 2.0)

        assert abs(sliced.duration - 1.0) < 0.001
        assert sliced.sample_rate == sample_rate

    def test_resample(self, sample_audio_data: np.ndarray):
        """Test resampling audio to different sample rate."""
        original_rate = 16000
        target_rate = 8000

        segment = AudioSegment(data=sample_audio_data, sample_rate=original_rate)
        resampled = segment.resample(target_rate)

        assert resampled.sample_rate == target_rate
        # Duration should be approximately the same
        assert abs(resampled.duration - segment.duration) < 0.01

    def test_to_mono(self, sample_rate: int):
        """Test converting stereo to mono."""
        # Create stereo data
        stereo_data = np.random.randn(sample_rate, 2).astype(np.float32)
        segment = AudioSegment(data=stereo_data, sample_rate=sample_rate, channels=2)

        mono = segment.to_mono()

        assert mono.channels == 1
        assert len(mono.data.shape) == 1

    def test_normalize(self, sample_rate: int):
        """Test audio normalization."""
        # Create audio with peak at 0.5
        data = np.random.randn(sample_rate).astype(np.float32) * 0.5
        segment = AudioSegment(data=data, sample_rate=sample_rate)

        normalized = segment.normalize()

        # Peak should be close to 1.0
        assert abs(np.max(np.abs(normalized.data)) - 1.0) < 0.01


class TestAudioChunk:
    """Tests for AudioChunk class."""

    def test_create_chunk(self, sample_rate: int):
        """Test creating AudioChunk."""
        data = np.random.randn(1600).astype(np.float32)  # 100ms at 16kHz

        chunk = AudioChunk(
            data=data,
            sample_rate=sample_rate,
            timestamp=0.0,
            is_speech=True,
        )

        assert chunk.sample_rate == sample_rate
        assert chunk.timestamp == 0.0
        assert chunk.is_speech is True

    def test_chunk_duration(self, sample_rate: int):
        """Test chunk duration calculation."""
        chunk_ms = 100
        samples = int(sample_rate * chunk_ms / 1000)
        data = np.zeros(samples, dtype=np.float32)

        chunk = AudioChunk(
            data=data,
            sample_rate=sample_rate,
            timestamp=0.0,
        )

        assert abs(chunk.duration_ms - chunk_ms) < 1

    def test_chunk_energy(self, sample_rate: int):
        """Test chunk energy calculation."""
        # Silent chunk
        silent_data = np.zeros(1600, dtype=np.float32)
        silent_chunk = AudioChunk(data=silent_data, sample_rate=sample_rate, timestamp=0.0)

        # Loud chunk
        loud_data = np.ones(1600, dtype=np.float32)
        loud_chunk = AudioChunk(data=loud_data, sample_rate=sample_rate, timestamp=0.0)

        assert silent_chunk.energy < loud_chunk.energy
        assert silent_chunk.energy == 0.0
