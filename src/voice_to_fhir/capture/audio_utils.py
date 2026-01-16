"""
Audio Utilities

Data types and utility functions for audio processing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
import io

import numpy as np


@dataclass
class AudioChunk:
    """A chunk of audio data for streaming processing."""

    data: np.ndarray
    sample_rate: int
    timestamp_ms: float
    is_speech: bool = False
    sequence_number: int = 0

    @property
    def duration_ms(self) -> float:
        """Duration of this chunk in milliseconds."""
        return (len(self.data) / self.sample_rate) * 1000


@dataclass
class AudioSegment:
    """A complete audio segment (e.g., a full utterance)."""

    data: np.ndarray
    sample_rate: int
    channels: int = 1
    metadata: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return len(self.data) / self.sample_rate

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_seconds * 1000

    @classmethod
    def from_file(cls, filepath: str | Path) -> "AudioSegment":
        """Load audio segment from file."""
        import soundfile as sf

        data, sample_rate = sf.read(filepath, dtype="float32")

        # Convert stereo to mono if needed
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        return cls(
            data=data,
            sample_rate=sample_rate,
            channels=1,
            metadata={"source_file": str(filepath)},
        )

    @classmethod
    def from_bytes(
        cls, audio_bytes: bytes, sample_rate: int = 16000, format: str = "wav"
    ) -> "AudioSegment":
        """Load audio segment from bytes."""
        import soundfile as sf

        with io.BytesIO(audio_bytes) as buffer:
            data, sr = sf.read(buffer, dtype="float32")

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        return cls(data=data, sample_rate=sr, channels=1)

    def to_bytes(self, format: str = "wav") -> bytes:
        """Export audio segment to bytes."""
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, self.data, self.sample_rate, format=format)
        buffer.seek(0)
        return buffer.read()

    def to_file(self, filepath: str | Path, format: str = "wav") -> None:
        """Save audio segment to file."""
        import soundfile as sf

        sf.write(filepath, self.data, self.sample_rate, format=format)

    def resample(self, target_sample_rate: int) -> "AudioSegment":
        """Resample audio to target sample rate."""
        if self.sample_rate == target_sample_rate:
            return self

        from scipy import signal

        num_samples = int(len(self.data) * target_sample_rate / self.sample_rate)
        resampled = signal.resample(self.data, num_samples)

        return AudioSegment(
            data=resampled.astype(np.float32),
            sample_rate=target_sample_rate,
            channels=self.channels,
            metadata={**self.metadata, "resampled_from": self.sample_rate},
        )

    def normalize(self, target_db: float = -20.0) -> "AudioSegment":
        """Normalize audio to target dB level."""
        rms = np.sqrt(np.mean(self.data**2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            normalized = self.data * (target_rms / rms)
            # Clip to prevent clipping
            normalized = np.clip(normalized, -1.0, 1.0)
        else:
            normalized = self.data

        return AudioSegment(
            data=normalized.astype(np.float32),
            sample_rate=self.sample_rate,
            channels=self.channels,
            metadata={**self.metadata, "normalized_to_db": target_db},
        )
