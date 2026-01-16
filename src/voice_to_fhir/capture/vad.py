"""
Voice Activity Detection (VAD)

Detects speech segments in audio streams.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from voice_to_fhir.capture.audio_utils import AudioChunk


@dataclass
class VADConfig:
    """Configuration for voice activity detection."""

    # Energy-based VAD settings
    energy_threshold: float = 0.01
    min_speech_duration_ms: float = 250
    min_silence_duration_ms: float = 500

    # WebRTC VAD settings (if available)
    webrtc_mode: int = 3  # 0-3, higher = more aggressive
    use_webrtc: bool = True

    # Frame settings
    frame_duration_ms: int = 30  # 10, 20, or 30 ms for WebRTC VAD


class VoiceActivityDetector:
    """Detects voice activity in audio streams."""

    def __init__(self, config: VADConfig | None = None):
        """Initialize VAD with configuration."""
        self.config = config or VADConfig()
        self._webrtc_vad = None
        self._init_webrtc_vad()

    def _init_webrtc_vad(self) -> None:
        """Initialize WebRTC VAD if available."""
        if not self.config.use_webrtc:
            return

        try:
            import webrtcvad

            self._webrtc_vad = webrtcvad.Vad(self.config.webrtc_mode)
        except ImportError:
            self._webrtc_vad = None

    def is_speech(self, chunk: AudioChunk) -> bool:
        """Determine if an audio chunk contains speech."""
        if self._webrtc_vad is not None:
            return self._is_speech_webrtc(chunk)
        return self._is_speech_energy(chunk)

    def _is_speech_energy(self, chunk: AudioChunk) -> bool:
        """Simple energy-based VAD."""
        energy = np.sqrt(np.mean(chunk.data**2))
        return energy > self.config.energy_threshold

    def _is_speech_webrtc(self, chunk: AudioChunk) -> bool:
        """WebRTC-based VAD (more accurate)."""
        # WebRTC VAD requires 16-bit PCM at specific sample rates
        if chunk.sample_rate not in (8000, 16000, 32000, 48000):
            # Fall back to energy-based
            return self._is_speech_energy(chunk)

        # Convert to 16-bit PCM
        audio_int16 = (chunk.data * 32767).astype(np.int16)

        # WebRTC VAD expects specific frame sizes
        frame_samples = int(chunk.sample_rate * self.config.frame_duration_ms / 1000)

        # Check if any frame contains speech
        for i in range(0, len(audio_int16) - frame_samples, frame_samples):
            frame = audio_int16[i : i + frame_samples]
            if self._webrtc_vad.is_speech(frame.tobytes(), chunk.sample_rate):
                return True

        return False

    def process_stream(
        self, chunks: Iterator[AudioChunk]
    ) -> Iterator[tuple[AudioChunk, bool]]:
        """Process a stream of audio chunks, yielding (chunk, is_speech) pairs."""
        for chunk in chunks:
            is_speech = self.is_speech(chunk)
            yield chunk, is_speech

    def extract_speech_segments(
        self, chunks: Iterator[AudioChunk]
    ) -> Iterator[list[AudioChunk]]:
        """Extract continuous speech segments from audio stream."""
        current_segment: list[AudioChunk] = []
        silence_duration_ms = 0

        for chunk, is_speech in self.process_stream(chunks):
            if is_speech:
                current_segment.append(chunk)
                silence_duration_ms = 0
            else:
                silence_duration_ms += chunk.duration_ms

                if current_segment:
                    # Check if silence is long enough to end segment
                    if silence_duration_ms >= self.config.min_silence_duration_ms:
                        # Check if segment is long enough
                        total_duration = sum(c.duration_ms for c in current_segment)
                        if total_duration >= self.config.min_speech_duration_ms:
                            yield current_segment
                        current_segment = []
                        silence_duration_ms = 0

        # Yield final segment if exists
        if current_segment:
            total_duration = sum(c.duration_ms for c in current_segment)
            if total_duration >= self.config.min_speech_duration_ms:
                yield current_segment
