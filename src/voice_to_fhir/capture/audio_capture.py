"""
Audio Capture

Cross-platform audio capture with streaming support.
"""

from dataclasses import dataclass
from typing import Iterator, Callable
import threading
import queue

import numpy as np

from voice_to_fhir.capture.audio_utils import AudioSegment, AudioChunk
from voice_to_fhir.capture.vad import VoiceActivityDetector, VADConfig


@dataclass
class CaptureConfig:
    """Configuration for audio capture."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 100
    dtype: str = "float32"
    device: int | str | None = None  # None = default device

    # VAD settings
    vad_enabled: bool = True
    vad_config: VADConfig | None = None


class AudioCapture:
    """Cross-platform audio capture with VAD support."""

    def __init__(self, config: CaptureConfig | None = None):
        """Initialize audio capture."""
        self.config = config or CaptureConfig()
        self._stream = None
        self._is_capturing = False
        self._audio_queue: queue.Queue[AudioChunk] = queue.Queue()
        self._recorded_chunks: list[AudioChunk] = []
        self._sequence_number = 0

        if self.config.vad_enabled:
            self._vad = VoiceActivityDetector(self.config.vad_config)
        else:
            self._vad = None

    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk."""
        return int(self.config.sample_rate * self.config.chunk_duration_ms / 1000)

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: dict, status
    ) -> None:
        """Callback for audio stream."""
        if status:
            print(f"Audio capture status: {status}")

        # Convert to mono if needed
        if len(indata.shape) > 1:
            data = np.mean(indata, axis=1)
        else:
            data = indata.flatten()

        chunk = AudioChunk(
            data=data.astype(np.float32),
            sample_rate=self.config.sample_rate,
            timestamp_ms=self._sequence_number * self.config.chunk_duration_ms,
            sequence_number=self._sequence_number,
        )

        # Apply VAD if enabled
        if self._vad:
            chunk.is_speech = self._vad.is_speech(chunk)

        self._sequence_number += 1
        self._audio_queue.put(chunk)
        self._recorded_chunks.append(chunk)

    def start(self) -> None:
        """Start audio capture."""
        import sounddevice as sd

        if self._is_capturing:
            return

        self._is_capturing = True
        self._sequence_number = 0
        self._recorded_chunks = []

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.chunk_samples,
            device=self.config.device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> AudioSegment:
        """Stop capture and return recorded audio."""
        if not self._is_capturing:
            return AudioSegment(
                data=np.array([], dtype=np.float32),
                sample_rate=self.config.sample_rate,
            )

        self._is_capturing = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Combine all chunks into single segment
        if self._recorded_chunks:
            all_data = np.concatenate([c.data for c in self._recorded_chunks])
        else:
            all_data = np.array([], dtype=np.float32)

        return AudioSegment(
            data=all_data,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
        )

    def stream(self) -> Iterator[AudioChunk]:
        """Stream audio chunks as they are captured."""
        while self._is_capturing or not self._audio_queue.empty():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                yield chunk
            except queue.Empty:
                continue

    def capture_until_silence(
        self, max_duration_seconds: float = 30.0
    ) -> AudioSegment:
        """Capture audio until silence is detected."""
        self.start()

        chunks: list[AudioChunk] = []
        silence_duration_ms = 0
        total_duration_ms = 0
        max_duration_ms = max_duration_seconds * 1000
        silence_threshold_ms = 1500  # End after 1.5s of silence

        try:
            for chunk in self.stream():
                chunks.append(chunk)
                total_duration_ms += chunk.duration_ms

                if chunk.is_speech:
                    silence_duration_ms = 0
                else:
                    silence_duration_ms += chunk.duration_ms

                # Check stop conditions
                if silence_duration_ms >= silence_threshold_ms and chunks:
                    break
                if total_duration_ms >= max_duration_ms:
                    break
        finally:
            self.stop()

        # Combine chunks
        if chunks:
            all_data = np.concatenate([c.data for c in chunks])
        else:
            all_data = np.array([], dtype=np.float32)

        return AudioSegment(
            data=all_data,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
        )

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio input devices."""
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                input_devices.append(
                    {
                        "index": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "sample_rate": device["default_samplerate"],
                    }
                )

        return input_devices
