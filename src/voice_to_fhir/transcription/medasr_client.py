"""
MedASR Cloud Client

HuggingFace Inference API client for MedASR transcription.
"""

from dataclasses import dataclass
import os
from typing import Iterator

from voice_to_fhir.capture.audio_utils import AudioSegment, AudioChunk
from voice_to_fhir.transcription.transcript_types import Transcript, TranscriptChunk


@dataclass
class MedASRClientConfig:
    """Configuration for MedASR cloud client."""

    api_key: str | None = None
    model_id: str = "google/medasr"
    api_url: str = "https://api-inference.huggingface.co/models"
    timeout_seconds: float = 30.0
    max_retries: int = 3


class MedASRClient:
    """HuggingFace Inference API client for MedASR."""

    def __init__(self, config: MedASRClientConfig | None = None):
        """Initialize MedASR client."""
        self.config = config or MedASRClientConfig()

        # Get API key from config or environment
        self.api_key = self.config.api_key or os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError(
                "HuggingFace API key required. "
                "Set HF_TOKEN environment variable or pass api_key in config."
            )

    @property
    def _headers(self) -> dict[str, str]:
        """Request headers."""
        return {"Authorization": f"Bearer {self.api_key}"}

    @property
    def _endpoint(self) -> str:
        """API endpoint URL."""
        return f"{self.config.api_url}/{self.config.model_id}"

    def transcribe(self, audio: AudioSegment) -> Transcript:
        """Transcribe audio to text."""
        import requests

        # Ensure correct sample rate (MedASR expects 16kHz)
        if audio.sample_rate != 16000:
            audio = audio.resample(16000)

        # Convert to bytes
        audio_bytes = audio.to_bytes(format="wav")

        # Make API request
        response = requests.post(
            self._endpoint,
            headers=self._headers,
            data=audio_bytes,
            timeout=self.config.timeout_seconds,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"MedASR API error: {response.status_code} - {response.text}"
            )

        result = response.json()

        # Parse response
        if isinstance(result, dict):
            text = result.get("text", "")
            confidence = result.get("confidence", 1.0)
        elif isinstance(result, list) and result:
            text = result[0].get("text", "")
            confidence = result[0].get("confidence", 1.0)
        else:
            text = str(result)
            confidence = 1.0

        return Transcript(
            text=text,
            confidence=confidence,
            language="en",
            metadata={"model": self.config.model_id, "backend": "cloud"},
        )

    def transcribe_streaming(
        self, chunks: Iterator[AudioChunk]
    ) -> Iterator[TranscriptChunk]:
        """Stream transcription (accumulate and transcribe)."""
        # Note: HuggingFace API doesn't support true streaming
        # Accumulate chunks and transcribe periodically
        accumulated_data = []
        accumulated_duration_ms = 0
        chunk_threshold_ms = 2000  # Transcribe every 2 seconds

        for chunk in chunks:
            if not chunk.is_speech:
                continue

            accumulated_data.append(chunk.data)
            accumulated_duration_ms += chunk.duration_ms

            if accumulated_duration_ms >= chunk_threshold_ms:
                # Transcribe accumulated audio
                import numpy as np

                combined = np.concatenate(accumulated_data)
                segment = AudioSegment(
                    data=combined,
                    sample_rate=chunk.sample_rate,
                )

                try:
                    transcript = self.transcribe(segment)
                    yield TranscriptChunk(
                        text=transcript.text,
                        is_final=False,
                        confidence=transcript.confidence,
                        timestamp_ms=accumulated_duration_ms,
                    )
                except Exception as e:
                    print(f"Transcription error: {e}")

                # Reset accumulator
                accumulated_data = []
                accumulated_duration_ms = 0

        # Final transcription
        if accumulated_data:
            import numpy as np

            combined = np.concatenate(accumulated_data)
            segment = AudioSegment(
                data=combined,
                sample_rate=16000,  # Assume 16kHz
            )

            try:
                transcript = self.transcribe(segment)
                yield TranscriptChunk(
                    text=transcript.text,
                    is_final=True,
                    confidence=transcript.confidence,
                    timestamp_ms=accumulated_duration_ms,
                )
            except Exception as e:
                print(f"Final transcription error: {e}")
