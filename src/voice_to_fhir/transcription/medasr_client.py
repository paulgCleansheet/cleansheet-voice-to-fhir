"""
MedASR Cloud Client

HuggingFace Inference API client for MedASR transcription.

Supports multiple backends:
- dedicated: HuggingFace Inference Endpoints (paid, recommended)
- local: Local server running medasr-server.py
- whisper: Whisper via HF Inference API (fallback)
"""

from dataclasses import dataclass
import os
from typing import Iterator

import requests

from voice_to_fhir.capture.audio_utils import AudioSegment, AudioChunk
from voice_to_fhir.transcription.transcript_types import Transcript, TranscriptChunk


@dataclass
class MedASRClientConfig:
    """Configuration for MedASR cloud client."""

    api_key: str | None = None
    model_id: str = "google/medasr"
    # Backend: "dedicated" (HF Endpoint), "local", or "whisper" (fallback)
    backend: str = "dedicated"
    # For dedicated HF Endpoints (e.g., "https://xxxxx.endpoints.huggingface.cloud")
    endpoint_url: str | None = None
    # For local server (e.g., "http://localhost:3002")
    local_url: str = "http://localhost:3002"
    # For Whisper fallback via HF Inference Router
    whisper_url: str = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
    # Legacy serverless API (doesn't work for MedASR)
    api_url: str = "https://api-inference.huggingface.co/models"
    timeout: float = 600.0
    timeout_seconds: float = 600.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "MedASRClientConfig":
        """Create configuration from environment variables."""
        return cls(
            api_key=os.environ.get("HF_TOKEN"),
            model_id=os.environ.get("MEDASR_MODEL_ID", "google/medasr"),
            backend=os.environ.get("MEDASR_BACKEND", "dedicated"),
            endpoint_url=os.environ.get("MEDASR_ENDPOINT_URL"),
            local_url=os.environ.get("MEDASR_LOCAL_URL", "http://localhost:3002"),
            timeout=float(os.environ.get("MEDASR_TIMEOUT", "600.0")),
        )


class MedASRClient:
    """HuggingFace client for MedASR transcription.

    Supports multiple backends:
    - dedicated: HuggingFace Inference Endpoints (paid, recommended for MedASR)
    - local: Local server running medasr-server.py
    - whisper: Whisper via HF Inference API (fallback, not medical-specific)
    """

    def __init__(self, config: MedASRClientConfig | None = None):
        """Initialize MedASR client."""
        self.config = config or MedASRClientConfig()

        # Get API key from config or environment
        self.api_key = self.config.api_key or os.environ.get("HF_TOKEN")

        # Validate based on backend
        if self.config.backend in ["dedicated", "whisper"] and not self.api_key:
            raise ValueError(
                "HuggingFace API key required for cloud backends. "
                "Set HF_TOKEN environment variable or pass api_key in config."
            )

        if self.config.backend == "dedicated" and not self.config.endpoint_url:
            raise ValueError(
                "endpoint_url required for dedicated backend. "
                "Set MEDASR_ENDPOINT_URL or pass endpoint_url in config."
            )

    @property
    def _headers(self) -> dict[str, str]:
        """Request headers."""
        headers = {"Content-Type": "audio/wav"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @property
    def _endpoint(self) -> str:
        """API endpoint URL based on backend."""
        if self.config.backend == "dedicated":
            return self.config.endpoint_url
        elif self.config.backend == "local":
            return self.config.local_url
        elif self.config.backend == "whisper":
            return self.config.whisper_url
        else:
            # Legacy serverless (doesn't work for MedASR)
            return f"{self.config.api_url}/{self.config.model_id}"

    def health_check(self) -> bool:
        """Check if the API is reachable."""
        try:
            url = self._endpoint
            if self.config.backend == "local":
                url = f"{self.config.local_url}/health"

            response = requests.get(
                url,
                headers=self._headers,
                timeout=10.0,
            )
            return response.status_code in [200, 503]  # 503 = model loading
        except Exception:
            return False

    def _prepare_audio(self, audio: AudioSegment) -> bytes:
        """Prepare audio data for API request."""
        # Ensure correct sample rate (MedASR expects 16kHz)
        if audio.sample_rate != 16000:
            audio = audio.resample(16000)
        return audio.to_bytes(format="wav")

    def transcribe(self, audio: AudioSegment, language_hint: str | None = None) -> Transcript:
        """Transcribe audio to text using configured backend."""
        # Prepare audio
        audio_bytes = self._prepare_audio(audio)

        backend = self.config.backend
        print(f"[MedASR] Transcribing with backend: {backend}")

        # Make API request
        response = requests.post(
            self._endpoint,
            headers=self._headers,
            data=audio_bytes,
            timeout=self.config.timeout or self.config.timeout_seconds,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"MedASR API error: {response.status_code} - {response.text[:500]}"
            )

        result = response.json()

        # Parse response (format varies by backend)
        if isinstance(result, dict):
            text = result.get("text", "")
            confidence = result.get("confidence", 1.0)
        elif isinstance(result, list) and result:
            # Whisper returns list of chunks
            text = " ".join(chunk.get("text", str(chunk)) for chunk in result)
            confidence = result[0].get("confidence", 1.0) if isinstance(result[0], dict) else 1.0
        else:
            text = str(result)
            confidence = 1.0

        # Clean up text
        text = text.strip()

        return Transcript(
            text=text,
            confidence=confidence,
            language=language_hint or "en",
            metadata={"model": self.config.model_id, "backend": backend},
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
