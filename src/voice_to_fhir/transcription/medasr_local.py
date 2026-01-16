"""
MedASR Local Inference

Local inference for MedASR on edge devices (Jetson, NUC, etc.).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from voice_to_fhir.capture.audio_utils import AudioSegment, AudioChunk
from voice_to_fhir.transcription.transcript_types import Transcript, TranscriptChunk


@dataclass
class MedASRLocalConfig:
    """Configuration for local MedASR inference."""

    model_path: str | Path = "models/medasr"
    device: str = "cuda"  # cuda, cpu, tensorrt
    precision: str = "fp16"  # fp32, fp16, int8
    use_tensorrt: bool = False
    tensorrt_cache_dir: str | None = None


class MedASRLocal:
    """Local MedASR inference for edge deployment."""

    def __init__(self, config: MedASRLocalConfig | None = None):
        """Initialize local MedASR model."""
        self.config = config or MedASRLocalConfig()
        self._model = None
        self._processor = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of model."""
        if self._initialized:
            return

        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        """Load MedASR model."""
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        import torch

        model_path = Path(self.config.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run 'python scripts/download_models.py' first."
            )

        # Determine device and dtype
        if self.config.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16 if self.config.precision == "fp16" else torch.float32
        else:
            device = "cpu"
            dtype = torch.float32

        # Load processor and model
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
        )

        # Apply TensorRT optimization if requested
        if self.config.use_tensorrt and device == "cuda":
            self._apply_tensorrt_optimization()

    def _apply_tensorrt_optimization(self) -> None:
        """Apply TensorRT optimization to model."""
        # TensorRT optimization is platform-specific
        # This is a placeholder for Jetson deployment
        try:
            import torch_tensorrt

            # TensorRT compilation would go here
            # This requires careful tuning for the specific model
            pass
        except ImportError:
            print("TensorRT not available, using standard PyTorch inference")

    def transcribe(self, audio: AudioSegment) -> Transcript:
        """Transcribe audio to text using local model."""
        import torch

        self._ensure_initialized()

        # Ensure correct sample rate
        if audio.sample_rate != 16000:
            audio = audio.resample(16000)

        # Process audio
        inputs = self._processor(
            audio.data,
            sampling_rate=16000,
            return_tensors="pt",
        )

        # Move to device
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate transcription
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs)

        # Decode
        transcription = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return Transcript(
            text=transcription.strip(),
            confidence=1.0,  # Local model doesn't provide confidence
            language="en",
            metadata={
                "model": str(self.config.model_path),
                "backend": "local",
                "device": self.config.device,
            },
        )

    def transcribe_streaming(
        self, chunks: Iterator[AudioChunk]
    ) -> Iterator[TranscriptChunk]:
        """Stream transcription from audio chunks."""
        import numpy as np

        accumulated_data = []
        accumulated_duration_ms = 0
        chunk_threshold_ms = 1500  # Lower threshold for local (faster)

        for chunk in chunks:
            if not chunk.is_speech:
                continue

            accumulated_data.append(chunk.data)
            accumulated_duration_ms += chunk.duration_ms

            if accumulated_duration_ms >= chunk_threshold_ms:
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
                    print(f"Local transcription error: {e}")

                accumulated_data = []
                accumulated_duration_ms = 0

        # Final chunk
        if accumulated_data:
            combined = np.concatenate(accumulated_data)
            segment = AudioSegment(
                data=combined,
                sample_rate=16000,
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
                print(f"Final local transcription error: {e}")

    def warmup(self) -> None:
        """Warm up the model with a dummy inference."""
        import numpy as np

        self._ensure_initialized()

        # Create dummy audio
        dummy_audio = AudioSegment(
            data=np.zeros(16000, dtype=np.float32),  # 1 second of silence
            sample_rate=16000,
        )

        # Run inference
        _ = self.transcribe(dummy_audio)
