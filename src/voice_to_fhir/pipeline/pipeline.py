"""
Voice-to-FHIR Pipeline

End-to-end orchestration of voice capture, transcription, extraction, and FHIR transformation.
"""

from pathlib import Path
from typing import Any, Iterator
import json

from voice_to_fhir.capture.audio_capture import AudioCapture, CaptureConfig
from voice_to_fhir.capture.audio_utils import AudioSegment
from voice_to_fhir.capture.vad import VADConfig
from voice_to_fhir.transcription.transcript_types import Transcript
from voice_to_fhir.transcription.medasr_client import MedASRClient, MedASRClientConfig
from voice_to_fhir.transcription.medasr_local import MedASRLocal, MedASRLocalConfig
from voice_to_fhir.extraction.extraction_types import ClinicalEntities
from voice_to_fhir.extraction.medgemma_client import MedGemmaClient, MedGemmaClientConfig
from voice_to_fhir.extraction.medgemma_local import MedGemmaLocal, MedGemmaLocalConfig
from voice_to_fhir.fhir.transformer import FHIRTransformer, FHIRConfig
from voice_to_fhir.fhir.validators import validate_bundle
from voice_to_fhir.pipeline.config import PipelineConfig, load_config


class Pipeline:
    """End-to-end voice-to-FHIR pipeline."""

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config

        # Initialize components (lazy)
        self._capture: AudioCapture | None = None
        self._transcriber: MedASRClient | MedASRLocal | None = None
        self._extractor: MedGemmaClient | MedGemmaLocal | None = None
        self._transformer: FHIRTransformer | None = None

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Pipeline":
        """Create pipeline from config file."""
        config = load_config(config_path)
        return cls(config)

    @classmethod
    def cloud(cls, hf_token: str | None = None) -> "Pipeline":
        """Create pipeline with cloud backends."""
        config = PipelineConfig()
        config.transcription.backend = "cloud"
        config.extraction.backend = "cloud"
        config.hf_token = hf_token
        return cls(config)

    @classmethod
    def local(cls, models_dir: str = "models") -> "Pipeline":
        """Create pipeline with local backends."""
        config = PipelineConfig()
        config.transcription.backend = "local"
        config.transcription.model_path = f"{models_dir}/medasr"
        config.extraction.backend = "local"
        config.extraction.model_path = f"{models_dir}/medgemma-4b"
        return cls(config)

    @property
    def capture(self) -> AudioCapture:
        """Get or create audio capture component."""
        if self._capture is None:
            cap_config = CaptureConfig(
                sample_rate=self.config.capture.sample_rate,
                channels=self.config.capture.channels,
                chunk_duration_ms=self.config.capture.chunk_duration_ms,
                vad_enabled=self.config.capture.vad_enabled,
                vad_config=VADConfig(webrtc_mode=self.config.capture.vad_mode),
            )
            self._capture = AudioCapture(cap_config)
        return self._capture

    @property
    def transcriber(self) -> MedASRClient | MedASRLocal:
        """Get or create transcription component."""
        if self._transcriber is None:
            backend = self.config.transcription.backend
            if backend == "local-model":
                # Local model files (requires downloaded models in models/ directory)
                local_config = MedASRLocalConfig(
                    model_path=self.config.transcription.model_path,
                    device=self.config.transcription.device,
                    precision=self.config.transcription.precision,
                    use_tensorrt=self.config.transcription.use_tensorrt,
                )
                self._transcriber = MedASRLocal(local_config)
            else:
                # Cloud backends: dedicated, whisper, or local HTTP server
                # "local" = MedASR server at localhost:3002
                # "dedicated" = HuggingFace dedicated endpoint
                # "whisper" = Whisper via HuggingFace API
                client_backend = backend
                if backend == "local":
                    client_backend = "local"  # HTTP server at local_url
                elif backend not in ["dedicated", "whisper"]:
                    client_backend = "whisper"  # Default fallback

                cloud_config = MedASRClientConfig(
                    api_key=self.config.hf_token,
                    model_id=self.config.transcription.model_id,
                    backend=client_backend,
                    endpoint_url=self.config.transcription.endpoint_url,
                    local_url=self.config.transcription.local_url,
                )
                self._transcriber = MedASRClient(cloud_config)
        return self._transcriber

    @property
    def extractor(self) -> MedGemmaClient | MedGemmaLocal:
        """Get or create extraction component."""
        if self._extractor is None:
            backend = self.config.extraction.backend
            if backend == "local":
                local_config = MedGemmaLocalConfig(
                    model_path=self.config.extraction.model_path,
                    device=self.config.extraction.device,
                    precision=self.config.extraction.precision,
                    max_tokens=self.config.extraction.max_tokens,
                    temperature=self.config.extraction.temperature,
                    prompts_dir=self.config.extraction.prompts_dir,
                )
                self._extractor = MedGemmaLocal(local_config)
            else:
                # Cloud backends: dedicated or serverless
                cloud_config = MedGemmaClientConfig(
                    api_key=self.config.hf_token,
                    model_id=self.config.extraction.model_id,
                    backend=backend if backend in ["dedicated", "serverless"] else "dedicated",
                    endpoint_url=self.config.extraction.endpoint_url,
                    local_url=self.config.extraction.local_url,
                    max_tokens=self.config.extraction.max_tokens,
                    temperature=self.config.extraction.temperature,
                    prompts_dir=self.config.extraction.prompts_dir,
                )
                self._extractor = MedGemmaClient(cloud_config)
        return self._extractor

    @property
    def transformer(self) -> FHIRTransformer:
        """Get or create FHIR transformer component."""
        if self._transformer is None:
            fhir_config = FHIRConfig(
                fhir_version=self.config.fhir.version,
                base_url=self.config.fhir.base_url,
                validate_output=self.config.fhir.validate,
            )
            self._transformer = FHIRTransformer(fhir_config)
        return self._transformer

    def process_audio(
        self, audio: AudioSegment, workflow: str | None = None
    ) -> dict[str, Any]:
        """Process audio segment to FHIR Bundle."""
        workflow = workflow or self.config.extraction.workflow

        # Step 1: Transcribe
        transcript = self.transcriber.transcribe(audio)

        # Step 2: Extract entities
        entities = self.extractor.extract(transcript.text, workflow)

        # Step 3: Transform to FHIR
        bundle = self.transformer.transform(entities)

        # Step 4: Validate if configured
        if self.config.fhir.validate:
            result = validate_bundle(bundle)
            if not result.is_valid:
                bundle["_validation"] = {
                    "valid": False,
                    "errors": [
                        {"path": e.path, "message": e.message}
                        for e in result.errors
                    ],
                }

        return bundle

    def process_file(
        self, filepath: str | Path, workflow: str | None = None
    ) -> dict[str, Any]:
        """Process audio file to FHIR Bundle."""
        audio = AudioSegment.from_file(filepath)
        return self.process_audio(audio, workflow)

    def process_transcript(
        self, transcript: str, workflow: str | None = None
    ) -> dict[str, Any]:
        """Process transcript text directly to FHIR Bundle."""
        workflow = workflow or self.config.extraction.workflow

        # Skip transcription, go directly to extraction
        entities = self.extractor.extract(transcript, workflow)
        bundle = self.transformer.transform(entities)

        if self.config.fhir.validate:
            result = validate_bundle(bundle)
            if not result.is_valid:
                bundle["_validation"] = {
                    "valid": False,
                    "errors": [
                        {"path": e.path, "message": e.message}
                        for e in result.errors
                    ],
                }

        return bundle

    def capture_and_process(
        self, max_duration_seconds: float = 30.0, workflow: str | None = None
    ) -> dict[str, Any]:
        """Capture audio from microphone and process to FHIR Bundle."""
        audio = self.capture.capture_until_silence(max_duration_seconds)
        return self.process_audio(audio, workflow)

    def process_realtime(
        self, workflow: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """Real-time processing with streaming output."""
        workflow = workflow or self.config.extraction.workflow

        # Start capture
        self.capture.start()

        try:
            # Stream transcription chunks
            for transcript_chunk in self.transcriber.transcribe_streaming(
                self.capture.stream()
            ):
                if transcript_chunk.is_final:
                    # Process final transcript
                    entities = self.extractor.extract(
                        transcript_chunk.text, workflow
                    )
                    bundle = self.transformer.transform(entities)
                    yield bundle
                else:
                    # Yield partial transcript
                    yield {
                        "_partial": True,
                        "transcript": transcript_chunk.text,
                        "confidence": transcript_chunk.confidence,
                    }
        finally:
            self.capture.stop()

    def to_json(self, bundle: dict[str, Any], indent: int = 2) -> str:
        """Serialize bundle to JSON string."""
        return self.transformer.to_json(bundle, indent)

    def save(
        self, bundle: dict[str, Any], filepath: str | Path, indent: int = 2
    ) -> None:
        """Save bundle to file."""
        path = Path(filepath)
        with open(path, "w") as f:
            json.dump(bundle, f, indent=indent)

    def warmup(self) -> None:
        """Warm up all components for faster first inference."""
        # Warm up local models if using local backend
        if self.config.transcription.backend == "local":
            if hasattr(self.transcriber, "warmup"):
                self.transcriber.warmup()

        if self.config.extraction.backend == "local":
            if hasattr(self.extractor, "warmup"):
                self.extractor.warmup()
