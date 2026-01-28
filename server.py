"""
Voice-to-FHIR FastAPI Server

REST API for the voice-to-fhir pipeline, enabling web UI integration.

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn server:app --reload --port 8000

Endpoints:
    POST /api/v1/process-audio     - Process audio file to FHIR
    POST /api/v1/process-transcript - Process transcript text to FHIR
    GET  /api/v1/workflows         - List available workflows
    GET  /api/v1/health            - Health check

Author: Cleansheet LLC
License: CC BY 4.0
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import the pipeline
from voice_to_fhir import Pipeline
from voice_to_fhir.extraction.post_processor import post_process


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert audio file to WAV format using ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000",  # 16kHz sample rate (required for MedASR)
                "-ac", "1",      # Mono channel
                "-f", "wav",
                output_path
            ],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("[WARNING] ffmpeg not found. Install ffmpeg for audio conversion.")
        return False
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        return False

# =============================================================================
# App Configuration
# =============================================================================

app = FastAPI(
    title="Voice-to-FHIR API",
    description="Clinical voice documentation pipeline using MedASR and MedGemma",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS - allow requests from the CleansheetMedical frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080",
        "file://",  # Allow file:// protocol for local HTML
        "*",  # For development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Pipeline Initialization
# =============================================================================

from voice_to_fhir.pipeline.config import PipelineConfig

# Global pipeline instance - cached by config hash
_pipelines: dict[str, Pipeline] = {}


def get_pipeline(
    hf_token: Optional[str] = None,
    medasr_endpoint: Optional[str] = None,
    medgemma_endpoint: Optional[str] = None,
    transcription_backend: str = "whisper",
    extraction_backend: str = "dedicated",
) -> Pipeline:
    """Get or create the pipeline instance for given configuration."""
    global _pipelines

    # Check for token: header takes precedence, then env var
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    # Check for endpoints from env if not provided
    medasr_url = medasr_endpoint or os.getenv("MEDASR_ENDPOINT_URL")
    medgemma_url = medgemma_endpoint or os.getenv("MEDGEMMA_ENDPOINT_URL")

    # Create cache key from config
    cache_key = f"{token or 'mock'}|{transcription_backend}|{medasr_url}|{extraction_backend}|{medgemma_url}"

    if cache_key not in _pipelines:
        if not token:
            print("[WARNING] No HF_TOKEN found. Using mock mode.")
            _pipelines[cache_key] = Pipeline.cloud(hf_token=None)
        else:
            print(f"[INFO] Creating pipeline with HF token: {token[:8]}...")
            print(f"[INFO] Transcription backend: {transcription_backend}")
            print(f"[INFO] Extraction backend: {extraction_backend}")
            if medasr_url:
                print(f"[INFO] MedASR endpoint: {medasr_url}")
            if medgemma_url:
                print(f"[INFO] MedGemma endpoint: {medgemma_url}")

            # Create config with endpoint URLs
            config = PipelineConfig()
            config.hf_token = token

            # Transcription config
            config.transcription.backend = transcription_backend
            if medasr_url:
                config.transcription.endpoint_url = medasr_url

            # Extraction config
            config.extraction.backend = extraction_backend
            if medgemma_url:
                config.extraction.endpoint_url = medgemma_url

            _pipelines[cache_key] = Pipeline(config)

    return _pipelines[cache_key]


# Available clinical workflows
WORKFLOWS = [
    {"id": "general", "name": "General Encounter", "description": "Standard clinical encounters"},
    {"id": "soap", "name": "SOAP Note", "description": "Structured Subjective/Objective/Assessment/Plan format"},
    {"id": "hp", "name": "History & Physical", "description": "Comprehensive H&P, annual physicals, wellness exams"},
    {"id": "emergency", "name": "Emergency / Trauma", "description": "ED visits, trauma, acute care"},
    {"id": "intake", "name": "Patient Intake", "description": "New patient registration, full history"},
    {"id": "followup", "name": "Follow-up Visit", "description": "Return visits, progress notes"},
    {"id": "procedure", "name": "Procedure Note", "description": "Surgical and procedure documentation"},
    {"id": "discharge", "name": "Discharge Summary", "description": "Hospital discharge documentation"},
    {"id": "radiology", "name": "Radiology Dictation", "description": "Imaging study interpretation"},
    {"id": "lab_review", "name": "Lab Review", "description": "Laboratory result review"},
    {"id": "respiratory", "name": "Respiratory Assessment", "description": "RT assessments, ventilator management"},
    {"id": "icu", "name": "ICU / Critical Care", "description": "Critical care documentation"},
    {"id": "cardiology", "name": "Cardiology", "description": "Cardiac encounters and procedures"},
    {"id": "pediatrics", "name": "Pediatrics", "description": "Pediatric encounters"},
    {"id": "neurology", "name": "Neurology", "description": "Neurological assessments"},
]


# =============================================================================
# Request/Response Models
# =============================================================================

class TranscriptRequest(BaseModel):
    """Request body for transcript processing."""
    transcript: str
    workflow: str = "general"


class PipelineMetrics(BaseModel):
    """Timing metrics for pipeline steps."""
    conversion_ms: Optional[float] = None
    transcription_ms: Optional[float] = None
    extraction_ms: Optional[float] = None
    fhir_transform_ms: Optional[float] = None
    total_ms: Optional[float] = None
    # Model info
    transcription_model: Optional[str] = None
    extraction_model: Optional[str] = None


class ProcessingResponse(BaseModel):
    """Response from processing endpoints."""
    success: bool
    transcript: Optional[str] = None
    fhir_bundle: Optional[dict] = None
    entities: Optional[dict] = None  # Extracted entities (chief_complaint, family_history, social_history, etc.)
    error: Optional[str] = None
    workflow: str = "general"
    metrics: Optional[PipelineMetrics] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    pipeline_ready: bool


class ChunkProcessingResponse(BaseModel):
    """Response from chunk processing endpoint."""
    success: bool
    chunk_id: str
    transcript: Optional[str] = None
    entities: Optional[dict] = None  # Raw extracted entities (not yet FHIR)
    fhir_bundle: Optional[dict] = None
    error: Optional[str] = None
    is_final: bool = False  # True if this is the final chunk
    metrics: Optional[PipelineMetrics] = None


class Recommendation(BaseModel):
    """A single clinical decision support recommendation."""
    id: str
    category: str  # drug_interaction, contraindication, missing_data, clinical_alert, guideline
    severity: str  # info, warning, critical
    title: str
    description: str
    source: Optional[str] = None
    related_items: list[str] = []
    suggested_action: Optional[str] = None


class RecommendationSummary(BaseModel):
    """Summary of recommendations by severity."""
    total_recommendations: int = 0
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0


class AnalysisResponse(BaseModel):
    """Response from patient analysis endpoint."""
    success: bool
    recommendations: list[Recommendation] = []
    summary: RecommendationSummary = RecommendationSummary()
    error: Optional[str] = None
    metrics: Optional[PipelineMetrics] = None
    # For transparency: include prompt and raw response
    model_id: Optional[str] = None
    prompt: Optional[str] = None
    raw_response: Optional[str] = None


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        pipeline = get_pipeline()
        pipeline_ready = pipeline is not None
    except Exception:
        pipeline_ready = False

    return HealthResponse(
        status="ok",
        version="0.1.0",
        pipeline_ready=pipeline_ready,
    )


@app.get("/api/v1/workflows")
async def list_workflows():
    """List available clinical workflows."""
    return {"workflows": WORKFLOWS}


@app.post("/api/v1/process-audio", response_model=ProcessingResponse)
async def process_audio(
    file: UploadFile = File(...),
    workflow: str = Form("general"),
    x_hf_token: Optional[str] = Header(None, alias="X-HF-Token"),
    x_medasr_endpoint: Optional[str] = Header(None, alias="X-MedASR-Endpoint"),
    x_medgemma_endpoint: Optional[str] = Header(None, alias="X-MedGemma-Endpoint"),
    x_transcription_backend: Optional[str] = Header("whisper", alias="X-Transcription-Backend"),
    x_extraction_backend: Optional[str] = Header("dedicated", alias="X-Extraction-Backend"),
):
    """
    Process an audio file through the voice-to-FHIR pipeline.

    - **file**: Audio file (WAV, MP3, WEBM, etc.)
    - **workflow**: Clinical workflow to use for extraction
    - **X-HF-Token**: Optional HuggingFace API token (header)
    - **X-MedASR-Endpoint**: Optional MedASR HF Endpoint URL
    - **X-MedGemma-Endpoint**: Optional MedGemma HF Endpoint URL
    - **X-Transcription-Backend**: Backend for transcription (whisper, dedicated, local)
    - **X-Extraction-Backend**: Backend for extraction (dedicated, local, serverless)

    Returns transcript and FHIR R4 Bundle.
    """
    # Validate workflow
    valid_workflows = [w["id"] for w in WORKFLOWS]
    if workflow not in valid_workflows:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid workflow. Must be one of: {valid_workflows}"
        )

    # Save uploaded file to temp location
    suffix = Path(file.filename).suffix.lower() if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    wav_path = None  # Track converted file for cleanup
    import time
    total_start = time.time()
    metrics = PipelineMetrics()

    try:
        pipeline = get_pipeline(
            hf_token=x_hf_token,
            medasr_endpoint=x_medasr_endpoint,
            medgemma_endpoint=x_medgemma_endpoint,
            transcription_backend=x_transcription_backend or "whisper",
            extraction_backend=x_extraction_backend or "dedicated",
        )

        # Record model info
        trans_backend = x_transcription_backend or "whisper"
        ext_backend = x_extraction_backend or "dedicated"
        metrics.transcription_model = f"Whisper" if trans_backend == "whisper" else f"MedASR ({trans_backend})"
        metrics.extraction_model = f"MedGemma ({ext_backend})"

        # Step 1: Convert to WAV if not already WAV format
        conversion_start = time.time()
        audio_path = tmp_path
        if suffix not in [".wav", ".flac", ".ogg"]:
            print(f"[INFO] Converting {suffix} to WAV...")
            wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"
            if convert_to_wav(tmp_path, wav_path):
                audio_path = wav_path
                print(f"[INFO] Converted to {wav_path}")
            else:
                return ProcessingResponse(
                    success=False,
                    error=f"Failed to convert {suffix} to WAV. Make sure ffmpeg is installed.",
                    workflow=workflow,
                )
        metrics.conversion_ms = round((time.time() - conversion_start) * 1000, 1)
        print(f"[TIMING] Audio conversion: {metrics.conversion_ms}ms")

        # Step 2: Transcription
        try:
            from voice_to_fhir.capture.audio_utils import AudioSegment
            audio = AudioSegment.from_file(audio_path)

            transcription_start = time.time()
            transcript_obj = pipeline.transcriber.transcribe(audio)
            transcript_text = transcript_obj.text
            metrics.transcription_ms = round((time.time() - transcription_start) * 1000, 1)
            print(f"[TIMING] Transcription: {metrics.transcription_ms}ms")

            # Step 3: Entity extraction
            extraction_start = time.time()
            entities = pipeline.extractor.extract(transcript_text, workflow)
            metrics.extraction_ms = round((time.time() - extraction_start) * 1000, 1)
            print(f"[TIMING] Extraction: {metrics.extraction_ms}ms")

            # Step 3.5: Post-processing - extract from markers, filter placeholders
            print(f"[Post-process] Applying transcript marker extraction and validation...")
            entities = post_process(entities, transcript_text)

            # Step 3.6: DIRECT BP extraction from transcript (bypasses module caching issues)
            import re
            bp_patterns = [
                r'blood\s+pressure\s+(?:today\s+|is\s+|of\s+)?(\d{2,3})\s*/\s*(\d{2,3})',
                r'\bBP\s+(\d{2,3})\s*/\s*(\d{2,3})',
                r'blood\s+pressure\s+(?:is\s+|today\s+)?(\d{2,3})\s+over\s+(\d{2,3})',
                r'\bBP\s+(\d{2,3})\s+over\s+(\d{2,3})',
            ]
            for pattern in bp_patterns:
                bp_match = re.search(pattern, transcript_text, re.IGNORECASE)
                if bp_match:
                    bp_value = f"{bp_match.group(1)}/{bp_match.group(2)}"
                    print(f"[BP Direct] Extracted from transcript: {bp_value}")
                    # Remove partial BP values and deduplicate complete BP values
                    from voice_to_fhir.extraction.extraction_types import Vital
                    seen_bp = set()
                    cleaned_vitals = []
                    for v in entities.vitals:
                        if hasattr(v, 'unit') and v.unit and v.unit.lower() in ('mmhg', 'mm hg'):
                            # Skip partial BP values (no slash)
                            if '/' not in str(v.value):
                                continue
                            # Skip duplicate complete BP values
                            if v.value in seen_bp:
                                continue
                            seen_bp.add(v.value)
                        cleaned_vitals.append(v)
                    # Add extracted BP at front if not already present
                    if bp_value not in seen_bp:
                        cleaned_vitals.insert(0, Vital(type="blood_pressure", value=bp_value, unit="mmHg"))
                    entities.vitals = cleaned_vitals
                    break

            # Step 4: FHIR transformation
            fhir_start = time.time()
            bundle = pipeline.transformer.transform(entities)
            metrics.fhir_transform_ms = round((time.time() - fhir_start) * 1000, 1)
            print(f"[TIMING] FHIR transform: {metrics.fhir_transform_ms}ms")

            # Build entities dict for response (includes family_history, social_history, etc.)
            # Use chief_complaint_text (symptom/reason for visit) if available, fallback to condition name
            chief_condition = entities.chief_complaint
            entities_dict = {
                "chief_complaint": entities.chief_complaint_text or (chief_condition.name if chief_condition else None),
                "conditions": [{"name": c.name, "icd10": c.icd10, "status": c.status, "severity": c.severity, "is_chief_complaint": c.is_chief_complaint} for c in entities.conditions],
                "medications": [{"name": m.name, "dose": m.dose, "frequency": m.frequency, "route": m.route} for m in entities.medications],
                "allergies": [{"substance": a.substance, "reaction": a.reaction, "severity": a.severity} for a in entities.allergies],
                "vitals": [{"type": v.type, "value": v.value, "unit": v.unit} for v in entities.vitals],
                "lab_results": [{"name": l.name, "value": l.value, "unit": l.unit, "interpretation": l.interpretation} for l in entities.lab_results],
                "family_history": [{"relationship": fh.relationship, "condition": fh.condition, "age_of_onset": fh.age_of_onset, "deceased": fh.deceased} for fh in entities.family_history],
                "social_history": {
                    "tobacco": entities.social_history.tobacco,
                    "alcohol": entities.social_history.alcohol,
                    "drugs": entities.social_history.drugs,
                    "occupation": entities.social_history.occupation,
                    "living_situation": entities.social_history.living_situation,
                } if entities.social_history else None,
                "patient": {
                    "name": entities.patient.name if entities.patient else None,
                    "age": getattr(entities.patient, 'age', None) if entities.patient else None,
                    "gender": entities.patient.gender if entities.patient else None,
                    "date_of_birth": getattr(entities.patient, 'date_of_birth', None) if entities.patient else None,
                } if entities.patient else None,
            }

        except Exception as e:
            # If pipeline fails (e.g., no HF_TOKEN), return mock data for demo
            error_msg = str(e)
            if "token" in error_msg.lower() or "auth" in error_msg.lower():
                # Return demo response
                return ProcessingResponse(
                    success=True,
                    transcript="[DEMO MODE - No HF_TOKEN configured] This is a simulated transcript for demonstration purposes.",
                    fhir_bundle=_get_demo_bundle(workflow),
                    workflow=workflow,
                )
            raise

        metrics.total_ms = round((time.time() - total_start) * 1000, 1)
        print(f"[TIMING] Total pipeline: {metrics.total_ms}ms")

        return ProcessingResponse(
            success=True,
            transcript=transcript_text,
            fhir_bundle=bundle,
            entities=entities_dict,
            workflow=workflow,
            metrics=metrics,
        )

    except Exception as e:
        return ProcessingResponse(
            success=False,
            error=str(e),
            workflow=workflow,
        )

    finally:
        # Clean up temp files
        for path in [tmp_path, wav_path]:
            if path:
                try:
                    os.unlink(path)
                except Exception:
                    pass


@app.post("/api/v1/process-transcript", response_model=ProcessingResponse)
async def process_transcript(
    request: TranscriptRequest,
    x_hf_token: Optional[str] = Header(None, alias="X-HF-Token"),
    x_medgemma_endpoint: Optional[str] = Header(None, alias="X-MedGemma-Endpoint"),
    x_extraction_backend: Optional[str] = Header("dedicated", alias="X-Extraction-Backend"),
):
    """
    Process a transcript text through the extraction and FHIR transformation pipeline.

    Skips the transcription step - useful for testing extraction without audio.

    - **transcript**: Clinical note text
    - **workflow**: Clinical workflow to use for extraction
    - **X-HF-Token**: Optional HuggingFace API token (header)
    - **X-MedGemma-Endpoint**: Optional MedGemma HF Endpoint URL
    - **X-Extraction-Backend**: Backend for extraction (dedicated, local, serverless)

    Returns FHIR R4 Bundle.
    """
    # Validate workflow
    valid_workflows = [w["id"] for w in WORKFLOWS]
    if request.workflow not in valid_workflows:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid workflow. Must be one of: {valid_workflows}"
        )

    try:
        pipeline = get_pipeline(
            hf_token=x_hf_token,
            medgemma_endpoint=x_medgemma_endpoint,
            extraction_backend=x_extraction_backend or "dedicated",
        )

        try:
            bundle = pipeline.process_transcript(
                request.transcript,
                workflow=request.workflow
            )
        except Exception as e:
            # If pipeline fails, return demo response
            error_msg = str(e)
            if "token" in error_msg.lower() or "auth" in error_msg.lower():
                return ProcessingResponse(
                    success=True,
                    transcript=request.transcript,
                    fhir_bundle=_get_demo_bundle(request.workflow),
                    workflow=request.workflow,
                )
            raise

        return ProcessingResponse(
            success=True,
            transcript=request.transcript,
            fhir_bundle=bundle,
            workflow=request.workflow,
        )

    except Exception as e:
        return ProcessingResponse(
            success=False,
            transcript=request.transcript,
            error=str(e),
            workflow=request.workflow,
        )


@app.post("/api/v1/process-chunk", response_model=ChunkProcessingResponse)
async def process_chunk(
    file: UploadFile = File(...),
    chunk_id: str = Form(...),
    context: str = Form(""),  # Previous transcript context
    workflow: str = Form("general"),
    is_final: bool = Form(False),
    x_hf_token: Optional[str] = Header(None, alias="X-HF-Token"),
    x_medasr_endpoint: Optional[str] = Header(None, alias="X-MedASR-Endpoint"),
    x_medgemma_endpoint: Optional[str] = Header(None, alias="X-MedGemma-Endpoint"),
    x_transcription_backend: Optional[str] = Header("whisper", alias="X-Transcription-Backend"),
    x_extraction_backend: Optional[str] = Header("dedicated", alias="X-Extraction-Backend"),
    x_skip_extraction: Optional[str] = Header(None, alias="X-Skip-Extraction"),
):
    """
    Process a single audio chunk from a streaming recording session.

    Designed for VAD-based chunking where audio is sent in segments
    detected by voice activity detection pauses.

    - **file**: Audio chunk (WAV, WEBM, etc.)
    - **chunk_id**: Unique identifier for this chunk
    - **context**: Previous transcript text for context continuity
    - **workflow**: Clinical workflow for extraction
    - **is_final**: True if this is the last chunk in the session
    - **X-HF-Token**: HuggingFace API token
    - **X-Transcription-Backend**: ASR backend (whisper, local)
    - **X-Extraction-Backend**: Extraction backend (dedicated, local, serverless)

    Returns transcript and extracted entities for this chunk.
    """
    import time
    import uuid as uuid_mod

    suffix = Path(file.filename).suffix.lower() if file.filename else ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    wav_path = None
    total_start = time.time()
    metrics = PipelineMetrics()

    try:
        pipeline = get_pipeline(
            hf_token=x_hf_token,
            medasr_endpoint=x_medasr_endpoint,
            medgemma_endpoint=x_medgemma_endpoint,
            transcription_backend=x_transcription_backend or "whisper",
            extraction_backend=x_extraction_backend or "dedicated",
        )

        # Record model info
        trans_backend = x_transcription_backend or "whisper"
        ext_backend = x_extraction_backend or "dedicated"
        metrics.transcription_model = f"Whisper" if trans_backend == "whisper" else f"MedASR ({trans_backend})"
        metrics.extraction_model = f"MedGemma ({ext_backend})"

        # Step 1: Convert to WAV if needed
        conversion_start = time.time()
        audio_path = tmp_path
        if suffix not in [".wav", ".flac", ".ogg"]:
            wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"
            if convert_to_wav(tmp_path, wav_path):
                audio_path = wav_path
            else:
                return ChunkProcessingResponse(
                    success=False,
                    chunk_id=chunk_id,
                    error=f"Failed to convert {suffix} to WAV",
                    is_final=is_final,
                )
        metrics.conversion_ms = round((time.time() - conversion_start) * 1000, 1)

        # Step 2: Transcription
        try:
            from voice_to_fhir.capture.audio_utils import AudioSegment
            audio = AudioSegment.from_file(audio_path)

            transcription_start = time.time()
            transcript_obj = pipeline.transcriber.transcribe(audio)
            chunk_transcript = transcript_obj.text
            metrics.transcription_ms = round((time.time() - transcription_start) * 1000, 1)

            # Combine with context for extraction
            full_transcript = f"{context} {chunk_transcript}".strip() if context else chunk_transcript

            # Check if extraction should be skipped (transcription-only mode for streaming)
            skip_extraction = x_skip_extraction and x_skip_extraction.lower() == 'true'

            if skip_extraction:
                # Transcription-only mode - skip entity extraction and FHIR transformation
                print(f"[Chunk] Transcription-only mode - skipping extraction")
                entities_dict = None
                bundle = None
            else:
                # Step 3: Entity extraction
                extraction_start = time.time()
                print(f"[Extraction] Full transcript length: {len(full_transcript)} chars")
                print(f"[Extraction] Transcript preview: {full_transcript[:200]}..." if len(full_transcript) > 200 else f"[Extraction] Full transcript: {full_transcript}")

                entities = pipeline.extractor.extract(full_transcript, workflow)
                metrics.extraction_ms = round((time.time() - extraction_start) * 1000, 1)

                # Step 3.5: Post-processing - extract from markers, filter placeholders
                print(f"[Post-process] Applying transcript marker extraction and validation...")
                entities = post_process(entities, full_transcript)

                # Step 3.6: DIRECT BP extraction from transcript (bypasses module caching issues)
                import re
                bp_patterns = [
                    r'blood\s+pressure\s+(?:today\s+|is\s+|of\s+)?(\d{2,3})\s*/\s*(\d{2,3})',
                    r'\bBP\s+(\d{2,3})\s*/\s*(\d{2,3})',
                    r'blood\s+pressure\s+(?:is\s+|today\s+)?(\d{2,3})\s+over\s+(\d{2,3})',
                    r'\bBP\s+(\d{2,3})\s+over\s+(\d{2,3})',
                ]
                for pattern in bp_patterns:
                    bp_match = re.search(pattern, full_transcript, re.IGNORECASE)
                    if bp_match:
                        bp_value = f"{bp_match.group(1)}/{bp_match.group(2)}"
                        print(f"[BP Direct] Extracted from transcript: {bp_value}")
                        # Remove partial BP values and deduplicate complete BP values
                        from voice_to_fhir.extraction.extraction_types import Vital
                        seen_bp = set()
                        cleaned_vitals = []
                        for v in entities.vitals:
                            if hasattr(v, 'unit') and v.unit and v.unit.lower() in ('mmhg', 'mm hg'):
                                # Skip partial BP values (no slash)
                                if '/' not in str(v.value):
                                    continue
                                # Skip duplicate complete BP values
                                if v.value in seen_bp:
                                    continue
                                seen_bp.add(v.value)
                            cleaned_vitals.append(v)
                        # Add extracted BP at front if not already present
                        if bp_value not in seen_bp:
                            cleaned_vitals.insert(0, Vital(type="blood_pressure", value=bp_value, unit="mmHg"))
                        entities.vitals = cleaned_vitals
                        break

                # Debug: Log extracted entities
                print(f"[Extraction] Conditions: {len(entities.conditions)}")
                print(f"[Extraction] Medications: {len(entities.medications)} - {[m.name for m in entities.medications]}")
                print(f"[Extraction] Allergies: {len(entities.allergies)} - {[a.substance for a in entities.allergies]}")
                print(f"[Extraction] Vitals: {len(entities.vitals)}")
                print(f"[Extraction] Family History: {len(entities.family_history)}")
                print(f"[Extraction] Social History: {'Yes' if entities.social_history else 'No'}")
                print(f"[Extraction] Chief Complaint: {entities.chief_complaint.name if entities.chief_complaint else 'None'}")

                # Step 4: FHIR transformation
                fhir_start = time.time()
                bundle = pipeline.transformer.transform(entities)
                metrics.fhir_transform_ms = round((time.time() - fhir_start) * 1000, 1)

                # Convert entities to dict for response
                # Use chief_complaint_text (symptom/reason for visit) if available, fallback to condition name
                chief_condition = entities.chief_complaint
                entities_dict = {
                    "chief_complaint": entities.chief_complaint_text or (chief_condition.name if chief_condition else None),
                    "conditions": [{"name": c.name, "icd10": c.icd10, "status": c.status, "severity": c.severity, "is_chief_complaint": c.is_chief_complaint} for c in entities.conditions],
                    "medications": [{"name": m.name, "dose": m.dose, "frequency": m.frequency, "route": m.route, "status": m.status, "is_new_order": m.is_new_order} for m in entities.medications],
                    "allergies": [{"substance": a.substance, "reaction": a.reaction, "severity": a.severity} for a in entities.allergies],
                    "vitals": [{"type": v.type, "value": v.value, "unit": v.unit} for v in entities.vitals],
                    "lab_results": [{"name": l.name, "value": l.value, "unit": l.unit, "interpretation": l.interpretation} for l in entities.lab_results],
                    "family_history": [{"relationship": fh.relationship, "condition": fh.condition, "age_of_onset": fh.age_of_onset, "deceased": fh.deceased} for fh in entities.family_history],
                    "social_history": {
                        "tobacco": entities.social_history.tobacco,
                        "alcohol": entities.social_history.alcohol,
                        "drugs": entities.social_history.drugs,
                        "occupation": entities.social_history.occupation,
                        "living_situation": entities.social_history.living_situation,
                    } if entities.social_history else None,
                    "patient": {
                        "name": entities.patient.name,
                        "age": getattr(entities.patient, 'age', None),
                        "gender": entities.patient.gender,
                        "date_of_birth": entities.patient.date_of_birth,
                    } if entities.patient else None,
                    "lab_orders": [{"name": lo.name, "loinc": lo.loinc} for lo in entities.lab_orders],
                    "medication_orders": [{"name": mo.name, "dose": mo.dose, "frequency": mo.frequency, "duration": getattr(mo, 'duration', None), "instructions": mo.instructions} for mo in entities.medication_orders],
                    "referral_orders": [{"specialty": ro.specialty, "reason": ro.reason} for ro in entities.referral_orders],
                    "procedure_orders": [{"name": po.name} for po in entities.procedure_orders],
                    "imaging_orders": [{"name": io.name} for io in entities.imaging_orders],
                }

        except Exception as e:
            error_msg = str(e)
            if "token" in error_msg.lower() or "auth" in error_msg.lower():
                # Demo mode
                return ChunkProcessingResponse(
                    success=True,
                    chunk_id=chunk_id,
                    transcript="[DEMO MODE] Chunk processed",
                    entities={
                        "chief_complaint": None,
                        "conditions": [], "medications": [], "allergies": [], "vitals": [], "lab_results": [],
                        "family_history": [], "social_history": None, "patient": None,
                        "lab_orders": [], "medication_orders": [], "referral_orders": [], "procedure_orders": [], "imaging_orders": [],
                    },
                    fhir_bundle=_get_demo_bundle(workflow),
                    is_final=is_final,
                    metrics=metrics,
                )
            raise

        metrics.total_ms = round((time.time() - total_start) * 1000, 1)

        return ChunkProcessingResponse(
            success=True,
            chunk_id=chunk_id,
            transcript=chunk_transcript,  # Return only this chunk's transcript
            entities=entities_dict,
            fhir_bundle=bundle,
            is_final=is_final,
            metrics=metrics,
        )

    except Exception as e:
        return ChunkProcessingResponse(
            success=False,
            chunk_id=chunk_id,
            error=str(e),
            is_final=is_final,
            metrics=metrics,
        )

    finally:
        for path in [tmp_path, wav_path]:
            if path:
                try:
                    os.unlink(path)
                except Exception:
                    pass


@app.post("/api/v1/analyze-patient", response_model=AnalysisResponse)
async def analyze_patient(
    patient_bundle: dict = Body(...),
    x_hf_token: Optional[str] = Header(None, alias="X-HF-Token"),
    x_medgemma_endpoint: Optional[str] = Header(None, alias="X-MedGemma-Endpoint"),
    x_extraction_backend: Optional[str] = Header("dedicated", alias="X-Extraction-Backend"),
):
    """
    Analyze patient data to generate clinical decision support recommendations.

    - **patient_bundle**: FHIR R4 Bundle containing patient data
    - **X-HF-Token**: Optional HuggingFace API token (header)
    - **X-MedGemma-Endpoint**: Optional MedGemma HF Endpoint URL
    - **X-Extraction-Backend**: Backend for extraction (dedicated, local, serverless)

    Returns recommendations for drug interactions, contraindications, missing data,
    clinical alerts, and guideline-based care.
    """
    import json
    import time
    import uuid

    total_start = time.time()
    metrics = PipelineMetrics()

    try:
        # Validate bundle has content
        if not patient_bundle or not patient_bundle.get("entry"):
            return AnalysisResponse(
                success=False,
                error="Empty or invalid FHIR bundle - please add patient data first"
            )

        # Get pipeline for MedGemma access
        pipeline = get_pipeline(
            hf_token=x_hf_token,
            medgemma_endpoint=x_medgemma_endpoint,
            extraction_backend=x_extraction_backend or "dedicated",
        )

        # Record model info
        metrics.extraction_model = f"MedGemma ({x_extraction_backend or 'dedicated'})"

        # Load recommendations prompt
        prompt_path = Path(__file__).parent / "src" / "voice_to_fhir" / "extraction" / "prompts" / "recommendations.txt"
        if not prompt_path.exists():
            return AnalysisResponse(
                success=False,
                error=f"Recommendations prompt not found at {prompt_path}"
            )

        prompt_template = prompt_path.read_text()

        # Serialize patient data for the prompt
        patient_data_str = json.dumps(patient_bundle, indent=2)
        prompt = prompt_template.replace("{patient_data}", patient_data_str)

        # Call MedGemma for analysis via direct HTTP
        import requests
        extraction_start = time.time()
        try:
            # Get config from pipeline extractor
            extractor = pipeline.extractor
            config = extractor.config

            # Build request based on backend
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"

            if config.backend == "dedicated":
                endpoint = f"{config.endpoint_url.rstrip('/')}/v1/chat/completions"
                payload = {
                    "model": config.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                }
            elif config.backend == "local":
                endpoint = config.local_url
                payload = {
                    "prompt": prompt,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                }
            else:
                endpoint = f"{config.api_url}/{config.model_id}"
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": config.max_tokens,
                        "temperature": config.temperature,
                        "return_full_text": False,
                    },
                }

            resp = requests.post(endpoint, headers=headers, json=payload, timeout=config.timeout_seconds)
            resp.raise_for_status()
            result_data = resp.json()

            # Extract response text based on backend format
            if config.backend == "dedicated":
                response = result_data["choices"][0]["message"]["content"]
            elif config.backend == "local":
                response = result_data.get("response", result_data.get("text", ""))
            else:
                if isinstance(result_data, list):
                    response = result_data[0].get("generated_text", "")
                else:
                    response = result_data.get("generated_text", "")

            metrics.extraction_ms = round((time.time() - extraction_start) * 1000, 1)
        except Exception as e:
            error_msg = str(e)
            if "token" in error_msg.lower() or "auth" in error_msg.lower() or "401" in error_msg:
                # Return demo recommendations for demo mode
                return _get_demo_recommendations(patient_bundle, metrics)
            raise

        # Store raw response for transparency
        raw_response = response

        # Get model ID for display
        model_id = config.model_id if hasattr(config, 'model_id') else "MedGemma"

        # Parse the JSON response
        try:
            # Clean up response if it contains markdown code blocks
            response_text = response.strip()
            if response_text.startswith("```"):
                # Remove markdown code block markers
                lines = response_text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                response_text = "\n".join(lines)

            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            return AnalysisResponse(
                success=False,
                error=f"Failed to parse MedGemma response as JSON: {e}",
                metrics=metrics,
                model_id=model_id,
                prompt=prompt,
                raw_response=raw_response,
            )

        # Convert to response model
        recommendations = []
        for rec in result.get("recommendations", []):
            recommendations.append(Recommendation(
                id=rec.get("id", str(uuid.uuid4())[:8]),
                category=rec.get("category", "guideline"),
                severity=rec.get("severity", "info"),
                title=rec.get("title", "Recommendation"),
                description=rec.get("description", ""),
                source=rec.get("source"),
                related_items=rec.get("related_items", []),
                suggested_action=rec.get("suggested_action"),
            ))

        summary_data = result.get("summary", {})
        summary = RecommendationSummary(
            total_recommendations=summary_data.get("total_recommendations", len(recommendations)),
            critical_count=summary_data.get("critical_count", sum(1 for r in recommendations if r.severity == "critical")),
            warning_count=summary_data.get("warning_count", sum(1 for r in recommendations if r.severity == "warning")),
            info_count=summary_data.get("info_count", sum(1 for r in recommendations if r.severity == "info")),
        )

        metrics.total_ms = round((time.time() - total_start) * 1000, 1)

        return AnalysisResponse(
            success=True,
            recommendations=recommendations,
            summary=summary,
            metrics=metrics,
            model_id=model_id,
            prompt=prompt,
            raw_response=raw_response,
        )

    except Exception as e:
        return AnalysisResponse(
            success=False,
            error=str(e),
            metrics=metrics,
        )


def _get_demo_recommendations(patient_bundle: dict, metrics: Optional[PipelineMetrics] = None) -> AnalysisResponse:
    """Generate demo recommendations when no HF_TOKEN is configured."""
    import uuid

    # Analyze the bundle to generate contextual demo recommendations
    recommendations = []
    entry = patient_bundle.get("entry", [])

    # Check for medications
    medications = [e["resource"] for e in entry if e.get("resource", {}).get("resourceType") == "MedicationStatement"]
    conditions = [e["resource"] for e in entry if e.get("resource", {}).get("resourceType") == "Condition"]
    allergies = [e["resource"] for e in entry if e.get("resource", {}).get("resourceType") == "AllergyIntolerance"]

    # Generate recommendations based on content
    if len(medications) >= 2:
        med_names = [m.get("medicationCodeableConcept", {}).get("text", "Unknown") for m in medications[:2]]
        recommendations.append(Recommendation(
            id=str(uuid.uuid4())[:8],
            category="drug_interaction",
            severity="warning",
            title=f"Potential Drug Interaction",
            description=f"[DEMO MODE] Review potential interaction between {med_names[0]} and {med_names[1]}. Consult drug interaction database for clinical significance.",
            source="Demo recommendation",
            related_items=med_names,
            suggested_action="Review interaction database and adjust therapy if needed",
        ))

    if conditions:
        cond_name = conditions[0].get("code", {}).get("text", "condition")
        recommendations.append(Recommendation(
            id=str(uuid.uuid4())[:8],
            category="guideline",
            severity="info",
            title=f"Care Guideline Available",
            description=f"[DEMO MODE] Evidence-based care guidelines are available for {cond_name}. Consider reviewing current management approach.",
            source="Demo recommendation",
            related_items=[cond_name],
            suggested_action="Review applicable clinical practice guidelines",
        ))

    if not allergies:
        recommendations.append(Recommendation(
            id=str(uuid.uuid4())[:8],
            category="missing_data",
            severity="warning",
            title="Allergy Status Not Documented",
            description="[DEMO MODE] No allergy information has been recorded. Please verify and document allergy status including NKDA if applicable.",
            source="Demo recommendation",
            related_items=[],
            suggested_action="Document allergy status or confirm No Known Drug Allergies (NKDA)",
        ))

    # Default recommendation if none generated
    if not recommendations:
        recommendations.append(Recommendation(
            id=str(uuid.uuid4())[:8],
            category="guideline",
            severity="info",
            title="Complete Clinical Assessment Recommended",
            description="[DEMO MODE] Add more patient data (conditions, medications, vitals) to generate clinical decision support recommendations.",
            source="Demo recommendation",
            related_items=[],
            suggested_action="Add patient clinical data to enable recommendations",
        ))

    summary = RecommendationSummary(
        total_recommendations=len(recommendations),
        critical_count=sum(1 for r in recommendations if r.severity == "critical"),
        warning_count=sum(1 for r in recommendations if r.severity == "warning"),
        info_count=sum(1 for r in recommendations if r.severity == "info"),
    )

    return AnalysisResponse(
        success=True,
        recommendations=recommendations,
        summary=summary,
        metrics=metrics,
        model_id="Demo Mode (no HF_TOKEN)",
        prompt=None,
        raw_response=None,
    )


# =============================================================================
# Demo/Mock Data (when no HF_TOKEN)
# =============================================================================

def _get_demo_bundle(workflow: str) -> dict:
    """Generate a demo FHIR bundle for when the real pipeline isn't available."""
    import uuid
    from datetime import datetime

    patient_id = str(uuid.uuid4())
    encounter_id = str(uuid.uuid4())

    return {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "transaction",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "meta": {
            "tag": [{"code": "demo", "display": "Demo/Mock Data"}]
        },
        "entry": [
            {
                "fullUrl": f"urn:uuid:{patient_id}",
                "resource": {
                    "resourceType": "Patient",
                    "id": patient_id,
                    "name": [{"text": "Demo Patient", "family": "Patient", "given": ["Demo"]}],
                    "gender": "unknown",
                },
                "request": {"method": "POST", "url": "Patient"}
            },
            {
                "fullUrl": f"urn:uuid:{encounter_id}",
                "resource": {
                    "resourceType": "Encounter",
                    "id": encounter_id,
                    "status": "in-progress",
                    "class": {"code": "AMB", "display": "ambulatory"},
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "period": {"start": datetime.utcnow().isoformat() + "Z"},
                },
                "request": {"method": "POST", "url": "Encounter"}
            },
            {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Condition",
                    "id": str(uuid.uuid4()),
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {
                        "coding": [{"system": "http://snomed.info/sct", "code": "DEMO", "display": f"Demo condition for {workflow} workflow"}],
                        "text": f"Demo condition ({workflow})"
                    },
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "encounter": {"reference": f"Encounter/{encounter_id}"},
                },
                "request": {"method": "POST", "url": "Condition"}
            },
        ]
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    print(f"Starting Voice-to-FHIR API server on port {port}")
    print("Docs available at: http://localhost:{port}/docs")

    uvicorn.run(app, host="0.0.0.0", port=port)
