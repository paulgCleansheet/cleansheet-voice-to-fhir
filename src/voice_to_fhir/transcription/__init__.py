"""
Transcription Module

MedASR integration for medical speech recognition.
"""

from voice_to_fhir.transcription.transcript_types import Transcript, TranscriptChunk
from voice_to_fhir.transcription.medasr_client import MedASRClient
from voice_to_fhir.transcription.medasr_local import MedASRLocal

__all__ = [
    "Transcript",
    "TranscriptChunk",
    "MedASRClient",
    "MedASRLocal",
]
