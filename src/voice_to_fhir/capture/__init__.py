"""
Audio Capture Module

Provides cross-platform audio capture with voice activity detection (VAD).
"""

from voice_to_fhir.capture.audio_capture import AudioCapture
from voice_to_fhir.capture.audio_utils import AudioSegment, AudioChunk
from voice_to_fhir.capture.vad import VoiceActivityDetector

__all__ = [
    "AudioCapture",
    "AudioSegment",
    "AudioChunk",
    "VoiceActivityDetector",
]
