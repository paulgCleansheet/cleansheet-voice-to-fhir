"""
Pytest Configuration and Shared Fixtures

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import numpy as np


# =============================================================================
# AUDIO FIXTURES
# =============================================================================


@pytest.fixture
def sample_audio_data() -> np.ndarray:
    """Generate sample audio data (1 second of silence with some noise)."""
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    # Generate low-amplitude noise
    return (np.random.randn(samples) * 0.01).astype(np.float32)


@pytest.fixture
def sample_audio_with_speech() -> np.ndarray:
    """Generate sample audio data simulating speech (higher amplitude)."""
    sample_rate = 16000
    duration = 2.0
    samples = int(sample_rate * duration)
    # Generate higher amplitude "speech-like" signal
    t = np.linspace(0, duration, samples)
    # Mix of frequencies to simulate speech
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 800 * t)
    )
    return signal.astype(np.float32)


@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate for tests."""
    return 16000


# =============================================================================
# TRANSCRIPT FIXTURES
# =============================================================================


@pytest.fixture
def sample_transcript_text() -> str:
    """Sample clinical transcript text."""
    return (
        "Patient is a 45-year-old male presenting with chest pain "
        "for the past two hours. Pain is substernal, radiating to the left arm. "
        "Patient reports associated shortness of breath and diaphoresis. "
        "Past medical history significant for hypertension and diabetes. "
        "Current medications include lisinopril 10mg daily and metformin 500mg twice daily. "
        "Allergic to penicillin, causes rash. "
        "Vital signs: blood pressure 150/90, heart rate 88, temperature 98.6, "
        "respiratory rate 18, oxygen saturation 97% on room air."
    )


@pytest.fixture
def sample_transcript_simple() -> str:
    """Simple clinical transcript for basic tests."""
    return "Patient has a headache and fever. Taking ibuprofen as needed."


# =============================================================================
# CLINICAL ENTITY FIXTURES
# =============================================================================


@pytest.fixture
def sample_clinical_entities() -> dict[str, Any]:
    """Sample extracted clinical entities."""
    return {
        "conditions": [
            {
                "name": "Chest pain",
                "icd10": "R07.9",
                "onset": "2 hours ago",
                "status": "active",
            },
            {
                "name": "Hypertension",
                "icd10": "I10",
                "status": "active",
            },
            {
                "name": "Type 2 Diabetes",
                "icd10": "E11.9",
                "status": "active",
            },
        ],
        "medications": [
            {
                "name": "Lisinopril",
                "dose": "10mg",
                "frequency": "daily",
                "rxnorm": "314076",
            },
            {
                "name": "Metformin",
                "dose": "500mg",
                "frequency": "twice daily",
                "rxnorm": "861004",
            },
        ],
        "allergies": [
            {
                "substance": "Penicillin",
                "reaction": "rash",
                "severity": "moderate",
            },
        ],
        "vitals": [
            {"type": "blood_pressure", "value": "150/90", "unit": "mmHg"},
            {"type": "heart_rate", "value": "88", "unit": "bpm"},
            {"type": "temperature", "value": "98.6", "unit": "F"},
            {"type": "respiratory_rate", "value": "18", "unit": "/min"},
            {"type": "oxygen_saturation", "value": "97", "unit": "%"},
        ],
        "patient": {
            "age": 45,
            "gender": "male",
        },
    }


@pytest.fixture
def sample_clinical_entities_minimal() -> dict[str, Any]:
    """Minimal clinical entities for simple tests."""
    return {
        "conditions": [
            {"name": "Headache", "status": "active"},
        ],
        "medications": [
            {"name": "Ibuprofen", "dose": "400mg", "frequency": "as needed"},
        ],
        "allergies": [],
        "vitals": [],
    }


# =============================================================================
# FHIR FIXTURES
# =============================================================================


@pytest.fixture
def sample_fhir_bundle() -> dict[str, Any]:
    """Sample FHIR R4 Bundle."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "patient-1",
                    "gender": "male",
                    "birthDate": "1979-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "condition-1",
                    "subject": {"reference": "Patient/patient-1"},
                    "code": {
                        "coding": [
                            {
                                "system": "http://hl7.org/fhir/sid/icd-10",
                                "code": "R07.9",
                                "display": "Chest pain, unspecified",
                            }
                        ]
                    },
                    "clinicalStatus": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                "code": "active",
                            }
                        ]
                    },
                }
            },
        ],
    }


@pytest.fixture
def invalid_fhir_bundle() -> dict[str, Any]:
    """Invalid FHIR Bundle for validation testing."""
    return {
        "resourceType": "Bundle",
        # Missing required "type" field
        "entry": [
            {
                "resource": {
                    # Missing resourceType
                    "id": "invalid-1",
                }
            },
        ],
    }


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def sample_config_dict() -> dict[str, Any]:
    """Sample pipeline configuration dictionary."""
    return {
        "name": "test-pipeline",
        "version": "1.0.0",
        "capture": {
            "sample_rate": 16000,
            "channels": 1,
            "vad_enabled": True,
        },
        "transcription": {
            "backend": "cloud",
            "model_id": "google/medasr",
        },
        "extraction": {
            "backend": "cloud",
            "model_id": "google/medgemma-4b",
            "workflow": "general",
        },
        "fhir": {
            "version": "R4",
            "validate": True,
        },
    }


@pytest.fixture
def temp_config_file(tmp_path: Path, sample_config_dict: dict) -> Path:
    """Create a temporary config file."""
    import yaml

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path


# =============================================================================
# MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_transcriber() -> MagicMock:
    """Mock transcriber that returns sample transcript."""
    from voice_to_fhir.transcription.transcript_types import Transcript, TranscriptSegment

    mock = MagicMock()
    mock.transcribe.return_value = Transcript(
        text="Patient has chest pain and shortness of breath.",
        segments=[
            TranscriptSegment(
                text="Patient has chest pain and shortness of breath.",
                start_time=0.0,
                end_time=3.0,
                confidence=0.95,
            )
        ],
        language="en",
        confidence=0.95,
    )
    return mock


@pytest.fixture
def mock_extractor(sample_clinical_entities_minimal: dict) -> MagicMock:
    """Mock extractor that returns sample entities."""
    from voice_to_fhir.extraction.extraction_types import ClinicalEntities

    mock = MagicMock()
    mock.extract.return_value = ClinicalEntities.from_dict(sample_clinical_entities_minimal)
    return mock


@pytest.fixture
def mock_transformer(sample_fhir_bundle: dict) -> MagicMock:
    """Mock transformer that returns sample bundle."""
    mock = MagicMock()
    mock.transform.return_value = sample_fhir_bundle
    mock.to_json.return_value = json.dumps(sample_fhir_bundle, indent=2)
    return mock


# =============================================================================
# PATH FIXTURES
# =============================================================================


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
