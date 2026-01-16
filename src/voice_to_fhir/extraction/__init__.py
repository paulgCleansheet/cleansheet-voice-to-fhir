"""
Extraction Module

MedGemma integration for structured clinical entity extraction.
"""

from voice_to_fhir.extraction.extraction_types import (
    ClinicalEntities,
    Condition,
    Medication,
    Observation,
    Procedure,
    Allergy,
)
from voice_to_fhir.extraction.medgemma_client import MedGemmaClient
from voice_to_fhir.extraction.medgemma_local import MedGemmaLocal

__all__ = [
    "ClinicalEntities",
    "Condition",
    "Medication",
    "Observation",
    "Procedure",
    "Allergy",
    "MedGemmaClient",
    "MedGemmaLocal",
]
