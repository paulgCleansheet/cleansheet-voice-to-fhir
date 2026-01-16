"""
FHIR Module

Transform extracted clinical entities to FHIR R4 resources.
"""

from voice_to_fhir.fhir.transformer import FHIRTransformer, FHIRConfig
from voice_to_fhir.fhir.validators import validate_bundle, ValidationResult

__all__ = [
    "FHIRTransformer",
    "FHIRConfig",
    "validate_bundle",
    "ValidationResult",
]
