"""
Voice-to-FHIR Pipeline

Edge-deployable clinical voice documentation using MedASR and MedGemma.

Usage:
    from voice_to_fhir import Pipeline

    pipeline = Pipeline.from_config("configs/cloud.yaml")
    bundle = pipeline.process_file("recording.wav")
    print(bundle.to_json())

Author: Cleansheet LLC
License: CC BY 4.0
"""

from voice_to_fhir.pipeline.pipeline import Pipeline
from voice_to_fhir.pipeline.config import PipelineConfig

__version__ = "0.1.0"
__author__ = "Cleansheet LLC"
__license__ = "CC BY 4.0"

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "__version__",
]
