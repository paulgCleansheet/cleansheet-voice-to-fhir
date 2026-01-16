#!/usr/bin/env python3
"""
Custom Configuration Example

Demonstrates creating a pipeline with custom configuration programmatically.

Usage:
    python examples/custom_config.py

Requirements:
    - HF_TOKEN environment variable set
"""

from voice_to_fhir import Pipeline, PipelineConfig
from voice_to_fhir.pipeline.config import (
    CaptureConfig,
    TranscriptionConfig,
    ExtractionConfig,
    FHIROutputConfig,
)


def main():
    # Create custom configuration programmatically
    config = PipelineConfig(
        name="custom-emergency-pipeline",
        version="1.0.0",

        # Audio capture settings
        capture=CaptureConfig(
            sample_rate=16000,
            channels=1,
            chunk_duration_ms=100,
            vad_enabled=True,
            vad_mode=2,  # Less aggressive for emergency - don't miss anything
        ),

        # Transcription settings
        transcription=TranscriptionConfig(
            backend="cloud",
            model_id="google/medasr",
        ),

        # Extraction settings
        extraction=ExtractionConfig(
            backend="cloud",
            model_id="google/medgemma-4b",
            max_tokens=2048,
            temperature=0.05,  # Lower for more deterministic output
            workflow="emergency",  # Emergency department workflow
        ),

        # FHIR output settings
        fhir=FHIROutputConfig(
            version="R4",
            base_url="http://ed-fhir.hospital.local/fhir",
            validate=True,
            output_format="json",
        ),
    )

    # Create pipeline with custom config
    pipeline = Pipeline(config)

    # Sample emergency transcript
    transcript = """
    32-year-old male, motorcycle accident, GCS 14. Patient was not wearing
    a helmet. Complains of severe left leg pain and headache. Obvious
    deformity of left femur. No loss of consciousness reported but patient
    is confused about events. Vitals: BP 90/60, HR 120, RR 24, O2 sat 96%
    on room air. Two large bore IVs placed, 1 liter NS bolus given.
    """

    print("Processing with custom emergency configuration...")
    print("-" * 50)

    # Process transcript
    bundle = pipeline.process_transcript(transcript)

    # Output
    print(pipeline.to_json(bundle, indent=2))

    # Show configuration used
    print("\n" + "=" * 50)
    print("Configuration used:")
    print(f"  Name: {config.name}")
    print(f"  Workflow: {config.extraction.workflow}")
    print(f"  Temperature: {config.extraction.temperature}")
    print(f"  FHIR Base URL: {config.fhir.base_url}")

    # Save config to file for reuse
    import yaml
    config_dict = config.to_dict()
    with open("my_custom_config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print("\nConfiguration saved to: my_custom_config.yaml")


if __name__ == "__main__":
    main()
