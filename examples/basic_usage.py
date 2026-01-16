#!/usr/bin/env python3
"""
Basic Usage Example

Demonstrates the simplest way to use the voice-to-FHIR pipeline.

Usage:
    python examples/basic_usage.py

Requirements:
    - HF_TOKEN environment variable set
    - pip install voice-to-fhir
"""

from voice_to_fhir import Pipeline, PipelineConfig


def main():
    # Create pipeline with default cloud configuration
    pipeline = Pipeline.cloud()

    # Process a transcript directly (skip audio capture/transcription)
    transcript = """
    Patient is a 52-year-old female presenting with chest pain for the past
    3 hours. Pain is described as pressure-like, substernal, radiating to
    the left arm. Associated symptoms include shortness of breath and
    diaphoresis. Past medical history significant for hypertension and
    hyperlipidemia. Current medications include lisinopril 20mg daily and
    atorvastatin 40mg at bedtime. No known drug allergies. Vital signs:
    blood pressure 168/95, heart rate 92, respiratory rate 20, oxygen
    saturation 96% on room air, temperature 98.4F.
    """

    # Extract and transform to FHIR
    bundle = pipeline.process_transcript(transcript)

    # Output as JSON
    print(pipeline.to_json(bundle, indent=2))

    # Summary
    print(f"\n--- Generated {len(bundle.get('entry', []))} FHIR resources ---")


if __name__ == "__main__":
    main()
