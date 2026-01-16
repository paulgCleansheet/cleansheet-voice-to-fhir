#!/usr/bin/env python3
"""
Entity Extraction Only Example

Demonstrates using the extraction component without full FHIR transformation.
Useful when you want structured data but not FHIR format.

Usage:
    python examples/extract_entities_only.py

Requirements:
    - HF_TOKEN environment variable set
"""

import json
import os

from voice_to_fhir.extraction.medgemma_client import MedGemmaClient, MedGemmaClientConfig


def main():
    # Create extractor directly (without full pipeline)
    config = MedGemmaClientConfig(
        api_key=os.environ.get("HF_TOKEN"),
        max_tokens=2048,
        temperature=0.1,
    )
    extractor = MedGemmaClient(config)

    # Sample transcript
    transcript = """
    This is a 58-year-old gentleman with a history of COPD and CHF who
    presents today with increased shortness of breath over the past 3 days.
    He reports sleeping on 3 pillows at night and has noticed increased
    lower extremity edema. He ran out of his furosemide 5 days ago.

    Current medications: metoprolol 25mg twice daily, lisinopril 10mg daily,
    albuterol inhaler as needed.

    Allergies: Sulfa drugs cause rash.

    Vitals: BP 152/94, HR 88, RR 22, O2 sat 91% on room air, weight up
    8 pounds from last visit.

    Assessment: Acute CHF exacerbation likely due to medication
    non-compliance. Will restart furosemide and increase to 40mg daily.
    """

    print("=" * 60)
    print("Entity Extraction (No FHIR Transformation)")
    print("=" * 60)
    print()

    # List available workflows
    print("Available workflows:", extractor.available_workflows())
    print()

    # Extract entities
    print("Extracting clinical entities...")
    print("-" * 60)

    entities = extractor.extract(transcript, workflow="general")

    # Access extracted data directly
    print("\n📋 EXTRACTED ENTITIES")
    print("=" * 60)

    if entities.patient:
        print(f"\n👤 Patient:")
        print(f"   Name: {entities.patient.name or 'Not specified'}")
        print(f"   DOB: {entities.patient.date_of_birth or 'Not specified'}")
        print(f"   Gender: {entities.patient.gender or 'Not specified'}")

    if entities.conditions:
        print(f"\n🏥 Conditions ({len(entities.conditions)}):")
        for c in entities.conditions:
            chief = "⭐ " if c.is_chief_complaint else "   "
            print(f"{chief}{c.description} [{c.severity.value}]")
            if c.onset:
                print(f"      Onset: {c.onset}")

    if entities.observations:
        print(f"\n📊 Observations ({len(entities.observations)}):")
        for o in entities.observations:
            unit = f" {o.unit}" if o.unit else ""
            print(f"   {o.name}: {o.value}{unit}")

    if entities.allergies:
        print(f"\n⚠️  Allergies ({len(entities.allergies)}):")
        for a in entities.allergies:
            reaction = f" - {a.reaction}" if a.reaction else ""
            print(f"   {a.substance}{reaction}")

    if entities.current_medications:
        print(f"\n💊 Current Medications ({len(entities.current_medications)}):")
        for m in entities.current_medications:
            dose = f" {m.dose}" if m.dose else ""
            freq = f" {m.frequency}" if m.frequency else ""
            print(f"   {m.name}{dose}{freq}")

    if entities.new_medications:
        print(f"\n💊 New/Changed Medications ({len(entities.new_medications)}):")
        for m in entities.new_medications:
            dose = f" {m.dose}" if m.dose else ""
            freq = f" {m.frequency}" if m.frequency else ""
            print(f"   {m.name}{dose}{freq}")

    # Export as JSON (non-FHIR)
    print("\n" + "=" * 60)
    print("Raw JSON output:")
    print("-" * 60)

    # Convert to dict for JSON output
    output = {
        "workflow": entities.workflow,
        "patient": entities.patient.__dict__ if entities.patient else None,
        "conditions": [
            {
                "description": c.description,
                "severity": c.severity.value,
                "onset": c.onset,
                "is_chief_complaint": c.is_chief_complaint,
            }
            for c in entities.conditions
        ],
        "observations": [
            {"name": o.name, "value": o.value, "unit": o.unit}
            for o in entities.observations
        ],
        "allergies": [
            {"substance": a.substance, "reaction": a.reaction}
            for a in entities.allergies
        ],
        "current_medications": [
            {"name": m.name, "dose": m.dose, "frequency": m.frequency}
            for m in entities.current_medications
        ],
        "new_medications": [
            {"name": m.name, "dose": m.dose, "frequency": m.frequency}
            for m in entities.new_medications
        ],
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
