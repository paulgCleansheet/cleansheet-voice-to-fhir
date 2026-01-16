#!/usr/bin/env python3
"""
Workflow Comparison Example

Demonstrates how different clinical workflows extract different information
from the same transcript.

Usage:
    python examples/workflow_comparison.py

Requirements:
    - HF_TOKEN environment variable set
"""

import json
from voice_to_fhir import Pipeline


def main():
    # Sample ED transcript
    transcript = """
    This is a 67-year-old male brought in by EMS with acute onset chest pain
    and shortness of breath starting approximately 45 minutes ago. Patient
    was at home watching television when he developed sudden substernal
    chest pressure, described as 8 out of 10 in severity, radiating to his
    left arm and jaw. Associated symptoms include diaphoresis, nausea, and
    dyspnea. He took two aspirin at home before EMS arrival.

    Past medical history significant for hypertension, type 2 diabetes,
    hyperlipidemia, and a prior MI in 2019 with stent placement to the LAD.
    Current medications include metoprolol 50mg twice daily, lisinopril 20mg
    daily, metformin 1000mg twice daily, and aspirin 81mg daily.

    Allergies: Penicillin causes hives.

    Vital signs on arrival: blood pressure 165/98, heart rate 102,
    respiratory rate 22, oxygen saturation 94% on room air, temperature 98.6F.

    On exam, patient appears diaphoretic and in moderate distress. Lungs
    with bibasilar crackles. Heart regular rate and rhythm with no murmurs.

    EKG shows ST elevations in leads V2-V4 concerning for STEMI. Troponin
    pending. Cardiology consulted for emergent cath lab activation.
    """

    # Create pipeline
    pipeline = Pipeline.cloud()

    # Get available workflows
    workflows_to_compare = ["general", "emergency", "intake"]

    print("=" * 70)
    print("Workflow Comparison: Same Transcript, Different Extractions")
    print("=" * 70)
    print()
    print("Transcript excerpt:")
    print("-" * 70)
    print(transcript[:300] + "...")
    print()

    for workflow in workflows_to_compare:
        print("=" * 70)
        print(f"WORKFLOW: {workflow.upper()}")
        print("=" * 70)

        try:
            bundle = pipeline.process_transcript(transcript, workflow=workflow)

            # Count resources by type
            resource_types = {}
            for entry in bundle.get("entry", []):
                rt = entry.get("resource", {}).get("resourceType", "Unknown")
                resource_types[rt] = resource_types.get(rt, 0) + 1

            print(f"\nResources generated: {len(bundle.get('entry', []))}")
            for rt, count in sorted(resource_types.items()):
                print(f"  - {rt}: {count}")

            # Show sample of extracted data
            print("\nSample extraction:")

            # Find conditions
            conditions = [
                e["resource"] for e in bundle.get("entry", [])
                if e.get("resource", {}).get("resourceType") == "Condition"
            ]
            if conditions:
                print(f"\n  Conditions ({len(conditions)}):")
                for c in conditions[:3]:
                    code = c.get("code", {})
                    display = code.get("coding", [{}])[0].get("display", "Unknown")
                    print(f"    - {display}")

            # Find observations
            observations = [
                e["resource"] for e in bundle.get("entry", [])
                if e.get("resource", {}).get("resourceType") == "Observation"
            ]
            if observations:
                print(f"\n  Observations ({len(observations)}):")
                for o in observations[:3]:
                    code = o.get("code", {}).get("coding", [{}])[0].get("display", "Unknown")
                    value = o.get("valueString") or o.get("valueQuantity", {}).get("value", "")
                    print(f"    - {code}: {value}")

        except Exception as e:
            print(f"\nError processing with {workflow} workflow: {e}")

        print()

    print("=" * 70)
    print("Note: Each workflow emphasizes different clinical aspects.")
    print("Choose the workflow that best matches your documentation context.")
    print("=" * 70)


if __name__ == "__main__":
    main()
