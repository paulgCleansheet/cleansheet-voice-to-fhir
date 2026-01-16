#!/usr/bin/env python3
"""
FHIR Server Integration Example

Demonstrates posting generated FHIR bundles to a FHIR server.

Usage:
    python examples/fhir_server_integration.py --server http://hapi.fhir.org/baseR4

Requirements:
    - HF_TOKEN environment variable set
    - FHIR server endpoint (HAPI FHIR public server used by default)
"""

import argparse
import json

import requests

from voice_to_fhir import Pipeline


def post_bundle_to_server(bundle: dict, server_url: str) -> dict:
    """Post a FHIR bundle to a server."""
    headers = {
        "Content-Type": "application/fhir+json",
        "Accept": "application/fhir+json",
    }

    response = requests.post(
        server_url,
        headers=headers,
        json=bundle,
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Post FHIR bundle to server")
    parser.add_argument(
        "--server", "-s",
        default="http://hapi.fhir.org/baseR4",
        help="FHIR server base URL"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate bundle but don't post to server"
    )
    args = parser.parse_args()

    # Sample transcript
    transcript = """
    Patient is a 45-year-old female with well-controlled type 2 diabetes,
    here for routine follow-up. HbA1c today is 6.8%, down from 7.2% three
    months ago. Blood pressure 128/78. Continue current medications:
    metformin 1000mg twice daily. Follow up in 3 months.
    """

    # Create pipeline
    pipeline = Pipeline.cloud()

    print("=" * 60)
    print("FHIR Server Integration")
    print("=" * 60)
    print(f"Server: {args.server}")
    print()

    # Process transcript
    print("Processing transcript...")
    bundle = pipeline.process_transcript(transcript, workflow="followup")

    # Convert to transaction bundle for posting
    transaction_bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [],
    }

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")
        resource_id = resource.get("id", "")

        transaction_bundle["entry"].append({
            "resource": resource,
            "request": {
                "method": "POST",
                "url": resource_type,
            },
        })

    print(f"Generated {len(transaction_bundle['entry'])} resources")

    if args.dry_run:
        print("\n--- DRY RUN: Bundle not posted ---")
        print(json.dumps(transaction_bundle, indent=2))
        return

    # Post to server
    print(f"\nPosting to {args.server}...")
    try:
        response = post_bundle_to_server(transaction_bundle, args.server)

        print("\nServer response:")
        print("-" * 40)

        for entry in response.get("entry", []):
            status = entry.get("response", {}).get("status", "Unknown")
            location = entry.get("response", {}).get("location", "")
            print(f"  {status}: {location}")

        print("\nBundle successfully posted to FHIR server!")

    except requests.RequestException as e:
        print(f"\nError posting to server: {e}")
        print("\nBundle that would have been posted:")
        print(json.dumps(transaction_bundle, indent=2))


if __name__ == "__main__":
    main()
