#!/usr/bin/env python3
"""
Process Audio File Example

Demonstrates processing an audio file through the full pipeline.

Usage:
    python examples/process_audio_file.py <audio_file> [--output output.json]

Requirements:
    - HF_TOKEN environment variable set
    - Audio file in WAV, MP3, or other supported format
"""

import argparse
import sys
from pathlib import Path

from voice_to_fhir import Pipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="Process audio file to FHIR")
    parser.add_argument("audio_file", type=Path, help="Audio file to process")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    parser.add_argument("--workflow", "-w", default="general", help="Clinical workflow")
    parser.add_argument("--config", "-c", type=Path, help="Config file")
    args = parser.parse_args()

    if not args.audio_file.exists():
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)

    # Create pipeline
    if args.config:
        pipeline = Pipeline.from_config(args.config)
    else:
        pipeline = Pipeline.cloud()

    print(f"Processing: {args.audio_file}")
    print(f"Workflow: {args.workflow}")

    # Process file
    bundle = pipeline.process_file(args.audio_file, workflow=args.workflow)

    # Output
    json_output = pipeline.to_json(bundle, indent=2)

    if args.output:
        args.output.write_text(json_output)
        print(f"Output saved to: {args.output}")
    else:
        print(json_output)

    # Summary
    entry_count = len(bundle.get("entry", []))
    print(f"\n--- Generated {entry_count} FHIR resources ---")

    # Show resource types
    resource_types = {}
    for entry in bundle.get("entry", []):
        rt = entry.get("resource", {}).get("resourceType", "Unknown")
        resource_types[rt] = resource_types.get(rt, 0) + 1

    for rt, count in sorted(resource_types.items()):
        print(f"  {rt}: {count}")


if __name__ == "__main__":
    main()
