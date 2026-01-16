#!/usr/bin/env python3
"""
Live Audio Capture Example

Demonstrates real-time audio capture from microphone with VAD.

Usage:
    python examples/live_capture.py [--duration 30]

Requirements:
    - HF_TOKEN environment variable set
    - Microphone connected
    - pip install sounddevice
"""

import argparse

from voice_to_fhir import Pipeline, PipelineConfig
from voice_to_fhir.capture import AudioCapture


def list_devices():
    """List available audio input devices."""
    devices = AudioCapture.list_devices()

    if not devices:
        print("No audio input devices found")
        return

    print("Available audio input devices:")
    print("-" * 50)
    for device in devices:
        print(f"  [{device['index']}] {device['name']}")
        print(f"      Channels: {device['channels']}, Sample Rate: {device['sample_rate']} Hz")
    print()


def main():
    parser = argparse.ArgumentParser(description="Live audio capture to FHIR")
    parser.add_argument("--duration", "-d", type=float, default=30.0,
                        help="Max recording duration in seconds")
    parser.add_argument("--workflow", "-w", default="general",
                        help="Clinical workflow")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices and exit")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Create pipeline
    pipeline = Pipeline.cloud()

    print("=" * 60)
    print("Voice-to-FHIR Live Capture")
    print("=" * 60)
    print(f"Workflow: {args.workflow}")
    print(f"Max duration: {args.duration}s")
    print()
    print("Speak now. Recording will stop after silence is detected.")
    print("Press Ctrl+C to stop early.")
    print("-" * 60)

    try:
        # Capture and process
        bundle = pipeline.capture_and_process(
            max_duration_seconds=args.duration,
            workflow=args.workflow,
        )

        print("-" * 60)
        print("Processing complete!")
        print()

        # Output
        json_output = pipeline.to_json(bundle, indent=2)

        if args.output:
            with open(args.output, "w") as f:
                f.write(json_output)
            print(f"Output saved to: {args.output}")
        else:
            print(json_output)

        # Summary
        entry_count = len(bundle.get("entry", []))
        print(f"\n--- Generated {entry_count} FHIR resources ---")

    except KeyboardInterrupt:
        print("\n\nCapture cancelled by user.")


if __name__ == "__main__":
    main()
