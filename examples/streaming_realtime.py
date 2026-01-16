#!/usr/bin/env python3
"""
Streaming Real-time Processing Example

Demonstrates real-time streaming processing with partial results.
Note: This is a preview of the streaming API - full implementation pending.

Usage:
    python examples/streaming_realtime.py

Requirements:
    - HF_TOKEN environment variable set
    - Microphone connected
"""

import signal
import sys

from voice_to_fhir import Pipeline, PipelineConfig


# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    global running
    print("\n\nStopping...")
    running = False


def main():
    global running

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Create pipeline
    pipeline = Pipeline.cloud()

    print("=" * 60)
    print("Real-time Streaming Processing")
    print("=" * 60)
    print()
    print("This example demonstrates the streaming API.")
    print("Speak continuously - partial transcripts and final FHIR")
    print("bundles will be output as they become available.")
    print()
    print("Press Ctrl+C to stop.")
    print("-" * 60)

    try:
        # Start real-time processing
        for result in pipeline.process_realtime(workflow="general"):
            if not running:
                break

            if result.get("_partial"):
                # Partial transcript (interim result)
                transcript = result.get("transcript", "")
                confidence = result.get("confidence", 0)
                print(f"\r[Partial] {transcript[:60]}... ({confidence:.0%})", end="", flush=True)
            else:
                # Final FHIR bundle
                print()  # New line after partials
                print("-" * 60)
                print("Final FHIR Bundle:")

                entry_count = len(result.get("entry", []))
                print(f"  Resources: {entry_count}")

                # Show resource types
                for entry in result.get("entry", []):
                    rt = entry.get("resource", {}).get("resourceType", "Unknown")
                    rid = entry.get("resource", {}).get("id", "")
                    print(f"    - {rt}/{rid}")

                print("-" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print("\nStreaming stopped.")


if __name__ == "__main__":
    main()
