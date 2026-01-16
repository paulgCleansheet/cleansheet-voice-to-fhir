#!/usr/bin/env python3
"""
Batch Processing Example

Demonstrates processing multiple audio files in batch.

Usage:
    python examples/batch_processing.py <input_dir> <output_dir> [--pattern "*.wav"]

Requirements:
    - HF_TOKEN environment variable set
    - Directory with audio files
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from voice_to_fhir import Pipeline, PipelineConfig


def process_file(pipeline: Pipeline, audio_file: Path, output_dir: Path, workflow: str):
    """Process a single file and save output."""
    try:
        bundle = pipeline.process_file(audio_file, workflow=workflow)

        output_file = output_dir / f"{audio_file.stem}.json"
        pipeline.save(bundle, output_file)

        entry_count = len(bundle.get("entry", []))
        return audio_file.name, entry_count, None
    except Exception as e:
        return audio_file.name, 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Batch process audio files to FHIR")
    parser.add_argument("input_dir", type=Path, help="Directory with audio files")
    parser.add_argument("output_dir", type=Path, help="Output directory for JSON files")
    parser.add_argument("--pattern", "-p", default="*.wav", help="File pattern to match")
    parser.add_argument("--workflow", "-w", default="general", help="Clinical workflow")
    parser.add_argument("--config", "-c", type=Path, help="Config file")
    parser.add_argument("--parallel", "-j", type=int, default=1,
                        help="Number of parallel workers")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find files
    files = list(args.input_dir.glob(args.pattern))
    if not files:
        print(f"No files matching '{args.pattern}' in {args.input_dir}")
        sys.exit(0)

    print("=" * 60)
    print("Voice-to-FHIR Batch Processing")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Workflow: {args.workflow}")
    print(f"Files found: {len(files)}")
    print(f"Parallel workers: {args.parallel}")
    print("-" * 60)

    # Create pipeline
    if args.config:
        pipeline = Pipeline.from_config(args.config)
    else:
        pipeline = Pipeline.cloud()

    # Process files
    success_count = 0
    error_count = 0
    total_resources = 0

    if args.parallel > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    process_file, pipeline, f, args.output_dir, args.workflow
                ): f for f in files
            }

            for future in as_completed(futures):
                filename, resources, error = future.result()
                if error:
                    print(f"  [ERROR] {filename}: {error}")
                    error_count += 1
                else:
                    print(f"  [OK] {filename}: {resources} resources")
                    success_count += 1
                    total_resources += resources
    else:
        # Sequential processing
        for i, audio_file in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] {audio_file.name}...", end=" ", flush=True)

            filename, resources, error = process_file(
                pipeline, audio_file, args.output_dir, args.workflow
            )

            if error:
                print(f"ERROR: {error}")
                error_count += 1
            else:
                print(f"OK ({resources} resources)")
                success_count += 1
                total_resources += resources

    # Summary
    print("-" * 60)
    print("Summary:")
    print(f"  Processed: {success_count}/{len(files)}")
    print(f"  Errors: {error_count}")
    print(f"  Total FHIR resources: {total_resources}")


if __name__ == "__main__":
    main()
