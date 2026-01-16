#!/usr/bin/env python3
"""
Model Download Script for Voice-to-FHIR Pipeline

Downloads MedASR and MedGemma models from HuggingFace Hub.
Requires HuggingFace token with access to HAI-DEF models.

Usage:
    export HF_TOKEN=your_token_here
    python scripts/download_models.py

    # Or with explicit token
    python scripts/download_models.py --token hf_xxxxx

    # Download specific models only
    python scripts/download_models.py --models medasr
    python scripts/download_models.py --models medgemma

Author: Cleansheet LLC
License: CC BY 4.0
"""

import argparse
import os
import sys
from pathlib import Path


def check_huggingface_hub() -> bool:
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub  # noqa: F401

        return True
    except ImportError:
        return False


def get_token(args_token: str | None) -> str | None:
    """Get HuggingFace token from args or environment."""
    if args_token:
        return args_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def download_medasr(token: str, output_dir: Path, quantized: bool = False) -> bool:
    """Download MedASR model."""
    from huggingface_hub import snapshot_download

    model_id = "google/medasr"
    target_dir = output_dir / "medasr"

    print(f"Downloading MedASR from {model_id}...")
    print(f"Target directory: {target_dir}")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            token=token,
            ignore_patterns=["*.md", "*.txt"] if not quantized else None,
        )
        print(f"MedASR downloaded successfully to {target_dir}")
        return True
    except Exception as e:
        print(f"Error downloading MedASR: {e}")
        return False


def download_medgemma(
    token: str, output_dir: Path, variant: str = "4b", quantized: bool = False
) -> bool:
    """Download MedGemma model."""
    from huggingface_hub import snapshot_download

    model_id = f"google/medgemma-{variant}"
    target_dir = output_dir / f"medgemma-{variant}"

    print(f"Downloading MedGemma from {model_id}...")
    print(f"Target directory: {target_dir}")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            token=token,
            ignore_patterns=["*.md", "*.txt"] if not quantized else None,
        )
        print(f"MedGemma downloaded successfully to {target_dir}")
        return True
    except Exception as e:
        print(f"Error downloading MedGemma: {e}")
        return False


def verify_hai_def_access(token: str) -> bool:
    """Verify that the token has access to HAI-DEF models."""
    from huggingface_hub import HfApi

    api = HfApi()

    print("Verifying HAI-DEF model access...")

    try:
        # Try to get model info (doesn't download)
        api.model_info("google/medasr", token=token)
        print("  MedASR access: OK")
    except Exception as e:
        print(f"  MedASR access: FAILED - {e}")
        print("\nYou may need to accept the HAI-DEF Terms of Use at:")
        print("https://developers.google.com/health-ai-developer-foundations")
        return False

    try:
        api.model_info("google/medgemma-4b", token=token)
        print("  MedGemma access: OK")
    except Exception as e:
        print(f"  MedGemma access: FAILED - {e}")
        return False

    return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download MedASR and MedGemma models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--token",
        "-t",
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("models"),
        help="Output directory for models (default: ./models)",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        choices=["medasr", "medgemma", "all"],
        default=["all"],
        help="Models to download (default: all)",
    )
    parser.add_argument(
        "--medgemma-variant",
        choices=["4b", "27b"],
        default="4b",
        help="MedGemma variant (default: 4b)",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Download quantized versions (smaller, for edge deployment)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify access, don't download",
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_huggingface_hub():
        print("Error: huggingface_hub not installed")
        print("Run: pip install huggingface_hub")
        return 2

    # Get token
    token = get_token(args.token)
    if not token:
        print("Error: HuggingFace token required")
        print("Set HF_TOKEN environment variable or use --token")
        return 2

    # Verify access
    if not verify_hai_def_access(token):
        return 1

    if args.verify_only:
        print("\nAccess verification complete.")
        return 0

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Determine which models to download
    models_to_download = args.models
    if "all" in models_to_download:
        models_to_download = ["medasr", "medgemma"]

    success = True

    # Download models
    if "medasr" in models_to_download:
        if not download_medasr(token, args.output, args.quantized):
            success = False

    if "medgemma" in models_to_download:
        if not download_medgemma(
            token, args.output, args.medgemma_variant, args.quantized
        ):
            success = False

    # Summary
    print("\n" + "=" * 50)
    if success:
        print("All models downloaded successfully!")
        print(f"Models are located in: {args.output.absolute()}")
    else:
        print("Some models failed to download. Check errors above.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
