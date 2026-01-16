"""
Clinical Workflow Prompts

Prompt templates for extracting clinical entities from transcripts.
Each workflow has specialized prompts optimized for its clinical context.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

AVAILABLE_WORKFLOWS = [
    "general",
    "emergency",
    "intake",
    "followup",
    "procedure",
    "discharge",
    "radiology",
    "lab_review",
    "respiratory",
    "icu",
    "cardiology",
    "pediatrics",
    "neurology",
]


def get_prompt_path(workflow: str) -> Path:
    """Get the path to a workflow's prompt file."""
    return PROMPTS_DIR / f"{workflow}.txt"


def load_prompt(workflow: str) -> str:
    """Load a workflow prompt template."""
    path = get_prompt_path(workflow)
    if not path.exists():
        raise ValueError(f"Unknown workflow: {workflow}. Available: {AVAILABLE_WORKFLOWS}")
    return path.read_text()


def list_workflows() -> list[str]:
    """List available workflows."""
    return AVAILABLE_WORKFLOWS.copy()
