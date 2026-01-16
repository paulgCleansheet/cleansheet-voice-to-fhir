"""
Tests for extraction prompts.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

import pytest
from pathlib import Path

from voice_to_fhir.extraction.prompts import (
    AVAILABLE_WORKFLOWS,
    PROMPTS_DIR,
    get_prompt_path,
    load_prompt,
    list_workflows,
)


class TestPromptsModule:
    """Tests for prompts module."""

    def test_available_workflows_list(self):
        """Test that available workflows are defined."""
        assert len(AVAILABLE_WORKFLOWS) > 0
        assert "general" in AVAILABLE_WORKFLOWS
        assert "emergency" in AVAILABLE_WORKFLOWS

    def test_prompts_dir_exists(self):
        """Test that prompts directory exists."""
        assert PROMPTS_DIR.exists()
        assert PROMPTS_DIR.is_dir()

    def test_list_workflows(self):
        """Test list_workflows function."""
        workflows = list_workflows()

        assert isinstance(workflows, list)
        assert "general" in workflows
        assert workflows == AVAILABLE_WORKFLOWS

    def test_get_prompt_path(self):
        """Test get_prompt_path function."""
        path = get_prompt_path("general")

        assert isinstance(path, Path)
        assert path.name == "general.txt"

    def test_load_prompt_general(self):
        """Test loading general workflow prompt."""
        prompt = load_prompt("general")

        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial
        assert "{transcript}" in prompt
        assert "conditions" in prompt.lower()
        assert "medications" in prompt.lower()

    def test_load_prompt_emergency(self):
        """Test loading emergency workflow prompt."""
        prompt = load_prompt("emergency")

        assert isinstance(prompt, str)
        assert "{transcript}" in prompt
        # Emergency-specific content
        assert "triage" in prompt.lower() or "critical" in prompt.lower()
        assert "acuity" in prompt.lower() or "urgent" in prompt.lower()

    def test_load_prompt_intake(self):
        """Test loading intake workflow prompt."""
        prompt = load_prompt("intake")

        assert isinstance(prompt, str)
        assert "{transcript}" in prompt
        # Intake-specific content
        assert "medical_history" in prompt.lower() or "history" in prompt.lower()
        assert "social" in prompt.lower() or "family" in prompt.lower()

    def test_load_prompt_followup(self):
        """Test loading followup workflow prompt."""
        prompt = load_prompt("followup")

        assert isinstance(prompt, str)
        assert "{transcript}" in prompt
        # Followup-specific content
        assert "progress" in prompt.lower()

    def test_load_prompt_procedure(self):
        """Test loading procedure workflow prompt."""
        prompt = load_prompt("procedure")

        assert isinstance(prompt, str)
        assert "{transcript}" in prompt
        # Procedure-specific content
        assert "technique" in prompt.lower() or "procedure" in prompt.lower()
        assert "specimen" in prompt.lower() or "findings" in prompt.lower()

    def test_load_prompt_discharge(self):
        """Test loading discharge workflow prompt."""
        prompt = load_prompt("discharge")

        assert isinstance(prompt, str)
        assert "{transcript}" in prompt
        # Discharge-specific content
        assert "discharge" in prompt.lower()
        assert "follow" in prompt.lower() or "medication" in prompt.lower()

    def test_load_prompt_radiology(self):
        """Test loading radiology workflow prompt."""
        prompt = load_prompt("radiology")

        assert isinstance(prompt, str)
        assert "{transcript}" in prompt
        # Radiology-specific content
        assert "modality" in prompt.lower() or "imaging" in prompt.lower()
        assert "findings" in prompt.lower()

    def test_load_prompt_lab_review(self):
        """Test loading lab_review workflow prompt."""
        prompt = load_prompt("lab_review")

        assert isinstance(prompt, str)
        assert "{transcript}" in prompt
        # Lab-specific content
        assert "result" in prompt.lower() or "lab" in prompt.lower()
        assert "interpretation" in prompt.lower() or "critical" in prompt.lower()

    def test_load_prompt_unknown_raises(self):
        """Test that unknown workflow raises error."""
        with pytest.raises(ValueError) as exc_info:
            load_prompt("unknown_workflow")

        assert "Unknown workflow" in str(exc_info.value)
        assert "unknown_workflow" in str(exc_info.value)

    def test_all_workflows_have_prompts(self):
        """Test that all listed workflows have prompt files."""
        for workflow in AVAILABLE_WORKFLOWS:
            path = get_prompt_path(workflow)
            assert path.exists(), f"Prompt file missing for workflow: {workflow}"

    def test_prompts_are_valid_templates(self):
        """Test that all prompts have the transcript placeholder."""
        for workflow in AVAILABLE_WORKFLOWS:
            prompt = load_prompt(workflow)
            assert "{transcript}" in prompt, f"Missing {{transcript}} in {workflow} prompt"

    def test_prompts_request_json_output(self):
        """Test that prompts instruct JSON output."""
        for workflow in AVAILABLE_WORKFLOWS:
            prompt = load_prompt(workflow)
            assert "json" in prompt.lower(), f"No JSON instruction in {workflow} prompt"

    def test_prompts_have_output_schema(self):
        """Test that prompts include output schema."""
        for workflow in AVAILABLE_WORKFLOWS:
            prompt = load_prompt(workflow)
            # Should have either ```json or { in the schema definition
            has_schema = "```json" in prompt or '"conditions"' in prompt
            assert has_schema, f"No output schema in {workflow} prompt"


class TestPromptContent:
    """Tests for specific prompt content requirements."""

    def test_general_has_all_entity_types(self):
        """Test general prompt covers all entity types."""
        prompt = load_prompt("general")

        entity_types = [
            "conditions",
            "medications",
            "allergies",
            "vitals",
            "patient",
        ]

        for entity_type in entity_types:
            assert entity_type in prompt.lower(), f"Missing {entity_type} in general prompt"

    def test_emergency_has_triage_fields(self):
        """Test emergency prompt has triage-specific fields."""
        prompt = load_prompt("emergency")

        triage_fields = ["chief_complaint", "acuity", "is_critical"]
        found = sum(1 for field in triage_fields if field in prompt)
        assert found >= 2, "Emergency prompt missing triage fields"

    def test_discharge_has_reconciliation(self):
        """Test discharge prompt has medication reconciliation."""
        prompt = load_prompt("discharge")

        assert "reconciliation" in prompt.lower() or "new_medications" in prompt.lower()
        assert "discontinued" in prompt.lower() or "stopped" in prompt.lower()

    def test_procedure_has_cpt_field(self):
        """Test procedure prompt includes CPT code field."""
        prompt = load_prompt("procedure")

        assert "cpt" in prompt.lower()

    def test_prompts_have_icd10_where_appropriate(self):
        """Test that diagnosis-focused prompts include ICD-10."""
        diagnosis_workflows = ["general", "emergency", "discharge", "intake"]

        for workflow in diagnosis_workflows:
            prompt = load_prompt(workflow)
            assert "icd" in prompt.lower(), f"Missing ICD reference in {workflow}"

    def test_lab_review_has_loinc(self):
        """Test lab review prompt includes LOINC codes."""
        prompt = load_prompt("lab_review")

        assert "loinc" in prompt.lower()
